import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import re
import hashlib

from PIL import Image, ImageOps, ImageSequence
import torch

import folder_paths
from nodes import MAX_RESOLUTION

from .saver.saver import save_image
from .utils import get_sha256, full_checkpoint_path_for
from .utils_civitai import get_civitai_sampler_name, get_civitai_metadata, MAX_HASH_LENGTH
from .prompt_metadata_extractor import PromptMetadataExtractor

def parse_checkpoint_name(ckpt_name: str) -> str:
    return os.path.basename(ckpt_name)

def parse_checkpoint_name_without_extension(ckpt_name: str) -> str:
    return os.path.splitext(parse_checkpoint_name(ckpt_name))[0]

def get_timestamp(time_format: str) -> str:
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp

def save_json(image_info: dict[str, Any] | None, filename: str) -> None:
    try:
        workflow = (image_info or {}).get('workflow')
        if workflow is None:
            print('No image info found, skipping saving of JSON')
        with open(f'{filename}.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file)
            print(f'Saved workflow to {filename}.json')
    except Exception as e:
        print(f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution')

def make_pathname(filename: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler_name: str, denoise: float, clip_skip: int) -> str:
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%model", parse_checkpoint_name(modelname))
    filename = filename.replace("%width", str(width))
    filename = filename.replace("%height", str(height))
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    filename = filename.replace("%sampler_name", sampler_name)
    filename = filename.replace("%steps", str(steps))
    filename = filename.replace("%cfg", str(cfg))
    filename = filename.replace("%scheduler_name", scheduler_name)
    filename = filename.replace("%basemodelname", parse_checkpoint_name_without_extension(modelname))
    filename = filename.replace("%denoise", str(denoise))
    filename = filename.replace("%clip_skip", str(clip_skip))
    return filename

def make_filename(filename: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler_name: str, denoise: float, clip_skip: int) -> str:
    filename = make_pathname(filename, width, height, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler_name, denoise, clip_skip)
    return get_timestamp(time_format) if filename == "" else filename

@dataclass
class Metadata:
    modelname: str
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler_name: str
    denoise: float
    clip_skip: int
    additional_hashes: str
    ckpt_path: str
    a111_params: str
    final_hashes: str

class LoadImageWithMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }

    CATEGORY = "ImageSaver"
    DESCRIPTION = "Load image and extract metadata for Image Saver"
    RETURN_TYPES = ("IMAGE", "METADATA")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            
            if len(output_images) == 0:
                w, h = image.size
                
            if image.size != (w, h):
                continue
                
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Extract metadata from image
        metadata = LoadImageWithMetadata.extract_metadata_from_image(img, image_path)
        
        return (output_image, metadata)

    @staticmethod
    def extract_metadata_from_image(img: Image.Image, image_path: str) -> Metadata:
        """Extract metadata from image and return Metadata object."""
        print(f"DEBUG: LoadImageWithMetadata - Extracting metadata from: {image_path}")
        
        # Default values
        positive = ""
        negative = ""
        width = img.width
        height = img.height
        seed = 0
        steps = 0
        cfg = 0
        sampler_name = ""
        scheduler_name = ""
        denoise = 1.0
        clip_skip = 0
        modelname = ""
        a111_params = ""
        
        try:
            # Try to get metadata from various sources
            metadata_found = False
            
            # Check PNG info (most common for AI-generated images)
            if hasattr(img, 'text') and img.text:
                print(f"DEBUG: LoadImageWithMetadata - Found PNG text metadata")
                for key, value in img.text.items():
                    print(f"DEBUG: LoadImageWithMetadata - PNG text key: {key}")
                    if key.lower() in ['parameters', 'params', 'generation_params']:
                        print(f"DEBUG: LoadImageWithMetadata - Found parameters in PNG text")
                        a111_params = value
                        metadata_found = True
                        break
            
            # Check EXIF data
            if not metadata_found and hasattr(img, '_getexif') and img._getexif():
                print(f"DEBUG: LoadImageWithMetadata - Checking EXIF data")
                exif = img._getexif()
                # Common EXIF tags that might contain AI generation parameters
                for tag_id, value in exif.items():
                    if isinstance(value, str) and ('Steps:' in value or 'Sampler:' in value):
                        print(f"DEBUG: LoadImageWithMetadata - Found parameters in EXIF tag {tag_id}")
                        a111_params = value
                        metadata_found = True
                        break
            
            # Check image info (PIL)
            if not metadata_found and hasattr(img, 'info') and img.info:
                print(f"DEBUG: LoadImageWithMetadata - Checking PIL info")
                for key, value in img.info.items():
                    print(f"DEBUG: LoadImageWithMetadata - PIL info key: {key}")
                    if isinstance(value, str) and ('Steps:' in value or 'Sampler:' in value):
                        print(f"DEBUG: LoadImageWithMetadata - Found parameters in PIL info")
                        a111_params = value
                        metadata_found = True
                        break
                    elif key.lower() in ['parameters', 'params', 'generation_params']:
                        print(f"DEBUG: LoadImageWithMetadata - Found parameters key in PIL info")
                        a111_params = str(value)
                        metadata_found = True
                        break
            
            # If we found A1111-style parameters, parse them
            if metadata_found and a111_params:
                print(f"DEBUG: LoadImageWithMetadata - Parsing A1111 parameters, length: {len(a111_params)}")
                parsed_metadata = ImageSaverSimple.parse_a1111_params(a111_params)
                
                # Use parsed values
                positive = parsed_metadata.positive
                negative = parsed_metadata.negative
                width = parsed_metadata.width
                height = parsed_metadata.height
                seed = parsed_metadata.seed
                steps = parsed_metadata.steps
                cfg = parsed_metadata.cfg
                sampler_name = parsed_metadata.sampler_name
                scheduler_name = parsed_metadata.scheduler_name
                denoise = parsed_metadata.denoise
                clip_skip = parsed_metadata.clip_skip
                modelname = parsed_metadata.modelname
                
                print(f"DEBUG: LoadImageWithMetadata - Successfully parsed metadata")
            else:
                print(f"DEBUG: LoadImageWithMetadata - No A1111-style parameters found, using defaults")
                # Use image dimensions if no metadata found
                width = img.width
                height = img.height
                a111_params = f"unknown\nNegative prompt: unknown\nSteps: {steps}, Sampler: {sampler_name or 'unknown'}, CFG scale: {cfg}, Seed: {seed}, Size: {width}x{height}, Model: {modelname or 'unknown'}, Version: ComfyUI"
                
        except Exception as e:
            print(f"DEBUG: LoadImageWithMetadata - Error extracting metadata: {e}")
            # Use defaults and image dimensions
            width = img.width
            height = img.height
            a111_params = f"unknown\nNegative prompt: unknown\nSteps: {steps}, Sampler: {sampler_name or 'unknown'}, CFG scale: {cfg}, Seed: {seed}, Size: {width}x{height}, Model: {modelname or 'unknown'}, Version: ComfyUI"
        
        # Create and return metadata object
        metadata = Metadata(
            modelname=modelname,
            positive=positive,
            negative=negative,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            denoise=denoise,
            clip_skip=clip_skip,
            additional_hashes="",
            ckpt_path="",
            a111_params=a111_params,
            final_hashes=""
        )
        
        print(f"DEBUG: LoadImageWithMetadata - Created metadata object with:")
        print(f"  positive: {repr(positive[:100])}...")
        print(f"  negative: {repr(negative[:100])}...")
        print(f"  size: {width}x{height}")
        print(f"  model: {repr(modelname)}")
        
        return metadata

    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

class ImageSaverMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "optional": {
                "modelname":             ("STRING",  {"default": '', "multiline": False,                           "tooltip": "model name (can be multiple, separated by commas)"}),
                "positive":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "positive prompt"}),
                "negative":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "negative prompt"}),
                "width":                 ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image width"}),
                "height":                ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image height"}),
                "seed_value":            ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "seed"}),
                "steps":                 ("INT",     {"default": 20, "min": 1, "max": 10000,                       "tooltip": "number of steps"}),
                "cfg":                   ("FLOAT",   {"default": 7.0, "min": 0.0, "max": 100.0,                    "tooltip": "CFG value"}),
                "sampler_name":          ("STRING",  {"default": '', "multiline": False,                           "tooltip": "sampler name (as string)"}),
                "scheduler_name":        ("STRING",  {"default": 'normal', "multiline": False,                     "tooltip": "scheduler name (as string)"}),
                "denoise":               ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,                      "tooltip": "denoise value"}),
                "clip_skip":             ("INT",     {"default": 0, "min": -24, "max": 24,                         "tooltip": "skip last CLIP layers (positive or negative value, 0 for no skip)"}),
                "additional_hashes":     ("STRING",  {"default": "", "multiline": False,                           "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')\nWith download_civitai_data set to true, weights can be added as well. (e.g., 'HASH:Weight', 'Name:HASH:Weight')"}),
                "download_civitai_data": ("BOOLEAN", {"default": True,                                             "tooltip": "Download and cache data from civitai.com to save correct metadata. Allows LoRA weights to be saved to the metadata."}),
                "easy_remix":            ("BOOLEAN", {"default": True,                                             "tooltip": "Strip LoRAs and simplify 'embedding:path' from the prompt to make the Remix option on civitai.com more seamless."}),
            },
        }

    RETURN_TYPES = ("METADATA","STRING","STRING")
    RETURN_NAMES = ("metadata","hashes","a1111_params")
    OUTPUT_TOOLTIPS = ("metadata for Image Saver Simple","Comma-separated list of the hashes to chain with other Image Saver additional_hashes","Written parameters to the image metadata")
    FUNCTION = "get_metadata"
    CATEGORY = "ImageSaver"
    DESCRIPTION = "Prepare metadata for Image Saver Simple"

    def get_metadata(
        self,
        modelname: str = "",
        positive: str = "unknown",
        negative: str = "unknown",
        width: int = 512,
        height: int = 512,
        seed_value: int = 0,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = "",
        scheduler_name: str = "",
        denoise: float = 1.0,
        clip_skip: int = 0,
        additional_hashes: str = "",
        download_civitai_data: bool = True,
        easy_remix: bool = True,
    ) -> tuple[Metadata, str, str]:
        metadata = ImageSaverMetadata.make_metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, additional_hashes, download_civitai_data, easy_remix)
        return (metadata, metadata.final_hashes, metadata.a111_params)

    @staticmethod
    def make_metadata(modelname: str, positive: str, negative: str, width: int, height: int, seed_value: int, steps: int, cfg: float, sampler_name: str, scheduler_name: str, denoise: float, clip_skip: int, additional_hashes: str, download_civitai_data: bool, easy_remix: bool) -> Metadata:
        modelname, additional_hashes = ImageSaver.get_multiple_models(modelname, additional_hashes)

        ckpt_path = full_checkpoint_path_for(modelname)
        if ckpt_path:
            modelhash = get_sha256(ckpt_path)[:10]
        else:
            modelhash = ""

        metadata_extractor = PromptMetadataExtractor([positive, negative])
        embeddings = metadata_extractor.get_embeddings()
        loras = metadata_extractor.get_loras()
        civitai_sampler_name = get_civitai_sampler_name(sampler_name.replace('_gpu', ''), scheduler_name)
        basemodelname = parse_checkpoint_name_without_extension(modelname)

        # Get existing hashes from model, loras, and embeddings
        existing_hashes = {modelhash.lower()} | {t[2].lower() for t in loras.values()} | {t[2].lower() for t in embeddings.values()}
        # Parse manual hashes
        manual_entries = ImageSaver.parse_manual_hashes(additional_hashes, existing_hashes, download_civitai_data)
        # Get Civitai metadata
        civitai_resources, hashes, add_model_hash = get_civitai_metadata(modelname, ckpt_path, modelhash, loras, embeddings, manual_entries, download_civitai_data)

        if easy_remix:
            positive = ImageSaver.clean_prompt(positive, metadata_extractor)
            negative = ImageSaver.clean_prompt(negative, metadata_extractor)

        positive_a111_params = positive.strip()
        negative_a111_params = f"\nNegative prompt: {negative.strip()}"
        clip_skip_str = f", Clip skip: {abs(clip_skip)}" if clip_skip != 0 else ""
        model_hash_str = f", Model hash: {add_model_hash}" if add_model_hash else ""
        hashes_str = f", Hashes: {json.dumps(hashes, separators=(',', ':'))}" if hashes else ""

        a111_params = (
            f"{positive_a111_params}{negative_a111_params}\n"
            f"Steps: {steps}, Sampler: {civitai_sampler_name}, CFG scale: {cfg}, Seed: {seed_value}, "
            f"Size: {width}x{height}{clip_skip_str}{model_hash_str}, Model: {basemodelname}{hashes_str}, Version: ComfyUI"
        )

        # Add Civitai resource listing
        if download_civitai_data and civitai_resources:
            a111_params += f", Civitai resources: {json.dumps(civitai_resources, separators=(',', ':'))}"

        final_hashes = ",".join(f"{Path(name.split(':')[-1]).stem + ':' if name else ''}{hash}{':' + str(weight) if weight is not None and download_civitai_data else ''}" for name, (_, weight, hash) in ({ modelname: ( ckpt_path, None, modelhash ) } | loras | embeddings | manual_entries).items())

        metadata = Metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, additional_hashes, ckpt_path, a111_params, final_hashes)
        return metadata

class ImageSaverSimple:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images":                ("IMAGE",    {                                                             "tooltip": "image(s) to save"}),
                "filename":              ("STRING",   {"default": '%time_%basemodelname_%seed', "multiline": False, "tooltip": "filename (available variables: %date, %time, %model, %width, %height, %seed, %counter, %sampler_name, %steps, %cfg, %scheduler_name, %basemodelname, %denoise, %clip_skip)"}),
                "path":                  ("STRING",   {"default": '', "multiline": False,                           "tooltip": "path to save the images (under Comfy's save directory)"}),
                "extension":             (['png', 'jpeg', 'jpg', 'webp'], {                                         "tooltip": "file extension/type to save image as"}),
                "lossless_webp":         ("BOOLEAN",  {"default": True,                                             "tooltip": "if True, saved WEBP files will be lossless"}),
                "quality_jpeg_or_webp":  ("INT",      {"default": 100, "min": 1, "max": 100,                        "tooltip": "quality setting of JPEG/WEBP"}),
                "optimize_png":          ("BOOLEAN",  {"default": False,                                            "tooltip": "if True, saved PNG files will be optimized (can reduce file size but is slower)"}),
                "embed_workflow":        ("BOOLEAN",  {"default": True,                                             "tooltip": "if True, embeds the workflow in the saved image files.\nStable for PNG, experimental for WEBP.\nJPEG experimental and only if metadata size is below 65535 bytes"}),
                "save_workflow_as_json": ("BOOLEAN",  {"default": False,                                            "tooltip": "if True, also saves the workflow as a separate JSON file"}),
            },
            "optional": {
                "metadata":              ("METADATA", {"default": None,                                             "tooltip": "metadata to embed in the image"}),
                "a1111_params":          ("STRING",   {"default": "", "multiline": True,                           "tooltip": "A1111-style parameters string to parse metadata from (only used if metadata input is not connected)"}),
                "counter":               ("INT",      {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "counter"}),
                "time_format":           ("STRING",   {"default": "%Y-%m-%d-%H%M%S", "multiline": False,            "tooltip": "timestamp format"}),
                "show_preview":          ("BOOLEAN",  {"default": True,                                             "tooltip": "if True, displays saved images in the UI preview"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("hashes","a1111_params")
    OUTPUT_TOOLTIPS = ("Comma-separated list of the hashes to chain with other Image Saver additional_hashes","Written parameters to the image metadata")
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver"
    DESCRIPTION = "Save images with civitai-compatible generation metadata"

    def save_images(self,
        images: list[torch.Tensor],
        filename: str,
        path: str,
        extension: str,
        lossless_webp: bool,
        quality_jpeg_or_webp: int,
        optimize_png: bool,
        embed_workflow: bool = True,
        save_workflow_as_json: bool = False,
        show_preview: bool = True,
        metadata: Metadata | None = None,
        a1111_params: str = "",
        counter: int = 0,
        time_format: str = "%Y-%m-%d-%H%M%S",
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if metadata is None:
            if a1111_params.strip():
                metadata = ImageSaverSimple.parse_a1111_params(a1111_params)
            else:
                metadata = Metadata('', '', '', 512, 512, 0, 20, 7.0, '', 'normal', 1.0, 0, '', '', '', '')

        path = make_pathname(path, metadata.width, metadata.height, metadata.seed, metadata.modelname, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.clip_skip)

        filenames = ImageSaver.save_images(images, filename, extension, path, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow, counter, time_format, metadata)

        subfolder = os.path.normpath(path)

        result: dict[str, Any] = {
            "result": (metadata.final_hashes, metadata.a111_params),
        }

        if show_preview:
            result["ui"] = {"images": [{"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'} for filename in filenames]}

        return result

    @staticmethod
    def parse_a1111_params(a1111_params: str) -> Metadata:
        """Parse A1111-style parameters string into Metadata object."""
        # Default values
        positive = "unknown"
        negative = "unknown"
        width = 512
        height = 512
        seed = 0
        steps = 20
        cfg = 7.0
        sampler_name = ""
        scheduler_name = "normal"
        denoise = 1.0
        clip_skip = 0
        modelname = ""
        
        try:
            # Split by lines and process
            lines = a1111_params.strip().split('\n')
            
            # First line is usually the positive prompt
            if lines:
                positive = lines[0].strip()
            
            # Look for negative prompt
            for i, line in enumerate(lines):
                if line.strip().startswith("Negative prompt:"):
                    negative = line.replace("Negative prompt:", "").strip()
                    break
            
            # Find the parameters line (usually starts with "Steps:")
            params_line = ""
            for line in lines:
                if line.strip().startswith("Steps:"):
                    params_line = line.strip()
                    break
            
            if params_line:
                # Parse parameters using regex patterns
                import re
                
                # Steps
                steps_match = re.search(r'Steps:\s*(\d+)', params_line)
                if steps_match:
                    steps = int(steps_match.group(1))
                
                # Sampler
                sampler_match = re.search(r'Sampler:\s*([^,]+)', params_line)
                if sampler_match:
                    sampler_name = sampler_match.group(1).strip()
                
                # CFG scale
                cfg_match = re.search(r'CFG scale:\s*([\d.]+)', params_line)
                if cfg_match:
                    cfg = float(cfg_match.group(1))
                
                # Seed
                seed_match = re.search(r'Seed:\s*(\d+)', params_line)
                if seed_match:
                    seed = int(seed_match.group(1))
                
                # Size
                size_match = re.search(r'Size:\s*(\d+)x(\d+)', params_line)
                if size_match:
                    width = int(size_match.group(1))
                    height = int(size_match.group(2))
                
                # Clip skip
                clip_skip_match = re.search(r'Clip skip:\s*(\d+)', params_line)
                if clip_skip_match:
                    clip_skip = int(clip_skip_match.group(1))
                
                # Model
                model_match = re.search(r'Model:\s*([^,]+)', params_line)
                if model_match:
                    modelname = model_match.group(1).strip()
                
                # Denoise (if present)
                denoise_match = re.search(r'Denoising strength:\s*([\d.]+)', params_line)
                if denoise_match:
                    denoise = float(denoise_match.group(1))
                
        except Exception as e:
            print(f"Error parsing A1111 params: {e}")
        
        # Create metadata object with parsed values
        # The a111_params field should contain the original input string
        metadata = Metadata(
            modelname=modelname,
            positive=positive,
            negative=negative,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            denoise=denoise,
            clip_skip=clip_skip,
            additional_hashes="",
            ckpt_path="",
            a111_params=a1111_params,
            final_hashes=""
        )
        
        return metadata

class ImageSaver:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images":                ("IMAGE",   {                                                             "tooltip": "image(s) to save"}),
                "filename":              ("STRING",  {"default": '%time_%basemodelname_%seed', "multiline": False, "tooltip": "filename (available variables: %date, %time, %model, %width, %height, %seed, %counter, %sampler_name, %steps, %cfg, %scheduler, %basemodelname, %denoise, %clip_skip)"}),
                "path":                  ("STRING",  {"default": '', "multiline": False,                           "tooltip": "path to save the images (under Comfy's save directory)"}),
                "extension":             (['png', 'jpeg', 'jpg', 'webp'], {                                        "tooltip": "file extension/type to save image as"}),
            },
            "optional": {
                "steps":                 ("INT",     {"default": 20, "min": 1, "max": 10000,                       "tooltip": "number of steps"}),
                "cfg":                   ("FLOAT",   {"default": 7.0, "min": 0.0, "max": 100.0,                    "tooltip": "CFG value"}),
                "modelname":             ("STRING",  {"default": '', "multiline": False,                           "tooltip": "model name (can be multiple, separated by commas)"}),
                "sampler_name":          ("STRING",  {"default": '', "multiline": False,                           "tooltip": "sampler name (as string)"}),
                "scheduler_name":        ("STRING",  {"default": 'normal', "multiline": False,                     "tooltip": "scheduler name (as string)"}),
                "positive":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "positive prompt"}),
                "negative":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "negative prompt"}),
                "seed_value":            ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "seed"}),
                "width":                 ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image width"}),
                "height":                ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image height"}),
                "lossless_webp":         ("BOOLEAN", {"default": True,                                             "tooltip": "if True, saved WEBP files will be lossless"}),
                "quality_jpeg_or_webp":  ("INT",     {"default": 100, "min": 1, "max": 100,                        "tooltip": "quality setting of JPEG/WEBP"}),
                "optimize_png":          ("BOOLEAN", {"default": False,                                            "tooltip": "if True, saved PNG files will be optimized (can reduce file size but is slower)"}),
                "counter":               ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "counter"}),
                "denoise":               ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,                      "tooltip": "denoise value"}),
                "clip_skip":             ("INT",     {"default": 0, "min": -24, "max": 24,                         "tooltip": "skip last CLIP layers (positive or negative value, 0 for no skip)"}),
                "time_format":           ("STRING",  {"default": "%Y-%m-%d-%H%M%S", "multiline": False,            "tooltip": "timestamp format"}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False,                                            "tooltip": "if True, also saves the workflow as a separate JSON file"}),
                "embed_workflow":        ("BOOLEAN", {"default": True,                                             "tooltip": "if True, embeds the workflow in the saved image files.\nStable for PNG, experimental for WEBP.\nJPEG experimental and only if metadata size is below 65535 bytes"}),
                "additional_hashes":     ("STRING",  {"default": "", "multiline": False,                           "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')\nWith download_civitai_data set to true, weights can be added as well. (e.g., 'HASH:Weight', 'Name:HASH:Weight')"}),
                "download_civitai_data": ("BOOLEAN", {"default": True,                                             "tooltip": "Download and cache data from civitai.com to save correct metadata. Allows LoRA weights to be saved to the metadata."}),
                "easy_remix":            ("BOOLEAN", {"default": True,                                             "tooltip": "Strip LoRAs and simplify 'embedding:path' from the prompt to make the Remix option on civitai.com more seamless."}),
                "show_preview":          ("BOOLEAN", {"default": True,                                             "tooltip": "if True, displays saved images in the UI preview"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("hashes","a1111_params")
    OUTPUT_TOOLTIPS = ("Comma-separated list of the hashes to chain with other Image Saver additional_hashes","Written parameters to the image metadata")
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver"
    DESCRIPTION = "Save images with civitai-compatible generation metadata"

    def save_files(
        self,
        images: list[torch.Tensor],
        filename: str,
        path: str,
        extension: str,
        steps: int = 20,
        cfg: float = 7.0,
        modelname: str = "",
        sampler_name: str = "",
        scheduler_name: str = "normal",
        positive: str = "unknown",
        negative: str = "unknown",
        seed_value: int = 0,
        width: int = 512,
        height: int = 512,
        lossless_webp: bool = True,
        quality_jpeg_or_webp: int = 100,
        optimize_png: bool = False,
        counter: int = 0,
        denoise: float = 1.0,
        clip_skip: int = 0,
        time_format: str = "%Y-%m-%d-%H%M%S",
        save_workflow_as_json: bool = False,
        embed_workflow: bool = True,
        additional_hashes: str = "",
        download_civitai_data: bool = True,
        easy_remix: bool = True,
        show_preview: bool = True,
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = ImageSaverMetadata.make_metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, additional_hashes, download_civitai_data, easy_remix)

        path = make_pathname(path, metadata.width, metadata.height, metadata.seed, metadata.modelname, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.clip_skip)

        filenames = ImageSaver.save_images(images, filename, extension, path, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json, embed_workflow, counter, time_format, metadata)

        subfolder = os.path.normpath(path)

        result: dict[str, Any] = {
            "result": (metadata.final_hashes, metadata.a111_params),
        }

        if show_preview:
            result["ui"] = {"images": [{"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'} for filename in filenames]}

        return result

    @staticmethod
    def save_images(
        images: list[torch.Tensor],
        filename_pattern: str,
        extension: str,
        path: str,
        quality_jpeg_or_webp: int,
        lossless_webp: bool,
        optimize_png: bool,
        prompt: dict[str, Any] | None,
        extra_pnginfo: dict[str, Any] | None,
        save_workflow_as_json: bool,
        embed_workflow: bool,
        counter: int,
        time_format: str,
        metadata: Metadata
    ) -> list[str]:
        filename_prefix = make_filename(filename_pattern, metadata.width, metadata.height, metadata.seed, metadata.modelname, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.clip_skip)

        output_path = os.path.join(folder_paths.output_directory, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        result_paths: list[str] = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            current_filename_prefix = ImageSaver.get_unique_filename(output_path, filename_prefix, extension)
            final_filename = f"{current_filename_prefix}.{extension}"
            filepath = os.path.join(output_path, final_filename)

            save_image(img, filepath, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, metadata.a111_params, prompt, extra_pnginfo, embed_workflow)

            if save_workflow_as_json:
                save_json(extra_pnginfo, os.path.join(output_path, current_filename_prefix))

            result_paths.append(final_filename)
        return result_paths

    # Match 'anything' or 'anything:anything' with trimmed white space
    re_manual_hash = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?\s*$')
    # Match 'anything', 'anything:anything' or 'anything:anything:number' with trimmed white space
    re_manual_hash_weights = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?(?:\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$')

    @staticmethod
    def get_multiple_models(modelname: str, additional_hashes: str) -> tuple[str, str]:
        model_names = [m.strip() for m in modelname.split(',')]
        modelname = model_names[0] # Use the first model as the primary one

        # Process additional model names and add to additional_hashes
        for additional_model in model_names[1:]:
            additional_ckpt_path = full_checkpoint_path_for(additional_model)
            if additional_ckpt_path:
                additional_modelhash = get_sha256(additional_ckpt_path)[:10]
                # Add to additional_hashes in "name:HASH" format
                if additional_hashes:
                    additional_hashes += ","
                additional_hashes += f"{additional_model}:{additional_modelhash}"
        return modelname, additional_hashes

    @staticmethod
    def parse_manual_hashes(additional_hashes: str, existing_hashes: set[str], download_civitai_data: bool) -> dict[str, tuple[str | None, float | None, str]]:
        """Process additional_hashes input (a string) by normalizing, removing extra spaces/newlines, and splitting by comma"""
        manual_entries: dict[str, tuple[str | None, float | None, str]] = {}
        unnamed_count = 0

        additional_hash_split = additional_hashes.replace("\n", ",").split(",") if additional_hashes else []
        for entry in additional_hash_split:
            match = (ImageSaver.re_manual_hash_weights if download_civitai_data else ImageSaver.re_manual_hash).search(entry)
            if match is None:
                print(f"ComfyUI-Image-Saver: Invalid additional hash string: '{entry}'")
                continue

            groups = tuple(group for group in match.groups() if group)

            # Read weight and remove from groups, if needed
            weight = None
            if download_civitai_data and len(groups) > 1:
                try:
                    weight = float(groups[-1])
                    groups = groups[:-1]
                except (ValueError, TypeError):
                    pass

            # Read hash, optionally preceded by name
            name, hash = groups if len(groups) > 1 else (None, groups[0])

            if len(hash) > MAX_HASH_LENGTH:
                print(f"ComfyUI-Image-Saver: Skipping hash. Length exceeds maximum of {MAX_HASH_LENGTH} characters: {hash}")
                continue

            if any(hash.lower() == existing_hash.lower() for _, _, existing_hash in manual_entries.values()):
                print(f"ComfyUI-Image-Saver: Skipping duplicate hash: {hash}")
                continue  # Skip duplicates

            if hash.lower() in existing_hashes:
                print(f"ComfyUI-Image-Saver: Skipping manual hash already present in resources: {hash}")
                continue

            if name is None:
                unnamed_count += 1
                name = f"manual{unnamed_count}"
            elif name in manual_entries:
                print(f"ComfyUI-Image-Saver: Duplicate manual hash name '{name}' is being overwritten.")

            manual_entries[name] = (None, weight, hash)

            if len(manual_entries) > 29:
                print("ComfyUI-Image-Saver: Reached maximum limit of 30 manual hashes. Skipping the rest.")
                break

        return manual_entries

    @staticmethod
    def clean_prompt(prompt: str, metadata_extractor: PromptMetadataExtractor) -> str:
        """Clean prompts for easier remixing by removing LoRAs and simplifying embeddings."""
        # Strip loras
        prompt = re.sub(metadata_extractor.LORA, "", prompt)
        # Shorten 'embedding:path/to/my_embedding' -> 'my_embedding'
        # Note: Possible inaccurate embedding name if the filename has been renamed from the default
        prompt = re.sub(metadata_extractor.EMBEDDING, lambda match: Path(match.group(1)).stem, prompt)
        # Remove prompt control edits. e.g., 'STYLE(A1111, mean)', 'SHIFT(1)`, etc.`
        prompt = re.sub(r'\b[A-Z]+\([^)]*\)', "", prompt)
        return prompt

    @staticmethod
    def get_unique_filename(output_path: str, filename_prefix: str, extension: str) -> str:
        existing_files = [f for f in os.listdir(output_path) if f.startswith(filename_prefix) and f.endswith(extension)]

        if not existing_files:
            return f"{filename_prefix}"

        suffixes: list[int] = []
        for f in existing_files:
            name, _ = os.path.splitext(f)
            parts = name.split('_')
            if parts[-1].isdigit():
                suffixes.append(int(parts[-1]))

        if suffixes:
            next_suffix = max(suffixes) + 1
        else:
            next_suffix = 1

        return f"{filename_prefix}_{next_suffix:02d}"

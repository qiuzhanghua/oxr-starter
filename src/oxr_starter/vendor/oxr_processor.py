"""OXR processor for the vLLM 0.25.1 plugin."""

import math
from functools import cached_property
from typing import Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import AutoProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
MIN_PIXELS = 200704
MAX_PIXELS = 921600


class OXRProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
        },
        "videos_kwargs": {
            "do_convert_rgb": True,
        },
    }


class OXRProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_pad_token"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_pad_token=None,
        patch_size=16,
        hidden_stride=2,
        temporal_patch_size=1,
        **kwargs,
    ):
        self.image_token = IMAGE_TOKEN
        self.video_token = VIDEO_TOKEN
        self.image_pad_token = image_pad_token or "<|image_pad|>"
        self.patch_size = patch_size
        self.hidden_stride = hidden_stride
        self.temporal_patch_size = temporal_patch_size
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @cached_property
    def extra_special_tokens(self):
        image_pad_token_id = self.tokenizer.get_vocab()[self.image_pad_token]
        return {
            "image_token": -200,
            "video_token": -201,
            "visual_atom": -300,
            "image_start": -301,
            "image_end": -302,
            "video_start": -303,
            "video_end": -304,
            "image_pad": image_pad_token_id,
        }

    def __call__(
        self,
        images: ImageInput = None,
        videos: Union[np.ndarray, list[ImageInput]] = None,
        text: Union[
            TextInput,
            PreTokenizedInput,
            list[TextInput],
            list[PreTokenizedInput],
        ] = None,
        **kwargs: Unpack[OXRProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            OXRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        visual_features = {}
        output = BatchFeature()

        if images is not None:
            processed_images = []
            image_placeholders_list = []
            grids = []
            for image in images if isinstance(images, list) else [images]:
                pixel_values, image_placeholders, grid = self.preprocess_multidata(
                    images=image,
                    **output_kwargs["images_kwargs"],
                )
                processed_images.append(pixel_values)
                image_placeholders_list.append(image_placeholders)
                grids.append(grid)

            if processed_images:
                visual_features["image_placeholders"] = image_placeholders_list
            output["pixel_values"] = processed_images
            output["grids"] = grids

        if videos is not None:
            processed_videos = []
            video_placeholders_list = []
            grids = []
            for video in videos if isinstance(videos, list) else [videos]:
                pixel_values, video_placeholders, grid = self.preprocess_multidata(
                    video=video,
                    **output_kwargs["videos_kwargs"],
                )
                processed_videos.append(pixel_values)
                video_placeholders_list.append(video_placeholders)
                grids.append(grid)

            if processed_videos:
                visual_features["video_placeholders"] = video_placeholders_list
            output["video_pixel_values"] = processed_videos
            output["video_grids"] = grids

        if text is None:
            return BatchFeature(data=visual_features)

        if not isinstance(text, list):
            text = [text]
        tokenized_batched_text = self._tokenize_with_visual_symbol(text)
        image_token_id = self.get_token_value("image_token")
        video_token_id = self.get_token_value("video_token")
        replaced_ids_list = []
        image_idx = 0
        video_idx = 0
        for ids_tensor in tokenized_batched_text:
            has_image_tokens = (
                image_token_id in ids_tensor
                and "image_placeholders" in visual_features
                and image_idx < len(visual_features["image_placeholders"])
            )
            has_video_tokens = (
                video_token_id in ids_tensor
                and "video_placeholders" in visual_features
                and video_idx < len(visual_features["video_placeholders"])
            )
            if has_image_tokens or has_video_tokens:
                new_ids = []
                for token_id in ids_tensor.tolist():
                    if token_id == image_token_id:
                        new_ids.extend(visual_features["image_placeholders"][image_idx])
                        image_idx += 1
                    elif token_id == video_token_id:
                        new_ids.extend(visual_features["video_placeholders"][video_idx])
                        video_idx += 1
                    else:
                        new_ids.append(token_id)
                ids_tensor = torch.tensor(new_ids, dtype=torch.long)
            replaced_ids_list.append(ids_tensor)

        if replaced_ids_list:
            output["input_ids"] = torch.stack(replaced_ids_list)
        else:
            output["input_ids"] = torch.tensor([], dtype=torch.long)
        return output

    def _tokenize_with_visual_symbol(self, text_list: list[str]) -> torch.LongTensor:
        batch_token_ids = []
        for text in text_list:
            token_ids = []
            video_token_id = self.get_token_value("video_token")
            image_token_id = self.get_token_value("image_token")
            video_split_texts = text.split(self.video_token)

            for j, video_segment in enumerate(video_split_texts):
                image_split_texts = video_segment.split(self.image_token)
                text_chunks = [
                    self.tokenizer(chunk, add_special_tokens=False).input_ids
                    for chunk in image_split_texts
                ]
                segment_tokens = []
                for i, chunk in enumerate(text_chunks):
                    segment_tokens.extend(chunk)
                    if i < len(text_chunks) - 1:
                        segment_tokens.append(image_token_id)
                token_ids.extend(segment_tokens)
                if j < len(video_split_texts) - 1:
                    token_ids.append(video_token_id)
            batch_token_ids.append(token_ids)
        return torch.tensor(batch_token_ids, dtype=torch.long)

    def smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 32,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
    ):
        if height < factor or width < factor:
            if height < width:
                width = round(factor / height * width)
                height = factor
            else:
                height = round(factor / width * height)
                width = factor
        elif max(height, width) / min(height, width) > 200:
            if height > width:
                height = 200 * width
            else:
                width = 200 * height

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def get_token_value(self, token_name: str):
        return self.extra_special_tokens[token_name]

    def construct_visual_indicators(self, grid, is_video: bool = False):
        if is_video:
            start_token = self.get_token_value("video_start")
            end_token = self.get_token_value("video_end")
        else:
            start_token = self.get_token_value("image_start")
            end_token = self.get_token_value("image_end")

        visual_indicators = [start_token, self.get_token_value("visual_atom")]
        if grid[0] * grid[1] > 1:
            for _ in range(grid[0]):
                for _ in range(grid[1]):
                    visual_indicators.append(self.get_token_value("visual_atom"))
        visual_indicators.append(end_token)
        return visual_indicators

    def construct_visual_placeholders(self, grid, is_video: bool = False):
        visual_placeholders = self.construct_visual_indicators((1, 1), is_video)
        visual_atom_token_id = self.get_token_value("visual_atom")
        image_padding_token_id = self.get_token_value("image_pad")

        num_image_atoms = grid[0] * grid[1] * grid[2]
        num_image_atoms //= self.hidden_stride**2
        num_image_atoms //= self.temporal_patch_size

        vision_start_token_id = self.tokenizer.get_vocab().get(
            "<|vision_start|>",
            image_padding_token_id,
        )
        vision_end_token_id = self.tokenizer.get_vocab().get(
            "<|vision_end|>",
            image_padding_token_id,
        )

        padded_placeholder_tokens = []
        for token in visual_placeholders:
            if token == visual_atom_token_id:
                padded_placeholder_tokens.extend(
                    [image_padding_token_id] * num_image_atoms
                )
            elif token in (
                self.get_token_value("image_start"),
                self.get_token_value("video_start"),
            ):
                padded_placeholder_tokens.append(vision_start_token_id)
            elif token in (
                self.get_token_value("image_end"),
                self.get_token_value("video_end"),
            ):
                padded_placeholder_tokens.append(vision_end_token_id)
            else:
                padded_placeholder_tokens.append(image_padding_token_id)
        return padded_placeholder_tokens

    def preprocess_multidata(
        self,
        images: Optional[Union[PIL.Image.Image, list[PIL.Image.Image]]] = None,
        video: Optional[Union[list[PIL.Image.Image], np.ndarray]] = None,
        do_convert_rgb: Optional[bool] = True,
        convert_to_rgb: Optional[bool] = None,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        return_tensors: Optional[str] = "pt",
    ):
        if convert_to_rgb is not None:
            do_convert_rgb = convert_to_rgb

        is_video = False
        if images is not None:
            if not isinstance(images, list):
                images = [images]
        elif video is not None:
            is_video = True
            if isinstance(video, np.ndarray):
                images = [
                    PIL.Image.fromarray(video[i].astype(np.uint8))
                    for i in range(video.shape[0])
                ]
            elif isinstance(video, list):
                images = video
        else:
            raise ValueError("Expected either images or video")

        min_pixels = min(
            max_pixels if max_pixels is not None else MAX_PIXELS,
            min_pixels if min_pixels is not None else MIN_PIXELS,
        )
        images = [
            image.convert("RGB") if do_convert_rgb and image.mode != "RGB" else image
            for image in images
        ]

        width, height = images[0].size
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=self.patch_size * self.hidden_stride,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image_pt = self.image_processor.preprocess(
                image,
                size=dict(height=resized_height, width=resized_width),
                return_tensors="np",
            )["pixel_values"][0]
            processed_images.append(image_pt)

        patches = np.array(processed_images)
        frame_remainder = patches.shape[0] % self.temporal_patch_size
        if frame_remainder != 0:
            num_to_pad = self.temporal_patch_size - frame_remainder
            repeats = np.repeat(patches[-1][np.newaxis], num_to_pad, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.hidden_stride,
            self.hidden_stride,
            self.patch_size,
            grid_w // self.hidden_stride,
            self.hidden_stride,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        visual_placeholders = self.construct_visual_placeholders(
            [grid_t, grid_h, grid_w],
            is_video,
        )
        return (
            torch.tensor(flatten_patches),
            visual_placeholders,
            torch.tensor([[grid_t, grid_h, grid_w]]),
        )


AutoProcessor.register("OXRProcessor", OXRProcessor)


# Standard library imports
import atexit
import gc
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import cache
from itertools import repeat
from json import JSONDecodeError
from json import dumps as json_dumps
from shutil import copy2
from json import load as json_load
from math import cos, pi  # For smooth fade effect
from multiprocessing import Process
from multiprocessing import Queue as multiprocessing_Queue
from multiprocessing import freeze_support as multiprocessing_freeze_support
from multiprocessing.pool import ThreadPool
from os import O_CREAT, O_WRONLY
from os import cpu_count as os_cpu_count
from os import devnull as os_devnull
from os import fdopen as os_fdopen
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os import open as os_open
from os import remove as os_remove
from os import sep as os_separator
from os.path import abspath as os_path_abspath
from os.path import basename as os_path_basename
from os.path import dirname as os_path_dirname
from os.path import exists as os_path_exists
from os.path import expanduser as os_path_expanduser
from os.path import getsize as os_path_getsize
from os.path import join as os_path_join
from os.path import splitext as os_path_splitext
from pathlib import Path
from shutil import move as shutil_move
from shutil import rmtree as remove_directory
from subprocess import CalledProcessError
from subprocess import run as subprocess_run
from threading import Event, Lock, RLock, Thread
from time import sleep
from timeit import default_timer as timer
# GUI imports
from tkinter import DISABLED, StringVar
from typing import Any, Callable, Dict, List, Optional, Union
from webbrowser import open as open_browser

from customtkinter import (CTk, CTkButton, CTkEntry, CTkFont, CTkFrame,
                           CTkImage, CTkLabel, CTkOptionMenu,
                           CTkScrollableFrame, CTkToplevel, filedialog,
                           set_appearance_mode, set_default_color_theme)
from cv2 import (CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT,
                 CAP_PROP_FRAME_WIDTH, COLOR_BGR2RGB, COLOR_BGR2RGBA,
                 COLOR_GRAY2RGB, COLOR_RGB2GRAY, IMREAD_UNCHANGED, INTER_AREA,
                 INTER_CUBIC)
from cv2 import VideoCapture as opencv_VideoCapture
from cv2 import addWeighted as opencv_addWeighted
from cv2 import cvtColor as opencv_cvtColor
from cv2 import imdecode as opencv_imdecode
from cv2 import imencode as opencv_imencode
from cv2 import resize as opencv_resize
# Third-party library imports
from natsort import natsorted
from numpy import ascontiguousarray as numpy_ascontiguousarray
from numpy import clip as numpy_clip
from numpy import concatenate as numpy_concatenate
from numpy import expand_dims as numpy_expand_dims
from numpy import float32
from numpy import frombuffer as numpy_frombuffer
from numpy import full as numpy_full
from numpy import max as numpy_max
from numpy import mean as numpy_mean
from numpy import ndarray as numpy_ndarray
from numpy import repeat as numpy_repeat
from numpy import squeeze as numpy_squeeze
from numpy import transpose as numpy_transpose
from numpy import uint8
from numpy import zeros as numpy_zeros
from onnxruntime import InferenceSession
from PIL.Image import fromarray as pillow_image_fromarray
from PIL.Image import open as pillow_image_open

# Define supported file extensions
supported_image_extensions = [".jpg", ".jpeg",
                              ".png", ".bmp", ".tiff", ".tif", ".webp"]
supported_video_extensions = [".mp4", ".avi",
                              ".mkv", ".mov", ".wmv", ".flv", ".webm"]
supported_file_extensions = supported_image_extensions + supported_video_extensions

if sys.stdout is None:
    sys.stdout = open(os_devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os_devnull, "w")


def find_by_relative_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(
        os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)


app_name = "Warlock-Studio"
version = "2.2"

background_color = "#000000"  # Negro grisáceo profundo
app_name_color = "#FF0000"  # Blanco puro para el nombre de la app
widget_background_color = "#5A5A5A"  # Rojo oscuro (Dark Red)
text_color = "#F4F4F4"  # Blanco opaco para texto legible

VRAM_model_usage = {
    'RealESR_Gx4':     2.2,
    'RealESR_Animex4': 2.2,
    'RealESRNetx4':   2.2,
    'BSRGANx4':        0.6,
    'BSRGANx2':        0.7,
    'RealESRGANx4':    0.6,
    'IRCNN_Mx1':       4,
    'IRCNN_Lx1':       4,
}

MENU_LIST_SEPARATOR = ["----"]
SRVGGNetCompact_models_list = ["RealESR_Gx4", "RealESR_Animex4"]
BSRGAN_models_list = ["BSRGANx4", "BSRGANx2", "RealESRGANx4", "RealESRNetx4"]
IRCNN_models_list = ["IRCNN_Mx1", "IRCNN_Lx1"]
RIFE_models_list = ["RIFE", "RIFE_Lite"]

AI_models_list = (SRVGGNetCompact_models_list + MENU_LIST_SEPARATOR + BSRGAN_models_list +
                  MENU_LIST_SEPARATOR + IRCNN_models_list + MENU_LIST_SEPARATOR + RIFE_models_list)
frame_interpolation_models_list = RIFE_models_list
frame_generation_options_list = [
    "x2", "x4", "x8", "Slowmotion x2", "Slowmotion x4", "Slowmotion x8"
]
AI_multithreading_list = ["OFF", "2 threads",
                          "4 threads", "6 threads", "8 threads"]
blending_list = ["OFF", "Low", "Medium", "High"]
gpus_list = ["Auto", "GPU 1", "GPU 2", "GPU 3", "GPU 4"]
keep_frames_list = ["OFF", "ON"]
image_extension_list = [".png", ".jpg", ".bmp", ".tiff"]
video_extension_list = [".mp4", ".mkv", ".avi", ".mov"]
video_codec_list = [
    "x264",       "x265",       MENU_LIST_SEPARATOR[0],
    "h264_nvenc", "hevc_nvenc", MENU_LIST_SEPARATOR[0],
    "h264_amf",   "hevc_amf",   MENU_LIST_SEPARATOR[0],
    "h264_qsv",   "hevc_qsv",
]
# -- FluidFrames: Integrate conditional interpolation option --

OUTPUT_PATH_CODED = "Same path as input files"
DOCUMENT_PATH = os_path_join(os_path_expanduser('~'), 'Documents')
USER_PREFERENCE_PATH = find_by_relative_path(
    f"{DOCUMENT_PATH}{os_separator}{app_name}_{version}_UserPreference.json")
FFMPEG_EXE_PATH = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")
EXIFTOOL_EXE_PATH = find_by_relative_path(f"Assets{os_separator}exiftool.exe")

ECTRACTION_FRAMES_FOR_CPU = 30
MULTIPLE_FRAMES_TO_SAVE = 8

COMPLETED_STATUS = "Completed"
ERROR_STATUS = "Error"
STOP_STATUS = "Stop"

if os_path_exists(FFMPEG_EXE_PATH):
    print(f"[{app_name}] ffmpeg.exe found")
else:
    print(f"[{app_name}] ffmpeg.exe not found, please install ffmpeg.exe following the guide")

if os_path_exists(USER_PREFERENCE_PATH):
    print(f"[{app_name}] Preference file exist")
    with open(USER_PREFERENCE_PATH, "r") as json_file:
        json_data = json_load(json_file)
        default_AI_model = json_data.get(
            "default_AI_model",             AI_models_list[0])
        default_AI_multithreading = json_data.get(
            "default_AI_multithreading",    AI_multithreading_list[0])
        default_gpu = json_data.get(
            "default_gpu",                  gpus_list[0])
        default_keep_frames = json_data.get(
            "default_keep_frames",          keep_frames_list[1])
        default_image_extension = json_data.get(
            "default_image_extension",      image_extension_list[0])
        default_video_extension = json_data.get(
            "default_video_extension",      video_extension_list[0])
        default_video_codec = json_data.get(
            "default_video_codec",          video_codec_list[0])
        default_blending = json_data.get(
            "default_blending",             blending_list[1])
        default_output_path = json_data.get(
            "default_output_path",          OUTPUT_PATH_CODED)
        default_input_resize_factor = json_data.get(
            "default_input_resize_factor",  str(50))
        default_output_resize_factor = json_data.get(
            "default_output_resize_factor", str(100))
        default_VRAM_limiter = json_data.get(
            "default_VRAM_limiter",         str(4))

else:
    print(f"[{app_name}] Preference file does not exist, using default coded value")
    default_AI_model = AI_models_list[0]
    default_AI_multithreading = AI_multithreading_list[0]
    default_gpu = gpus_list[0]
    default_keep_frames = keep_frames_list[1]
    default_image_extension = image_extension_list[0]
    default_video_extension = video_extension_list[0]
    default_video_codec = video_codec_list[0]
    default_blending = blending_list[1]
    default_output_path = OUTPUT_PATH_CODED
    default_input_resize_factor = str(50)
    default_output_resize_factor = str(100)
    default_VRAM_limiter = str(4)

offset_y_options = 0.0825
row1 = 0.125
row2 = row1 + offset_y_options
row3 = row2 + offset_y_options
row4 = row3 + offset_y_options
row5 = row4 + offset_y_options
row6 = row5 + offset_y_options
row7 = row6 + offset_y_options
row8 = row7 + offset_y_options
row9 = row8 + offset_y_options
row10 = row9 + offset_y_options

column_offset = 0.2
column_info1 = 0.625
column_info2 = 0.858
column_1 = 0.66
column_2 = column_1 + column_offset
column_1_5 = column_info1 + 0.08
column_1_4 = column_1_5 - 0.0127
column_3 = column_info2 + 0.08
column_2_9 = column_3 - 0.0127
column_3_5 = column_2 + 0.0355

little_textbox_width = 74
little_menu_width = 98


# Remove duplicate definitions - using the ones defined earlier


# AI -------------------

class AI_upscale:

    # CLASS INIT FUNCTIONS

    def __init__(
            self,
            AI_model_name: str,
            directml_gpu: str,
            input_resize_factor: int,
            output_resize_factor: int,
            max_resolution: int
    ):

        # Passed variables
        self.AI_model_name = AI_model_name
        self.directml_gpu = directml_gpu
        self.input_resize_factor = input_resize_factor
        self.output_resize_factor = output_resize_factor
        self.max_resolution = max_resolution

        # Calculated variables
        self.AI_model_path = find_by_relative_path(
            f"AI-onnx{os_separator}{self.AI_model_name}_fp16.onnx")
        self.upscale_factor = self._get_upscale_factor()
        self.inferenceSession = None

    def _get_upscale_factor(self) -> int:
        if "x1" in self.AI_model_name:
            return 1
        elif "x2" in self.AI_model_name:
            return 2
        elif "x4" in self.AI_model_name:
            return 4

    def _load_inferenceSession(self) -> None:
        try:
            # Check if model file exists
            if not os_path_exists(self.AI_model_path):
                raise FileNotFoundError(
                    f"AI model file not found: {self.AI_model_path}")

            providers = ['DmlExecutionProvider']

            match self.directml_gpu:
                case 'Auto':  provider_options = [{"performance_preference": "high_performance"}]
                case 'GPU 1': provider_options = [{"device_id": "0"}]
                case 'GPU 2': provider_options = [{"device_id": "1"}]
                case 'GPU 3': provider_options = [{"device_id": "2"}]
                case 'GPU 4': provider_options = [{"device_id": "3"}]

            inference_session = InferenceSession(
                path_or_bytes=self.AI_model_path,
                providers=providers,
                provider_options=provider_options,
            )

            self.inferenceSession = inference_session
            print(
                f"[AI] Successfully loaded model: {os_path_basename(self.AI_model_path)}")

        except Exception as e:
            error_msg = f"Failed to load AI model {os_path_basename(self.AI_model_path)}: {str(e)}"
            print(f"[AI ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    # INTERNAL CLASS FUNCTIONS

    def get_image_mode(self, image: numpy_ndarray) -> str:
        if image is None:
            raise ValueError("Image is None")
        shape = image.shape
        if len(shape) == 2:  # Grayscale: 2D array (rows, cols)
            return "Grayscale"
        # RGB: 3D array with 3 channels
        elif len(shape) == 3 and shape[2] == 3:
            return "RGB"
        # RGBA: 3D array with 4 channels
        elif len(shape) == 3 and shape[2] == 4:
            return "RGBA"
        else:
            raise ValueError(f"Unsupported image shape: {shape}")

    def get_image_resolution(self, image: numpy_ndarray) -> tuple:
        height = image.shape[0]
        width = image.shape[1]

        return height, width

    def calculate_target_resolution(self, image: numpy_ndarray) -> tuple:
        height, width = self.get_image_resolution(image)
        target_height = height * self.upscale_factor
        target_width = width * self.upscale_factor

        return target_height, target_width

    def resize_with_input_factor(self, image: numpy_ndarray) -> numpy_ndarray:

        old_height, old_width = self.get_image_resolution(image)

        new_width = int(old_width * self.input_resize_factor)
        new_height = int(old_height * self.input_resize_factor)

        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        if self.input_resize_factor > 1:
            return opencv_resize(image, (new_width, new_height), interpolation=INTER_CUBIC)
        elif self.input_resize_factor < 1:
            return opencv_resize(image, (new_width, new_height), interpolation=INTER_AREA)
        else:
            return image

    def resize_with_output_factor(self, image: numpy_ndarray) -> numpy_ndarray:

        old_height, old_width = self.get_image_resolution(image)

        new_width = int(old_width * self.output_resize_factor)
        new_height = int(old_height * self.output_resize_factor)

        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        if self.output_resize_factor > 1:
            return opencv_resize(image, (new_width, new_height), interpolation=INTER_CUBIC)
        elif self.output_resize_factor < 1:
            return opencv_resize(image, (new_width, new_height), interpolation=INTER_AREA)
        else:
            return image

    # VIDEO CLASS FUNCTIONS

    def calculate_multiframes_supported_by_gpu(self, video_frame_path: str) -> int:
        resized_video_frame = self.resize_with_input_factor(
            image_read(video_frame_path))
        height, width = self.get_image_resolution(resized_video_frame)
        image_pixels = height * width
        max_supported_pixels = self.max_resolution * self.max_resolution

        frames_simultaneously = max_supported_pixels // image_pixels

        print(
            f" Frames supported simultaneously by GPU: {frames_simultaneously}")

        return frames_simultaneously

    # TILLING FUNCTIONS

    def image_need_tilling(self, image: numpy_ndarray) -> bool:
        height, width = self.get_image_resolution(image)
        image_pixels = height * width
        max_supported_pixels = self.max_resolution * self.max_resolution

        if image_pixels > max_supported_pixels:
            return True
        else:
            return False

    def add_alpha_channel(self, image: numpy_ndarray) -> numpy_ndarray:
        if image.shape[2] == 3:
            alpha = numpy_full(
                (image.shape[0], image.shape[1], 1), 255, dtype=uint8)
            image = numpy_concatenate((image, alpha), axis=2)
        return image

    def calculate_tiles_number(self, image: numpy_ndarray) -> tuple:

        height, width = self.get_image_resolution(image)

        tiles_x = (width + self.max_resolution - 1) // self.max_resolution
        tiles_y = (height + self.max_resolution - 1) // self.max_resolution

        return tiles_x, tiles_y

    def split_image_into_tiles(self, image: numpy_ndarray, tiles_x: int, tiles_y: int) -> list[numpy_ndarray]:

        img_height, img_width = self.get_image_resolution(image)

        tile_width = img_width // tiles_x
        tile_height = img_height // tiles_y

        tiles = []

        for y in range(tiles_y):
            y_start = y * tile_height
            y_end = (y + 1) * tile_height

            for x in range(tiles_x):
                x_start = x * tile_width
                x_end = (x + 1) * tile_width
                tile = image[y_start:y_end, x_start:x_end]
                tiles.append(tile)

        return tiles

    def combine_tiles_into_image(self, image: numpy_ndarray, tiles: list[numpy_ndarray], t_height: int, t_width: int, num_tiles_x: int) -> numpy_ndarray:

        match self.get_image_mode(image):
            case "Grayscale": tiled_image = numpy_zeros((t_height, t_width, 3), dtype=uint8)
            case "RGB":       tiled_image = numpy_zeros((t_height, t_width, 3), dtype=uint8)
            case "RGBA":      tiled_image = numpy_zeros((t_height, t_width, 4), dtype=uint8)
            # Default fallback
            case _:           tiled_image = numpy_zeros((t_height, t_width, 3), dtype=uint8)

        for tile_index in range(len(tiles)):
            actual_tile = tiles[tile_index]

            tile_height, tile_width = self.get_image_resolution(actual_tile)

            row = tile_index // num_tiles_x
            col = tile_index % num_tiles_x
            y_start = row * tile_height
            y_end = y_start + tile_height
            x_start = col * tile_width
            x_end = x_start + tile_width

            match self.get_image_mode(image):
                case "Grayscale": tiled_image[y_start:y_end, x_start:x_end] = actual_tile
                case "RGB":       tiled_image[y_start:y_end, x_start:x_end] = actual_tile
                case "RGBA":      tiled_image[y_start:y_end, x_start:x_end] = self.add_alpha_channel(actual_tile)
                # Default fallback
                case _:           tiled_image[y_start:y_end, x_start:x_end] = actual_tile

        return tiled_image

    # AI CLASS FUNCTIONS

    def normalize_image(self, image: numpy_ndarray) -> tuple:
        range = 255
        if numpy_max(image) > 256:
            range = 65535
        normalized_image = image / range

        return normalized_image, range

    def preprocess_image(self, image: numpy_ndarray) -> numpy_ndarray:
        image = numpy_transpose(image, (2, 0, 1))
        image = numpy_expand_dims(image, axis=0)

        return image

    def onnxruntime_inference(self, image: numpy_ndarray) -> numpy_ndarray:

        # IO BINDING
        # io_binding = self.inferenceSession.io_binding()
        # io_binding.bind_cpu_input(self.inferenceSession.get_inputs()[0].name, image.astype(float16))
        # io_binding.bind_output(self.inferenceSession.get_outputs()[0].name)
        # self.inferenceSession.run_with_iobinding(io_binding)
        # onnx_output = io_binding.copy_outputs_to_cpu()[0]

        onnx_input = {self.inferenceSession.get_inputs()[0].name: image}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]

        return onnx_output

    def postprocess_output(self, onnx_output: numpy_ndarray) -> numpy_ndarray:
        onnx_output = numpy_squeeze(onnx_output, axis=0)
        onnx_output = numpy_clip(onnx_output, 0, 1)
        onnx_output = numpy_transpose(onnx_output, (1, 2, 0))

        return onnx_output

    def de_normalize_image(self, onnx_output: numpy_ndarray, max_range: int) -> numpy_ndarray:
        match max_range:
            case 255: return (onnx_output * max_range).astype(uint8)
            case 65535: return (onnx_output * max_range).round().astype(float32)
            # Default fallback to 255
            case _: return (onnx_output * 255).astype(uint8)

    def AI_upscale(self, image: numpy_ndarray) -> numpy_ndarray:
        image = image.astype(float32)
        image_mode = self.get_image_mode(image)
        image, range = self.normalize_image(image)

        match image_mode:
            case "RGB":
                image = self.preprocess_image(image)
                onnx_output = self.onnxruntime_inference(image)
                onnx_output = self.postprocess_output(onnx_output)
                output_image = self.de_normalize_image(onnx_output, range)

                return output_image

            case "RGBA":
                alpha = image[:, :, 3]
                image = image[:, :, :3]
                image = opencv_cvtColor(image, COLOR_BGR2RGB)

                image = image.astype(float32)
                alpha = alpha.astype(float32)

                # Image
                image = self.preprocess_image(image)
                onnx_output_image = self.onnxruntime_inference(image)
                onnx_output_image = self.postprocess_output(onnx_output_image)
                onnx_output_image = opencv_cvtColor(
                    onnx_output_image, COLOR_BGR2RGBA)

                # Alpha
                alpha = numpy_expand_dims(alpha, axis=-1)
                alpha = numpy_repeat(alpha, 3, axis=-1)
                alpha = self.preprocess_image(alpha)
                onnx_output_alpha = self.onnxruntime_inference(alpha)
                onnx_output_alpha = self.postprocess_output(onnx_output_alpha)
                onnx_output_alpha = opencv_cvtColor(
                    onnx_output_alpha, COLOR_RGB2GRAY)

                # Fusion Image + Alpha
                onnx_output_image[:, :, 3] = onnx_output_alpha
                output_image = self.de_normalize_image(
                    onnx_output_image, range)

                return output_image

            case "Grayscale":
                image = opencv_cvtColor(image, COLOR_GRAY2RGB)

                image = self.preprocess_image(image)
                onnx_output = self.onnxruntime_inference(image)
                onnx_output = self.postprocess_output(onnx_output)
                output_image = opencv_cvtColor(onnx_output, COLOR_RGB2GRAY)
                output_image = self.de_normalize_image(onnx_output, range)

                return output_image

    def AI_upscale_with_tilling(self, image: numpy_ndarray) -> numpy_ndarray:
        t_height, t_width = self.calculate_target_resolution(image)
        tiles_x, tiles_y = self.calculate_tiles_number(image)
        tiles_list = self.split_image_into_tiles(image, tiles_x, tiles_y)
        tiles_list = [self.AI_upscale(tile) for tile in tiles_list]

        return self.combine_tiles_into_image(image, tiles_list, t_height, t_width, tiles_x)

    # EXTERNAL FUNCTION

    def AI_orchestration(self, image: numpy_ndarray) -> numpy_ndarray:

        if self.inferenceSession == None:
            self._load_inferenceSession()

        resized_image = self.resize_with_input_factor(image)

        if self.image_need_tilling(resized_image):
            upscaled_image = self.AI_upscale_with_tilling(resized_image)
        else:
            upscaled_image = self.AI_upscale(resized_image)

        return self.resize_with_output_factor(upscaled_image)

# AI INTERPOLATION for frame generation -----------------


class AI_interpolation:

    # CLASS INIT FUNCTIONS

    def __init__(
            self,
            AI_model_name: str,
            frame_gen_factor: int,
            directml_gpu: str,
            input_resize_factor: int,
            output_resize_factor: int,
    ):

        # Passed variables
        self.AI_model_name = AI_model_name
        self.frame_gen_factor = frame_gen_factor
        self.directml_gpu = directml_gpu
        self.input_resize_factor = input_resize_factor
        self.output_resize_factor = output_resize_factor

        # Calculated variables
        self.AI_model_path = find_by_relative_path(
            f"AI-onnx{os_separator}{self.AI_model_name}_fp32.onnx")
        self.inferenceSession = self._load_inferenceSession()

    def _load_inferenceSession(self) -> InferenceSession:
        try:
            # Check if model file exists
            if not os_path_exists(self.AI_model_path):
                raise FileNotFoundError(
                    f"AI model file not found: {self.AI_model_path}")

            providers = ['DmlExecutionProvider']

            match self.directml_gpu:
                case 'Auto':        provider_options = [{"performance_preference": "high_performance"}]
                case 'GPU 1':       provider_options = [{"device_id": "0"}]
                case 'GPU 2':       provider_options = [{"device_id": "1"}]
                case 'GPU 3':       provider_options = [{"device_id": "2"}]
                case 'GPU 4':       provider_options = [{"device_id": "3"}]

            inference_session = InferenceSession(
                path_or_bytes=self.AI_model_path,
                providers=providers,
                provider_options=provider_options
            )

            print(
                f"[AI] Successfully loaded interpolation model: {os_path_basename(self.AI_model_path)}")
            return inference_session

        except Exception as e:
            error_msg = f"Failed to load AI interpolation model {os_path_basename(self.AI_model_path)}: {str(e)}"
            print(f"[AI ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    # INTERNAL CLASS FUNCTIONS

    def get_image_mode(self, image: numpy_ndarray) -> str:
        if image is None:
            raise ValueError("Image is None")
        shape = image.shape
        if len(shape) == 2:  # Grayscale: 2D array (rows, cols)
            return "Grayscale"
        # RGB: 3D array with 3 channels
        elif len(shape) == 3 and shape[2] == 3:
            return "RGB"
        # RGBA: 3D array with 4 channels
        elif len(shape) == 3 and shape[2] == 4:
            return "RGBA"
        else:
            raise ValueError(f"Unsupported image shape: {shape}")

    def get_image_resolution(self, image: numpy_ndarray) -> tuple:
        height = image.shape[0]
        width = image.shape[1]

        return height, width

    def resize_with_input_factor(self, image: numpy_ndarray) -> numpy_ndarray:

        old_height, old_width = self.get_image_resolution(image)

        new_width = int(old_width * self.input_resize_factor)
        new_height = int(old_height * self.input_resize_factor)

        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        if self.input_resize_factor > 1:
            return opencv_resize(image, (new_width, new_height), interpolation=INTER_CUBIC)
        elif self.input_resize_factor < 1:
            return opencv_resize(image, (new_width, new_height), interpolation=INTER_AREA)
        else:
            return image

    def resize_with_output_factor(self, image: numpy_ndarray) -> numpy_ndarray:

        old_height, old_width = self.get_image_resolution(image)

        new_width = int(old_width * self.output_resize_factor)
        new_height = int(old_height * self.output_resize_factor)

        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1

        if self.output_resize_factor > 1:
            return opencv_resize(image, (new_width, new_height), interpolation=INTER_CUBIC)
        elif self.output_resize_factor < 1:
            return opencv_resize(image, (new_width, new_height), interpolation=INTER_AREA)
        else:
            return image

    # AI CLASS FUNCTIONS

    def concatenate_images(self, image1: numpy_ndarray, image2: numpy_ndarray) -> numpy_ndarray:
        image1 = image1 / 255
        image2 = image2 / 255
        concateneted_image = numpy_concatenate((image1, image2), axis=2)
        return concateneted_image

    def preprocess_image(self, image: numpy_ndarray) -> numpy_ndarray:
        image = numpy_transpose(image, (2, 0, 1))
        image = numpy_expand_dims(image, axis=0)
        return image

    def onnxruntime_inference(self, image: numpy_ndarray) -> numpy_ndarray:
        onnx_input = {self.inferenceSession.get_inputs()[0].name: image}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]
        return onnx_output

    def postprocess_output(self, onnx_output: numpy_ndarray) -> numpy_ndarray:
        onnx_output = numpy_squeeze(onnx_output, axis=0)
        onnx_output = numpy_clip(onnx_output, 0, 1)
        onnx_output = numpy_transpose(onnx_output, (1, 2, 0))
        return onnx_output.astype(float32)

    def de_normalize_image(self, onnx_output: numpy_ndarray, max_range: int) -> numpy_ndarray:
        match max_range:
            case 255: return (onnx_output * max_range).astype(uint8)
            case 65535: return (onnx_output * max_range).round().astype(float32)
            # Default fallback to 255
            case _: return (onnx_output * 255).astype(uint8)

    def AI_interpolation(self, image1: numpy_ndarray, image2: numpy_ndarray) -> numpy_ndarray:
        image = self.concatenate_images(image1, image2).astype(float32)
        image = self.preprocess_image(image)
        onnx_output = self.onnxruntime_inference(image)
        onnx_output = self.postprocess_output(onnx_output)
        output_image = self.de_normalize_image(onnx_output, 255)
        return output_image

    # EXTERNAL FUNCTION

    def AI_orchestration(self, image1: numpy_ndarray, image2: numpy_ndarray) -> list[numpy_ndarray]:
        generated_images = []

        # Generate 1 image [image1 / image_A / image2]
        if self.frame_gen_factor == 2:
            image_A = self.AI_interpolation(image1, image2)
            generated_images.append(image_A)

        # Generate 3 images [image1 / image_A / image_B / image_C / image2]
        elif self.frame_gen_factor == 4:
            image_B = self.AI_interpolation(image1, image2)
            image_A = self.AI_interpolation(image1, image_B)
            image_C = self.AI_interpolation(image_B, image2)
            generated_images.append(image_A)
            generated_images.append(image_B)
            generated_images.append(image_C)

        # Generate 7 images [image1 / image_A / image_B / image_C / image_D / image_E / image_F / image_G / image2]
        elif self.frame_gen_factor == 8:
            image_D = self.AI_interpolation(image1, image2)
            image_B = self.AI_interpolation(image1, image_D)
            image_A = self.AI_interpolation(image1, image_B)
            image_C = self.AI_interpolation(image_B, image_D)
            image_F = self.AI_interpolation(image_D, image2)
            image_E = self.AI_interpolation(image_D, image_F)
            image_G = self.AI_interpolation(image_F, image2)
            generated_images.append(image_A)
            generated_images.append(image_B)
            generated_images.append(image_C)
            generated_images.append(image_D)
            generated_images.append(image_E)
            generated_images.append(image_F)
            generated_images.append(image_G)

        return generated_images


# GUI utils ---------------------------

class MessageBox(CTkToplevel):

    def __init__(
            self,
            messageType: str,
            title: str,
            subtitle: str,
            default_value: str,
            option_list: list,
    ) -> None:

        super().__init__()

        self._running: bool = False

        self._messageType = messageType
        self._title = title
        self._subtitle = subtitle
        self._default_value = default_value
        self._option_list = option_list
        self._ctkwidgets_index = 0

        self.title('')
        self.lift()                # lift window on top
        self.attributes("-topmost", True)    # stay on top
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # create widgets with slight delay, to avoid white flickering of background
        self.after(10, self._create_widgets)
        self.resizable(True, True)
        self.grab_set()                # make other windows not clickable

    def _ok_event(
            self,
            event=None
    ) -> None:
        self.grab_release()
        self.destroy()

    def _on_closing(
            self
    ) -> None:
        self.grab_release()
        self.destroy()

    def createEmptyLabel(self) -> CTkLabel:
        return CTkLabel(
            master=self,
            fg_color="transparent",
            width=500,
            height=17,
            text=''
        )

    def placeInfoMessageTitleSubtitle(self) -> None:

        spacingLabel1 = self.createEmptyLabel()
        spacingLabel2 = self.createEmptyLabel()

        if self._messageType == "info":
            title_subtitle_text_color = "#FFD700"  # Amarillo dorado
        elif self._messageType == "error":
            title_subtitle_text_color = "#FF3131"  # Rojo brillante

        titleLabel = CTkLabel(
            master=self,
            width=500,
            anchor='w',
            justify="left",
            fg_color="transparent",
            text_color=title_subtitle_text_color,
            font=bold22,
            text=self._title
        )

        if self._default_value != None:
            defaultLabel = CTkLabel(
                master=self,
                width=500,
                anchor='w',
                justify="left",
                fg_color="transparent",
                text_color="#FFD700",  # Amarillo dorado
                font=bold17,
                text=f"Default: {self._default_value}"
            )

        subtitleLabel = CTkLabel(
            master=self,
            width=500,
            anchor='w',
            justify="left",
            fg_color="transparent",
            text_color=title_subtitle_text_color,
            font=bold14,
            text=self._subtitle
        )

        spacingLabel1.grid(row=self._ctkwidgets_index, column=0,
                           columnspan=2, padx=0, pady=0, sticky="ew")

        self._ctkwidgets_index += 1
        titleLabel.grid(row=self._ctkwidgets_index, column=0,
                        columnspan=2, padx=25, pady=0, sticky="ew")

        if self._default_value != None:
            self._ctkwidgets_index += 1
            defaultLabel.grid(row=self._ctkwidgets_index, column=0,
                              columnspan=2, padx=25, pady=0, sticky="ew")

        self._ctkwidgets_index += 1
        subtitleLabel.grid(row=self._ctkwidgets_index, column=0,
                           columnspan=2, padx=25, pady=0, sticky="ew")

        self._ctkwidgets_index += 1
        spacingLabel2.grid(row=self._ctkwidgets_index, column=0,
                           columnspan=2, padx=0, pady=0, sticky="ew")

    def placeInfoMessageOptionsText(self) -> None:

        for option_text in self._option_list:
            optionLabel = CTkLabel(
                master=self,
                width=600,
                height=45,
                anchor='w',
                justify="left",
                text_color=text_color,
                fg_color="#282828",
                bg_color="transparent",
                font=bold13,
                text=option_text,
                corner_radius=10,
            )

            self._ctkwidgets_index += 1
            optionLabel.grid(row=self._ctkwidgets_index, column=0,
                             columnspan=2, padx=25, pady=4, sticky="ew")

        spacingLabel3 = self.createEmptyLabel()

        self._ctkwidgets_index += 1
        spacingLabel3.grid(row=self._ctkwidgets_index, column=0,
                           columnspan=2, padx=0, pady=0, sticky="ew")

    def placeInfoMessageOkButton(
            self
    ) -> None:

        ok_button = CTkButton(
            master=self,
            command=self._ok_event,
            text='OK',
            width=125,
            font=bold11,
            border_width=1,
            fg_color="#282828",
            text_color="#E0E0E0",
            border_color="#F5E358"
        )

        self._ctkwidgets_index += 1
        ok_button.grid(row=self._ctkwidgets_index, column=1,
                       columnspan=1, padx=(10, 20), pady=(10, 20), sticky="e")

    def _create_widgets(
            self
    ) -> None:

        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self.placeInfoMessageTitleSubtitle()
        self.placeInfoMessageOptionsText()
        self.placeInfoMessageOkButton()


class FileWidget(CTkScrollableFrame):

    def __init__(
            self,
            master,
            selected_file_list,
            upscale_factor=1,
            input_resize_factor=0,
            output_resize_factor=0,
            **kwargs
    ) -> None:

        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.file_list = selected_file_list
        self.upscale_factor = upscale_factor
        self.input_resize_factor = input_resize_factor
        self.output_resize_factor = output_resize_factor

        self.index_row = 1
        self.ui_components = []
        self._create_widgets()

    def _destroy_(self) -> None:
        self.file_list = []
        self.destroy()
        place_loadFile_section()

    def _create_widgets(self) -> None:
        self.add_clean_button()
        for file_path in self.file_list:
            file_name_label, file_info_label = self.add_file_information(
                file_path)
            self.ui_components.append(file_name_label)
            self.ui_components.append(file_info_label)

    def add_file_information(self, file_path) -> tuple:
        infos, icon = self.extract_file_info(file_path)

        # File name
        file_name_label = CTkLabel(
            self,
            text=os_path_basename(file_path),
            font=bold14,
            text_color=text_color,
            compound="left",
            anchor="w",
            padx=10,
            pady=5,
            justify="left",
        )
        file_name_label.grid(
            row=self.index_row,
            column=0,
            pady=(0, 2),
            padx=(3, 3),
            sticky="w"
        )

        # File infos and icon
        file_info_label = CTkLabel(
            self,
            text=infos,
            image=icon,
            font=bold12,
            text_color=text_color,
            compound="left",
            anchor="w",
            padx=10,
            pady=5,
            justify="left",
        )
        file_info_label.grid(
            row=self.index_row + 1,
            column=0,
            pady=(0, 15),
            padx=(3, 3),
            sticky="w"
        )

        self.index_row += 2

        return file_name_label, file_info_label

    def add_clean_button(self) -> None:

        button = CTkButton(
            master=self,
            command=self._destroy_,
            text="CLEAN",
            image=clear_icon,
            width=90,
            height=28,
            font=bold11,
            border_width=1,
            corner_radius=1,
            fg_color="#282828",
            text_color="#E0E0E0",
            border_color="#FFD53D"
        )

        button.grid(row=0, column=2, pady=(7, 7), padx=(0, 7))

    @cache
    def extract_file_icon(self, file_path) -> CTkImage:
        max_size = 60

        if check_if_file_is_video(file_path):
            video_cap = opencv_VideoCapture(file_path)
            _, frame = video_cap.read()
            source_icon = opencv_cvtColor(frame, COLOR_BGR2RGB)
            video_cap.release()
        else:
            source_icon = opencv_cvtColor(image_read(file_path), COLOR_BGR2RGB)

        ratio = min(
            max_size / source_icon.shape[0], max_size / source_icon.shape[1])
        new_width = int(source_icon.shape[1] * ratio)
        new_height = int(source_icon.shape[0] * ratio)
        source_icon = opencv_resize(source_icon, (new_width, new_height))
        ctk_icon = CTkImage(pillow_image_fromarray(
            source_icon, mode="RGB"), size=(new_width, new_height))

        return ctk_icon

    def extract_file_info(self, file_path) -> tuple:

        if check_if_file_is_video(file_path):
            cap = opencv_VideoCapture(file_path)
            width = round(cap.get(CAP_PROP_FRAME_WIDTH))
            height = round(cap.get(CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
            frame_rate = cap.get(CAP_PROP_FPS)
            duration = num_frames/frame_rate
            minutes = int(duration/60)
            seconds = duration % 60
            cap.release()

            file_icon = self.extract_file_icon(file_path)
            file_infos = f"{minutes}m:{round(seconds)}s • {num_frames}frames • {width}x{height} \n"

            if self.input_resize_factor != 0 and self.output_resize_factor != 0 and self.upscale_factor != 0:
                input_resized_height = int(
                    height * (self.input_resize_factor/100))
                input_resized_width = int(
                    width * (self.input_resize_factor/100))

                upscaled_height = int(
                    input_resized_height * self.upscale_factor)
                upscaled_width = int(input_resized_width * self.upscale_factor)

                output_resized_height = int(
                    upscaled_height * (self.output_resize_factor/100))
                output_resized_width = int(
                    upscaled_width * (self.output_resize_factor/100))

                file_infos += (
                    f"AI input ({self.input_resize_factor}%) ➜ {input_resized_width}x{input_resized_height} \n"
                    f"AI output (x{self.upscale_factor}) ➜ {upscaled_width}x{upscaled_height} \n"
                    f"Video output ({self.output_resize_factor}%) ➜ {output_resized_width}x{output_resized_height}"
                )

        else:
            height, width = get_image_resolution(image_read(file_path))
            file_icon = self.extract_file_icon(file_path)

            file_infos = f"{width}x{height}\n"

            if self.input_resize_factor != 0 and self.output_resize_factor != 0 and self.upscale_factor != 0:
                input_resized_height = int(
                    height * (self.input_resize_factor/100))
                input_resized_width = int(
                    width * (self.input_resize_factor/100))

                upscaled_height = int(
                    input_resized_height * self.upscale_factor)
                upscaled_width = int(input_resized_width * self.upscale_factor)

                output_resized_height = int(
                    upscaled_height * (self.output_resize_factor/100))
                output_resized_width = int(
                    upscaled_width * (self.output_resize_factor/100))

                file_infos += (
                    f"AI input ({self.input_resize_factor}%) ➜ {input_resized_width}x{input_resized_height} \n"
                    f"AI output (x{self.upscale_factor}) ➜ {upscaled_width}x{upscaled_height} \n"
                    f"Image output ({self.output_resize_factor}%) ➜ {output_resized_width}x{output_resized_height}"
                )

        return file_infos, file_icon

    # EXTERNAL FUNCTIONS

    def clean_file_list(self) -> None:
        self.index_row = 1
        for ui_component in self.ui_components:
            ui_component.grid_forget()

    def get_selected_file_list(self) -> list:
        return self.file_list

    def set_upscale_factor(self, upscale_factor) -> None:
        self.upscale_factor = upscale_factor

    def set_input_resize_factor(self, input_resize_factor) -> None:
        self.input_resize_factor = input_resize_factor

    def set_output_resize_factor(self, output_resize_factor) -> None:
        self.output_resize_factor = output_resize_factor


def get_values_for_file_widget() -> tuple:
    # Upscale factor
    upscale_factor = get_upscale_factor()

    # Input resolution %
    try:
        input_resize_factor = int(
            float(str(selected_input_resize_factor.get())))
    except (ValueError, TypeError):
        input_resize_factor = 0

    # Output resolution %
    try:
        output_resize_factor = int(
            float(str(selected_output_resize_factor.get())))
    except (ValueError, TypeError):
        output_resize_factor = 0

    return upscale_factor, input_resize_factor, output_resize_factor


def update_file_widget(a, b, c) -> None:
    try:
        selected_file_list = file_widget.get_selected_file_list()
    except Exception:
        return

    upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()

    file_widget.clean_file_list()
    file_widget.set_upscale_factor(upscale_factor)
    file_widget.set_input_resize_factor(input_resize_factor)
    file_widget.set_output_resize_factor(output_resize_factor)
    file_widget._create_widgets()


def create_option_background():
    return CTkFrame(
        master=window,
        bg_color=background_color,
        fg_color=widget_background_color,
        height=46,
        corner_radius=10
    )


def create_info_button(command: Callable, text: str, width: int = 200) -> CTkFrame:

    frame = CTkFrame(
        master=window, fg_color=widget_background_color, height=25)

    button = CTkButton(
        master=frame,
        command=command,
        font=bold12,
        text="?",
        border_color="#ECD125",
        border_width=1,
        fg_color=widget_background_color,
        hover_color=background_color,
        width=23,
        height=15,
        corner_radius=1
    )
    button.grid(row=0, column=0, padx=(0, 7), pady=2, sticky="w")

    label = CTkLabel(
        master=frame,
        text=text,
        width=width,
        height=22,
        fg_color="transparent",
        bg_color=widget_background_color,
        text_color=text_color,
        font=bold13,
        anchor="w"
    )
    label.grid(row=0, column=1, sticky="w")

    frame.grid_propagate(False)
    frame.grid_columnconfigure(1, weight=1)

    return frame


def create_option_menu(
    command: Callable,
    values: list,
    default_value: str,
    border_color: str = "#404040",
    border_width: int = 1,
    width: int = 159
) -> CTkFrame:

    width = width
    height = 28

    total_width = (width + 2 * border_width)
    total_height = (height + 2 * border_width)

    frame = CTkFrame(
        master=window,
        fg_color=border_color,
        width=total_width,
        height=total_height,
        border_width=0,
        corner_radius=1,
    )

    option_menu = CTkOptionMenu(
        master=frame,
        command=command,
        values=values,
        width=width,
        height=height,
        corner_radius=0,
        dropdown_font=bold12,
        font=bold11,
        anchor="center",
        text_color=text_color,
        fg_color=background_color,
        button_color=background_color,
        button_hover_color=background_color,
        dropdown_fg_color=background_color
    )

    option_menu.place(
        x=(total_width - width) / 2,
        y=(total_height - height) / 2
    )
    option_menu.set(default_value)
    return frame


def create_text_box(textvariable: StringVar, width: int) -> CTkEntry:
    return CTkEntry(
        master=window,
        textvariable=textvariable,
        corner_radius=1,
        width=width,
        height=28,
        font=bold11,
        justify="center",
        text_color=text_color,
        fg_color="#000000",
        border_width=1,
        border_color="#404040",
    )


def create_text_box_output_path(textvariable: StringVar) -> CTkEntry:
    return CTkEntry(
        master=window,
        textvariable=textvariable,
        corner_radius=1,
        width=250,
        height=28,
        font=bold11,
        justify="center",
        text_color=text_color,
        fg_color="#000000",
        border_width=1,
        border_color="#404040",
        state=DISABLED
    )


def create_active_button(
        command: Callable,
        text: str,
        icon: CTkImage = None,
        width: int = 140,
        height: int = 30,
        border_color: str = "#C11919"
) -> CTkButton:

    return CTkButton(
        master=window,
        command=command,
        text=text,
        image=icon,
        width=width,
        height=height,
        font=bold11,
        border_width=1,
        corner_radius=1,
        fg_color="#282828",
        text_color="#E0E0E0",
        border_color=border_color
    )


# ==== ERROR HANDLING AND LOGGING SECTION ====

# Configure logging paths in Documents folder
LOG_FOLDER_PATH = os_path_join(DOCUMENT_PATH, f"{app_name}_{version}_Logs")

# Define log file names
MAIN_LOG_FILENAME = 'warlock_studio.log'
ERROR_LOG_FILENAME = 'error_log.txt'

try:
    if not os_path_exists(LOG_FOLDER_PATH):
        os_makedirs(LOG_FOLDER_PATH)
    MAIN_LOG_PATH = os_path_join(LOG_FOLDER_PATH, MAIN_LOG_FILENAME)
    ERROR_LOG_PATH = os_path_join(LOG_FOLDER_PATH, ERROR_LOG_FILENAME)
except Exception as e:
    # Fallback to current directory if Documents folder is not accessible
    print(f"[WARNING] Could not create logs folder in Documents: {str(e)}")
    print(f"[WARNING] Using current directory for logs as fallback")
    MAIN_LOG_PATH = MAIN_LOG_FILENAME
    ERROR_LOG_PATH = ERROR_LOG_FILENAME

# Configure logging
# Ensure logging is set up with a backup/rotation mechanism for maintaining log length.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(MAIN_LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def log_and_report_error(msg: str) -> None:
    """Unified error logging and reporting function."""
    logging.error(msg)
    show_error_message(msg)
    try:
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {msg}\n")
    except Exception as e:
        print(f"[ERROR] Could not write to error log file: {str(e)}")


@contextmanager
def safe_execution(operation_name: str):
    """Context manager for safe execution with error handling."""
    try:
        yield
    except Exception as e:
        error_msg = f"Error during {operation_name}: {str(e)}"
        log_and_report_error(error_msg)
        raise


def validate_environment() -> bool:
    """Validate the runtime environment before starting."""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            log_and_report_error("Python 3.8 or higher required")
            return False

        # Check required modules
        required_modules = ['cv2', 'numpy',
                            'customtkinter', 'onnxruntime', 'PIL']
        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            log_and_report_error(
                f"Missing required modules: {', '.join(missing_modules)}")
            return False

        # Check AI model directory
        ai_model_dir = find_by_relative_path("AI-onnx")
        if not os_path_exists(ai_model_dir):
            log_and_report_error(
                f"AI model directory not found: {ai_model_dir}")
            return False

        return True
    except Exception as e:
        log_and_report_error(f"Environment validation failed: {str(e)}")
        return False


def cleanup_on_exit():
    """Cleanup function to run on application exit."""
    try:
        # Clean up temporary files
        temp_files = [f for f in os_listdir('.') if f.endswith(
            '.tmp') or f.endswith('.checkpoint')]
        for temp_file in temp_files:
            try:
                os_remove(temp_file)
            except Exception:
                pass

        # Stop any running processes
        stop_upscale_process()

        # Force garbage collection
        gc.collect()

        logging.info("Application cleanup completed")
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")


# Register cleanup function
atexit.register(cleanup_on_exit)

# Signal handlers for graceful shutdown


def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_on_exit()
    sys.exit(0)


try:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except AttributeError:
    # Windows doesn't have all signals
    pass


def create_checkpoint(video_path: str, completed_frames: list[str]) -> None:
    """Create checkpoint for video processing recovery."""
    try:
        checkpoint_path = f"{video_path}.checkpoint"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            f.write(f"completed_frames={len(completed_frames)}\n")
            for frame in completed_frames:
                f.write(f"{frame}\n")
        print(
            f"[CHECKPOINT] Created checkpoint with {len(completed_frames)} completed frames")
    except Exception as e:
        print(f"[CHECKPOINT] Could not create checkpoint: {str(e)}")


def load_checkpoint(video_path: str) -> list[str]:
    """Load checkpoint for video processing recovery."""
    try:
        checkpoint_path = f"{video_path}.checkpoint"
        if not os_path_exists(checkpoint_path):
            return []

        completed_frames = []
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip first line with count
                frame = line.strip()
                if frame and os_path_exists(frame):
                    completed_frames.append(frame)

        print(
            f"[CHECKPOINT] Loaded checkpoint with {len(completed_frames)} completed frames")
        return completed_frames
    except Exception as e:
        print(f"[CHECKPOINT] Could not load checkpoint: {str(e)}")
        return []


def cleanup_checkpoint(video_path: str) -> None:
    """Clean up checkpoint file after successful completion."""
    try:
        checkpoint_path = f"{video_path}.checkpoint"
        if os_path_exists(checkpoint_path):
            os_remove(checkpoint_path)
            print(f"[CHECKPOINT] Cleaned up checkpoint file")
    except Exception as e:
        print(f"[CHECKPOINT] Could not cleanup checkpoint: {str(e)}")


def clean_directory(directory_path: str) -> None:
    """Remove all files in a directory."""
    try:
        if os_path_exists(directory_path):
            for file in os_listdir(directory_path):
                file_path = os_path_join(directory_path, file)
                if os_path_exists(file_path):
                    os_remove(file_path)
    except Exception as e:
        logging.error(f"Failed to clean directory {directory_path}: {str(e)}")


def optimize_memory_usage() -> None:
    """Optimize memory usage by triggering garbage collection."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass


def validate_video_file(video_path: str) -> bool:
    """Validate video file integrity and readability."""
    try:
        if not os_path_exists(video_path):
            return False

        # Test if video can be opened
        cap = opencv_VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False

        # Try to read first frame
        ret, frame = cap.read()
        cap.release()

        return ret and frame is not None
    except Exception:
        return False


def get_video_info(video_path: str) -> dict:
    """Get comprehensive video information."""
    try:
        cap = opencv_VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(CAP_PROP_FPS)
        frame_count = int(cap.get(CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'resolution': f"{width}x{height}",
            'file_size': os_path_getsize(video_path) if os_path_exists(video_path) else 0
        }
    except Exception as e:
        log_and_report_error(
            f"Error getting video info for {video_path}: {str(e)}")
        return {}


def estimate_processing_time(video_info: dict, ai_model: str) -> dict:
    """Estimate processing time based on video properties and AI model."""
    try:
        frame_count = video_info.get('frame_count', 0)
        resolution = video_info.get(
            'width', 1920) * video_info.get('height', 1080)

        # Base processing time per frame (in seconds) - rough estimates
        model_speeds = {
            'RealESR_Gx4': 0.5,
            'RealESR_Animex4': 0.5,
            'RealESRNetx4': 1.0,
            'BSRGANx4': 2.0,
            'BSRGANx2': 1.5,
            'RealESRGANx4': 2.0,
            'IRCNN_Mx1': 0.3,
            'IRCNN_Lx1': 0.3,
            'RIFE': 0.8,
            'RIFE_Lite': 0.6
        }

        base_time = model_speeds.get(ai_model, 1.0)
        resolution_factor = resolution / (1920 * 1080)  # Normalize to 1080p

        estimated_time_per_frame = base_time * resolution_factor
        total_estimated_time = estimated_time_per_frame * frame_count

        return {
            'time_per_frame': estimated_time_per_frame,
            'total_time': total_estimated_time,
            'total_time_formatted': format_time_duration(total_estimated_time)
        }
    except Exception:
        return {'time_per_frame': 0, 'total_time': 0, 'total_time_formatted': 'Unknown'}


def format_time_duration(seconds: float) -> str:
    """Format time duration in human readable format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def create_video_backup(video_path: str) -> str:
    """Create backup of original video before processing."""
    try:
        backup_path = f"{video_path}.backup"
        if not os_path_exists(backup_path):
            copy2(video_path, backup_path)
            print(f"[BACKUP] Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"[BACKUP] Warning: Could not create backup: {str(e)}")
        return video_path


def verify_frame_sequence(frame_paths: list[str]) -> bool:
    """Verify that frame sequence is complete and valid."""
    try:
        if not frame_paths:
            return False

        missing_frames = []
        corrupted_frames = []

        for frame_path in frame_paths:
            if not os_path_exists(frame_path):
                missing_frames.append(frame_path)
            else:
                try:
                    # Try to read frame to verify it's not corrupted
                    frame = image_read(frame_path)
                    if frame is None or frame.size == 0:
                        corrupted_frames.append(frame_path)
                except Exception:
                    corrupted_frames.append(frame_path)

        if missing_frames:
            print(f"[FRAME_CHECK] Missing frames: {len(missing_frames)}")
        if corrupted_frames:
            print(f"[FRAME_CHECK] Corrupted frames: {len(corrupted_frames)}")

        return len(missing_frames) == 0 and len(corrupted_frames) == 0
    except Exception as e:
        print(f"[FRAME_CHECK] Error verifying frame sequence: {str(e)}")
        return False


def cleanup_incomplete_frames(target_directory: str, expected_count: int) -> None:
    """Clean up incomplete frame extraction."""
    try:
        if not os_path_exists(target_directory):
            return

        files = os_listdir(target_directory)
        frame_files = [f for f in files if f.startswith(
            'frame_') and f.endswith('.jpg')]

        if len(frame_files) < expected_count:
            print(
                f"[CLEANUP] Removing incomplete frame extraction: {len(frame_files)}/{expected_count} frames")
            for file in frame_files:
                try:
                    os_remove(os_path_join(target_directory, file))
                except Exception:
                    pass
    except Exception as e:
        print(f"[CLEANUP] Error cleaning incomplete frames: {str(e)}")


def monitor_disk_space(required_space_gb: float = 5.0) -> bool:
    """Monitor available disk space during processing."""
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)

        if free_gb < required_space_gb:
            log_and_report_error(
                f"Low disk space: {free_gb:.1f}GB available, {required_space_gb}GB required")
            return False

        if free_gb < required_space_gb * 2:  # Warning threshold
            print(f"[WARNING] Low disk space: {free_gb:.1f}GB available")

        return True
    except Exception:
        return True  # Assume OK if we can't check


def create_frame_index(frame_paths: list[str]) -> dict:
    """Create index of frames for faster lookup."""
    try:
        frame_index = {}
        for i, path in enumerate(frame_paths):
            frame_number = extract_frame_number_from_path(path)
            frame_index[frame_number] = {
                'path': path,
                'index': i,
                'exists': os_path_exists(path)
            }
        return frame_index
    except Exception:
        return {}


def extract_frame_number_from_path(frame_path: str) -> int:
    """Extract frame number from frame file path."""
    try:
        filename = os_path_basename(frame_path)
        # Extract number from patterns like "frame_001.jpg"
        import re
        match = re.search(r'frame_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    except Exception:
        return 0


def validate_ai_model_compatibility(ai_model: str, operation: str) -> bool:
    """Validate AI model compatibility with requested operation."""
    try:
        if operation == "upscaling":
            return ai_model not in RIFE_models_list
        elif operation == "interpolation":
            return ai_model in RIFE_models_list
        return True
    except Exception:
        return False


def validate_file_paths(file_paths: list[str]) -> bool:
    """Validate that all file paths exist and are accessible."""
    if not file_paths:
        return False

    missing_files = []
    invalid_files = []

    for path in file_paths:
        if not os_path_exists(path):
            missing_files.append(path)
        else:
            try:
                # Test if file is readable
                with open(path, 'rb') as f:
                    f.read(1)
            except Exception as e:
                invalid_files.append(f"{path}: {str(e)}")

    if missing_files:
        log_and_report_error(f"Missing files detected: {missing_files}")
    if invalid_files:
        log_and_report_error(f"Inaccessible files detected: {invalid_files}")

    return len(missing_files) == 0 and len(invalid_files) == 0


def validate_output_path(output_path: str) -> bool:
    """Validate output path is writable."""
    if output_path == OUTPUT_PATH_CODED:
        return True

    if not os_path_exists(output_path):
        try:
            os_makedirs(output_path, exist_ok=True)
        except Exception as e:
            log_and_report_error(
                f"Cannot create output directory {output_path}: {str(e)}")
            return False

    # Test write permissions
    test_file = os_path_join(output_path, "test_write_permissions.tmp")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os_remove(test_file)
        return True
    except Exception as e:
        log_and_report_error(
            f"Output path not writable {output_path}: {str(e)}")
        return False


def validate_system_requirements() -> bool:
    """Validate system requirements for processing."""
    errors = []

    # Check FFmpeg
    if not os_path_exists(FFMPEG_EXE_PATH):
        errors.append("FFmpeg executable not found")

    # Check available disk space (enhanced check)
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        if free < (1024 * 1024 * 1024):  # Less than 1GB free
            errors.append("Low disk space: less than 1GB available")
        elif free < (2 * 1024 * 1024 * 1024):  # Less than 2GB free
            print(
                f"[WARNING] Low disk space: {free // (1024*1024*1024):.1f}GB available")
    except Exception:
        pass  # Ignore if we can't check disk space

    # Check available RAM
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available < (2 * 1024 * 1024 * 1024):  # Less than 2GB available
            errors.append(
                f"Low available RAM: {memory.available // (1024*1024*1024):.1f}GB")
    except ImportError:
        print("[WARNING] psutil not available, cannot check RAM")
    except Exception:
        pass

    if errors:
        for error in errors:
            log_and_report_error(error)
        return False
    return True

# ==== FILE UTILITIES SECTION ====


def create_dir(name_dir: str) -> None:
    if os_path_exists(name_dir):
        remove_directory(name_dir)
    if not os_path_exists(name_dir):
        os_makedirs(name_dir, mode=0o777)


def stop_thread() -> None:
    """Notifica al hilo de monitoreo que debe detenerse de forma segura."""
    global stop_thread_flag
    stop_thread_flag.set()


def image_read(file_path: str) -> numpy_ndarray:
    with open(file_path, 'rb') as file:
        return opencv_imdecode(numpy_ascontiguousarray(numpy_frombuffer(file.read(), uint8)), IMREAD_UNCHANGED)


def image_write(file_path: str, file_data: numpy_ndarray, file_extension: str = ".jpg") -> None:
    opencv_imencode(file_extension, file_data)[1].tofile(file_path)


def copy_file_metadata(original_file_path: str, upscaled_file_path: str) -> None:
    try:
        # Check if exiftool exists
        if not os_path_exists(EXIFTOOL_EXE_PATH):
            print("[ExifTool] ExifTool not found, skipping metadata copy")
            return

        # Check if files exist
        if not os_path_exists(original_file_path):
            print(f"[ExifTool] Original file not found: {original_file_path}")
            return

        if not os_path_exists(upscaled_file_path):
            print(f"[ExifTool] Upscaled file not found: {upscaled_file_path}")
            return

        exiftool_cmd = [
            EXIFTOOL_EXE_PATH,
            '-fast',
            '-TagsFromFile',
            original_file_path,
            '-overwrite_original',
            '-all:all',
            '-unsafe',
            '-largetags',
            upscaled_file_path
        ]

        result = subprocess_run(exiftool_cmd, check=True,
                                shell=False, capture_output=True, text=True)
        print(f"[ExifTool] Successfully copied metadata")

    except CalledProcessError as e:
        print(
            f"[ExifTool] ExifTool failed: {e.stderr if e.stderr else str(e)}")
    except Exception as e:
        print(f"[ExifTool] Could not copy metadata: {str(e)}")


def prepare_output_image_filename(
        image_path: str,
        selected_output_path: str,
        selected_AI_model: str,
        input_resize_factor: int,
        output_resize_factor: int,
        selected_image_extension: str,
        selected_blending_factor: float
) -> str:

    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(image_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(image_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected input resize
    to_append += f"_InputR-{str(int(input_resize_factor * 100))}"

    # Selected output resize
    to_append += f"_OutputR-{str(int(output_resize_factor * 100))}"

    # Selected intepolation
    match selected_blending_factor:
        case 0.3:
            to_append += "_Blending-Low"
        case 0.5:
            to_append += "_Blending-Medium"
        case 0.7:
            to_append += "_Blending-High"

    # Selected image extension
    to_append += f"{selected_image_extension}"

    output_path += to_append

    return output_path


def prepare_output_video_frame_filename(
        frame_path: str,
        selected_AI_model: str,
        input_resize_factor: int,
        output_resize_factor: int,
        selected_blending_factor: float
) -> str:

    file_path_no_extension, _ = os_path_splitext(frame_path)
    output_path = file_path_no_extension

    # Selected AI model
    to_append = f"_{selected_AI_model}"

    # Selected input resize
    to_append += f"_InputR-{str(int(input_resize_factor * 100))}"

    # Selected output resize
    to_append += f"_OutputR-{str(int(output_resize_factor * 100))}"

    # Selected intepolation
    match selected_blending_factor:
        case 0.3:
            to_append += "_Blending-Low"
        case 0.5:
            to_append += "_Blending-Medium"
        case 0.7:
            to_append += "_Blending-High"

    # Selected image extension
    to_append += f".jpg"

    output_path += to_append

    return output_path


def prepare_output_video_filename(
        video_path: str,
        selected_output_path: str,
        selected_AI_model: str,
        frame_gen_factor: int,
        slowmotion: bool,
        input_resize_factor: int,
        output_resize_factor: int,
        selected_video_extension: str,
) -> str:
    # FluidFrames-compatible signature and logic
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(video_path)
        file_path_no_extension, _ = os_path_splitext(file_name)
        output_path = f"{selected_output_path}{os_separator}{file_path_no_extension}"

    # Selected AI model
    to_append = f"_{selected_AI_model}x{str(frame_gen_factor)}"
    # Slowmotion?
    if slowmotion:
        to_append += f"_slowmo"
    # Selected input resize
    to_append += f"_InputR-{str(int(input_resize_factor * 100))}"
    # Selected output resize
    to_append += f"_OutputR-{str(int(output_resize_factor * 100))}"
    # Video extension
    to_append += f"{selected_video_extension}"

    output_path += to_append

    return output_path


def prepare_output_video_directory_name(
        video_path: str,
        selected_output_path: str,
        selected_AI_model: str,
        frame_gen_factor: int,
        slowmotion: bool,
        input_resize_factor: int,
        output_resize_factor: int,
) -> str:
    # FluidFrames-style: compatible with interpolation models and upscalers
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name = os_path_basename(video_path)
        file_path_no_extension, _ = os_path_splitext(file_name)
        output_path = f"{selected_output_path}{os_separator}{file_path_no_extension}"

    # Selected AI model
    to_append = f"_{selected_AI_model}x{str(frame_gen_factor)}"
    # Slowmotion?
    if slowmotion:
        to_append += f"_slowmo"
    # Selected input resize
    to_append += f"_InputR-{str(int(input_resize_factor * 100))}"
    # Selected output resize
    to_append += f"_OutputR-{str(int(output_resize_factor * 100))}"
    output_path += to_append
    return output_path


# ==== IMAGE/VIDEO UTILITIES SECTION ====

def get_video_fps(video_path: str) -> float:
    video_capture = opencv_VideoCapture(video_path)
    frame_rate = video_capture.get(CAP_PROP_FPS)
    video_capture.release()
    return frame_rate


def get_image_resolution(image: numpy_ndarray) -> tuple:
    height = image.shape[0]
    width = image.shape[1]

    return height, width


def save_extracted_frames(
        extracted_frames_paths: list[str],
        extracted_frames: list[numpy_ndarray],
        cpu_number: int
) -> None:

    with ThreadPool(cpu_number) as pool:
        pool.starmap(image_write, zip(
            extracted_frames_paths, extracted_frames))


def extract_video_frames(
    process_status_q: multiprocessing_Queue,
    file_number: int,
    target_directory: str,
    AI_instance,
    video_path: str,
    cpu_number: int,
    selected_image_extension: str
) -> list[str]:
    # FluidFrames-compatible implementation
    try:
        create_dir(target_directory)

        # Check if video file exists
        if not os_path_exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames_number_to_save = cpu_number * ECTRACTION_FRAMES_FOR_CPU
        video_capture = opencv_VideoCapture(video_path)

        # Check if video was opened successfully
        if not video_capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count = int(video_capture.get(CAP_PROP_FRAME_COUNT))

        # Check if frame count is valid
        if frame_count <= 0:
            raise ValueError(
                f"Invalid frame count ({frame_count}) for video: {video_path}")

        extracted_frames = []
        extracted_frames_paths = []
        video_frames_list = []
        frame_index = 0

        for frame_number in range(frame_count):
            success, frame = video_capture.read()
            if not success:
                if frame_number == 0:
                    raise ValueError(
                        f"Could not read any frames from video: {video_path}")
                print(
                    f"Warning: Could not read frame {frame_number}, stopping extraction")
                break

            try:
                frame_path = f"{target_directory}{os_separator}frame_{frame_number:03d}{selected_image_extension}"
                frame = AI_instance.resize_with_input_factor(frame)
                extracted_frames.append(frame)
                extracted_frames_paths.append(frame_path)
                video_frames_list.append(frame_path)
            except Exception as e:
                print(
                    f"Warning: Error processing frame {frame_number}: {str(e)}")
                continue

            if len(extracted_frames) == frames_number_to_save:
                percentage_extraction = (frame_number / frame_count) * 100
                write_process_status(
                    process_status_q, f"{file_number}. Extracting video frames ({round(percentage_extraction, 2)}%)")
                try:
                    save_extracted_frames(extracted_frames_paths,
                                          extracted_frames, cpu_number)
                except Exception as e:
                    print(f"Warning: Error saving frames batch: {str(e)}")
                extracted_frames = []
                extracted_frames_paths = []

            frame_index += 1

        video_capture.release()

        if len(extracted_frames) > 0:
            try:
                save_extracted_frames(extracted_frames_paths,
                                      extracted_frames, cpu_number)
            except Exception as e:
                print(f"Warning: Error saving final frames batch: {str(e)}")

        if len(video_frames_list) == 0:
            raise ValueError(
                f"No frames were successfully extracted from video: {video_path}")

        return video_frames_list

    except Exception as e:
        if 'video_capture' in locals():
            video_capture.release()
        write_process_status(
            process_status_q, f"{ERROR_STATUS}Error extracting frames from {os_path_basename(video_path)}: {str(e)}")
        raise


def validate_ffmpeg_executable() -> bool:
    """Validate FFmpeg executable and check its functionality."""
    try:
        if not os_path_exists(FFMPEG_EXE_PATH):
            log_and_report_error(
                "FFmpeg executable not found at expected path")
            return False

        # Test FFmpeg by getting version info
        result = subprocess_run(
            [FFMPEG_EXE_PATH, "-version"],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            log_and_report_error("FFmpeg executable test failed")
            return False

        print(f"[FFMPEG] Validation successful")
        return True
    except Exception as e:
        log_and_report_error(f"FFmpeg validation error: {str(e)}")
        return False


def get_video_codec_settings(selected_video_codec: str, video_info: dict) -> dict:
    """Get optimized codec settings based on video properties and user selection."""
    width = video_info.get('width', 1920)
    height = video_info.get('height', 1080)

    # Base settings for different codecs
    codec_settings = {
        'x264': {
            'codec': 'libx264',
            'preset': 'medium',
            'crf': '18',
            'profile': 'high',
            'level': '4.1',
            'pix_fmt': 'yuv420p'
        },
        'x265': {
            'codec': 'libx265',
            'preset': 'medium',
            'crf': '20',
            'profile': 'main',
            'pix_fmt': 'yuv420p'
        },
        'h264_nvenc': {
            'codec': 'h264_nvenc',
            'preset': 'p4',
            'cq': '20',
            'profile': 'high',
            'level': '4.1',
            'pix_fmt': 'yuv420p',
            'rc': 'vbr'
        },
        'hevc_nvenc': {
            'codec': 'hevc_nvenc',
            'preset': 'p4',
            'cq': '22',
            'profile': 'main',
            'pix_fmt': 'yuv420p',
            'rc': 'vbr'
        },
        'h264_amf': {
            'codec': 'h264_amf',
            'quality': 'balanced',
            'rc': 'cqp',
            'qp_i': '20',
            'qp_p': '22',
            'qp_b': '24',
            'profile': 'high'
        },
        'hevc_amf': {
            'codec': 'hevc_amf',
            'quality': 'balanced',
            'rc': 'cqp',
            'qp_i': '22',
            'qp_p': '24',
            'qp_b': '26',
            'profile': 'main'
        },
        'h264_qsv': {
            'codec': 'h264_qsv',
            'preset': 'medium',
            'global_quality': '20',
            'profile': 'high',
            'pix_fmt': 'nv12'
        },
        'hevc_qsv': {
            'codec': 'hevc_qsv',
            'preset': 'medium',
            'global_quality': '22',
            'profile': 'main',
            'pix_fmt': 'nv12'
        }
    }

    # Get base settings for the selected codec
    settings = codec_settings.get(selected_video_codec, codec_settings['x264'])

    # Adjust bitrate based on resolution
    pixels = width * height
    if pixels <= 720 * 480:  # SD
        bitrate = '2000k'
    elif pixels <= 1280 * 720:  # HD
        bitrate = '5000k'
    elif pixels <= 1920 * 1080:  # FHD
        bitrate = '8000k'
    elif pixels <= 2560 * 1440:  # QHD
        bitrate = '12000k'
    else:  # 4K+
        bitrate = '20000k'

    settings['bitrate'] = bitrate
    return settings


def test_codec_compatibility(codec_name: str) -> bool:
    """Test if a specific codec is available and working."""
    try:
        # Test encoding a single black frame
        test_command = [
            FFMPEG_EXE_PATH,
            "-f", "lavfi",
            "-i", "color=black:size=64x64:duration=0.1",
            "-c:v", codec_name,
            "-f", "null",
            "-"
        ]

        result = subprocess_run(
            test_command,
            capture_output=True,
            text=True,
            timeout=10
        )

        return result.returncode == 0
    except Exception:
        return False


def build_encoding_command(
    video_path: str,
    txt_path: str,
    no_audio_path: str,
    codec_settings: dict,
    video_fps: str
) -> list[str]:
    """Build FFmpeg encoding command with proper settings."""

    base_command = [
        FFMPEG_EXE_PATH,
        "-y",
        "-loglevel", "error",
        "-stats",
        "-f", "concat",
        "-safe", "0",
        "-r", video_fps,
        "-i", txt_path,
        "-c:v", codec_settings['codec']
    ]

    # Add codec-specific parameters
    codec = codec_settings['codec']

    if 'libx264' in codec:
        base_command.extend([
            "-preset", codec_settings['preset'],
            "-crf", codec_settings['crf'],
            "-profile:v", codec_settings['profile'],
            "-level:v", codec_settings['level'],
            "-pix_fmt", codec_settings['pix_fmt'],
            "-movflags", "+faststart"
        ])
    elif 'libx265' in codec:
        base_command.extend([
            "-preset", codec_settings['preset'],
            "-crf", codec_settings['crf'],
            "-profile:v", codec_settings['profile'],
            "-pix_fmt", codec_settings['pix_fmt'],
            "-tag:v", "hvc1",
            "-movflags", "+faststart"
        ])
    elif 'nvenc' in codec:
        base_command.extend([
            "-preset", codec_settings['preset'],
            "-rc", codec_settings['rc'],
            "-cq", codec_settings['cq'],
            "-profile:v", codec_settings['profile'],
            "-pix_fmt", codec_settings['pix_fmt'],
            "-movflags", "+faststart"
        ])
    elif 'amf' in codec:
        base_command.extend([
            "-quality", codec_settings['quality'],
            "-rc", codec_settings['rc'],
            "-qp_i", codec_settings['qp_i'],
            "-qp_p", codec_settings['qp_p'],
            "-qp_b", codec_settings['qp_b'],
            "-profile:v", codec_settings['profile']
        ])
    elif 'qsv' in codec:
        base_command.extend([
            "-preset", codec_settings['preset'],
            "-global_quality", codec_settings['global_quality'],
            "-profile:v", codec_settings['profile'],
            "-pix_fmt", codec_settings['pix_fmt']
        ])
    else:
        # Fallback for unknown codecs
        base_command.extend([
            "-b:v", codec_settings['bitrate'],
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"
        ])

    # Add output file
    base_command.append(no_audio_path)

    return base_command


def create_frame_list_file(frame_paths: list[str], txt_path: str) -> bool:
    """Create frame list file for FFmpeg concat demuxer with validation."""
    try:
        # Verify all frames exist and are readable
        valid_frames = []
        invalid_count = 0

        for frame_path in frame_paths:
            if os_path_exists(frame_path):
                try:
                    # Quick file size check
                    if os_path_getsize(frame_path) > 0:
                        valid_frames.append(frame_path)
                    else:
                        invalid_count += 1
                        print(f"[WARNING] Empty frame file: {frame_path}")
                except Exception:
                    invalid_count += 1
                    print(f"[WARNING] Cannot access frame file: {frame_path}")
            else:
                invalid_count += 1
                print(f"[WARNING] Missing frame file: {frame_path}")

        if invalid_count > 0:
            print(
                f"[WARNING] Found {invalid_count} invalid/missing frames out of {len(frame_paths)}")

        if len(valid_frames) == 0:
            raise ValueError("No valid frames found for video encoding")

        # Create the frame list file
        with open(txt_path, 'w', encoding='utf-8') as f:
            for frame_path in valid_frames:
                # Escape path for FFmpeg and use forward slashes
                escaped_path = frame_path.replace(
                    '\\', '/').replace("'", "'\"'\"'")
                f.write(f"file '{escaped_path}'\n")

        print(f"[FFMPEG] Frame list created with {len(valid_frames)} frames")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to create frame list file: {str(e)}")
        return False


def video_encoding(
        process_status_q: multiprocessing_Queue,
        video_path: str,
        video_output_path: str,
        upscaled_frame_paths: list[str],
        selected_video_codec: str,
) -> None:
    """Enhanced video encoding with robust error handling and codec support."""

    try:
        # Validate inputs
        if not upscaled_frame_paths:
            raise ValueError("No frame paths provided for video encoding")

        if not validate_ffmpeg_executable():
            raise RuntimeError("FFmpeg validation failed")

        # Get video information
        video_info = get_video_info(video_path)
        if not video_info:
            raise ValueError("Could not get video information")

        # Get optimized codec settings
        codec_settings = get_video_codec_settings(
            selected_video_codec, video_info)

        # Test codec compatibility
        if not test_codec_compatibility(codec_settings['codec']):
            print(
                f"[WARNING] Codec {codec_settings['codec']} not available, falling back to libx264")
            codec_settings = get_video_codec_settings('x264', video_info)

        # Prepare file paths
        base_name = os_path_splitext(video_output_path)[0]
        txt_path = f"{base_name}_frames.txt"
        no_audio_path = f"{base_name}_no_audio{os_path_splitext(video_output_path)[1]}"

        # Clean up any existing temporary files
        for temp_file in [txt_path, no_audio_path]:
            if os_path_exists(temp_file):
                try:
                    os_remove(temp_file)
                except Exception as e:
                    print(
                        f"[WARNING] Could not remove temporary file {temp_file}: {e}")

        # Get video FPS with fallback
        try:
            video_fps = get_video_fps(video_path)
            if video_fps <= 0 or video_fps > 1000:  # Sanity check
                raise ValueError(f"Invalid frame rate: {video_fps}")
            video_fps_str = f"{video_fps:.6f}"  # High precision for FFmpeg
        except Exception as e:
            print(f"[WARNING] Could not get video FPS: {e}, using 30.0")
            video_fps_str = "30.000000"

        # Create frame list file
        if not create_frame_list_file(upscaled_frame_paths, txt_path):
            raise RuntimeError("Failed to create frame list file")

        # Build encoding command
        encoding_command = build_encoding_command(
            video_path, txt_path, no_audio_path, codec_settings, video_fps_str
        )

        # Execute video encoding
        print(f"[FFMPEG] Starting encoding with {codec_settings['codec']}")
        print(
            f"[FFMPEG] Processing {len(upscaled_frame_paths)} frames at {video_fps_str} FPS")

        try:
            result = subprocess_run(
                encoding_command,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Verify output file was created and has reasonable size
            if not os_path_exists(no_audio_path):
                raise RuntimeError(
                    "Video encoding completed but output file was not created")

            output_size = os_path_getsize(no_audio_path)
            if output_size < 1024:  # Less than 1KB indicates failure
                raise RuntimeError(
                    f"Video encoding produced suspiciously small file: {output_size} bytes")

            print(
                f"[FFMPEG] Video encoding completed: {output_size / (1024*1024):.1f} MB")

        except subprocess.TimeoutExpired:
            error_msg = "Video encoding timeout (exceeded 1 hour)"
            log_and_report_error(error_msg)
            write_process_status(
                process_status_q, f"{ERROR_STATUS}{error_msg}")
            return
        except CalledProcessError as e:
            error_details = e.stderr if e.stderr else str(e)
            error_msg = f"FFmpeg encoding failed: {error_details}"

            # Try to provide helpful error messages
            if "Unknown encoder" in error_details:
                error_msg += "\nThe selected codec is not supported. Try x264 instead."
            elif "Device or resource busy" in error_details:
                error_msg += "\nGPU encoder is busy. Try software encoding (x264/x265)."
            elif "Invalid data" in error_details:
                error_msg += "\nFrame data may be corrupted. Check input images."

            log_and_report_error(error_msg)
            write_process_status(
                process_status_q, f"{ERROR_STATUS}{error_msg}")
            return

        # Audio passthrough with multiple fallback strategies
        print("[FFMPEG] Processing audio track")

        # Check if original video has audio
        audio_info_command = [
            FFMPEG_EXE_PATH,
            "-i", video_path,
            "-hide_banner",
            "-loglevel", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0"
        ]

        has_audio = False
        try:
            audio_result = subprocess_run(
                audio_info_command,
                capture_output=True,
                text=True,
                timeout=30
            )
            has_audio = audio_result.returncode == 0 and audio_result.stdout.strip()
        except Exception:
            print("[WARNING] Could not detect audio stream, assuming no audio")

        if has_audio:
            # Strategy 1: Copy audio as-is
            audio_command = [
                FFMPEG_EXE_PATH,
                "-y",
                "-loglevel", "error",
                "-i", video_path,
                "-i", no_audio_path,
                "-c:v", "copy",
                "-c:a", "copy",
                "-map", "1:v:0",
                "-map", "0:a:0",
                "-shortest",
                video_output_path
            ]

            try:
                result = subprocess_run(
                    audio_command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if os_path_exists(no_audio_path):
                    os_remove(no_audio_path)
                print("[FFMPEG] Audio passthrough completed successfully")

            except (CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"[WARNING] Audio copy failed: {e}")

                # Strategy 2: Re-encode audio
                print("[FFMPEG] Trying audio re-encoding...")
                audio_reencode_command = [
                    FFMPEG_EXE_PATH,
                    "-y",
                    "-loglevel", "error",
                    "-i", video_path,
                    "-i", no_audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-map", "1:v:0",
                    "-map", "0:a:0",
                    "-shortest",
                    video_output_path
                ]

                try:
                    result = subprocess_run(
                        audio_reencode_command,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )

                    if os_path_exists(no_audio_path):
                        os_remove(no_audio_path)
                    print("[FFMPEG] Audio re-encoding completed successfully")

                except Exception as audio_error:
                    print(
                        f"[WARNING] Audio re-encoding also failed: {audio_error}")
                    # Strategy 3: Use video without audio
                    try:
                        if os_path_exists(no_audio_path):
                            shutil_move(no_audio_path, video_output_path)
                            print("[FFMPEG] Using video without audio")
                    except Exception as move_error:
                        raise RuntimeError(
                            f"Failed to save final video: {move_error}")
        else:
            # No audio in original, just rename the video file
            try:
                shutil_move(no_audio_path, video_output_path)
                print("[FFMPEG] Video saved successfully (no audio track)")
            except Exception as move_error:
                raise RuntimeError(f"Failed to save final video: {move_error}")

        # Clean up temporary files
        for temp_file in [txt_path]:
            if os_path_exists(temp_file):
                try:
                    os_remove(temp_file)
                except Exception:
                    pass

        # Final validation
        if not os_path_exists(video_output_path):
            raise RuntimeError(
                "Video encoding completed but final output file is missing")

        final_size = os_path_getsize(video_output_path)
        print(
            f"[FFMPEG] Final video created: {final_size / (1024*1024):.1f} MB")

    except Exception as e:
        error_msg = f"Video encoding failed: {str(e)}"
        log_and_report_error(error_msg)
        write_process_status(process_status_q, f"{ERROR_STATUS}{error_msg}")

        # Clean up on failure
        for temp_file in [txt_path, no_audio_path] if 'txt_path' in locals() and 'no_audio_path' in locals() else []:
            if os_path_exists(temp_file):
                try:
                    os_remove(temp_file)
                except Exception:
                    pass


def check_video_upscaling_resume(
        target_directory: str,
        selected_AI_model: str
) -> bool:

    if os_path_exists(target_directory):
        directory_files = os_listdir(target_directory)
        upscaled_frames_path = [
            file for file in directory_files if selected_AI_model in file]

        if len(upscaled_frames_path) > 1:
            return True
        else:
            return False
    else:
        return False


def get_video_frames_for_upscaling_resume(
        target_directory: str,
        selected_AI_model: str,
) -> list[str]:

    # Only file names
    directory_files = os_listdir(target_directory)
    original_frames_path = [
        file for file in directory_files if file.endswith('.jpg')]
    original_frames_path = [
        file for file in original_frames_path if selected_AI_model not in file]

    # Adding the complete path to file
    original_frames_path = natsorted(
        [os_path_join(target_directory, file) for file in original_frames_path])

    return original_frames_path


def calculate_time_to_complete_video(
        time_for_frame: float,
        remaining_frames: int,
) -> str:

    remaining_time = time_for_frame * remaining_frames

    hours_left = remaining_time // 3600
    minutes_left = (remaining_time % 3600) // 60
    seconds_left = round((remaining_time % 3600) % 60)

    time_left = ""

    if int(hours_left) > 0:
        time_left = f"{int(hours_left):02d}h"

    if int(minutes_left) > 0:
        time_left = f"{time_left}{int(minutes_left):02d}m"

    if seconds_left > 0:
        time_left = f"{time_left}{seconds_left:02d}s"

    return time_left


def blend_images_and_save(
        target_path: str,
        starting_image: numpy_ndarray,
        upscaled_image: numpy_ndarray,
        starting_image_importance: float,
        file_extension: str = ".jpg"
) -> None:

    def add_alpha_channel(image: numpy_ndarray) -> numpy_ndarray:
        if image.shape[2] == 3:
            alpha = numpy_full(
                (image.shape[0], image.shape[1], 1), 255, dtype=uint8)
            image = numpy_concatenate((image, alpha), axis=2)
        return image

    def get_image_mode(image: numpy_ndarray) -> str:
        shape = image.shape
        if len(shape) == 2:
            return "Grayscale"
        elif len(shape) == 3 and shape[2] == 3:
            return "RGB"
        elif len(shape) == 3 and shape[2] == 4:
            return "RGBA"

    upscaled_image_importance = 1 - starting_image_importance
    starting_height, starting_width = get_image_resolution(starting_image)
    target_height, target_width = get_image_resolution(upscaled_image)

    starting_resolution = starting_height + starting_width
    target_resolution = target_height + target_width

    if starting_resolution > target_resolution:
        starting_image = opencv_resize(
            starting_image, (target_width, target_height), INTER_AREA)
    else:
        starting_image = opencv_resize(
            starting_image, (target_width, target_height))

    try:
        if get_image_mode(starting_image) == "RGBA":
            starting_image = add_alpha_channel(starting_image)
            upscaled_image = add_alpha_channel(upscaled_image)

        interpolated_image = opencv_addWeighted(
            starting_image, starting_image_importance, upscaled_image, upscaled_image_importance, 0)
        image_write(target_path, interpolated_image, file_extension)

    except Exception as e:
        print(
            f"[BLEND] Blending failed, saving original upscaled image: {str(e)}")
        image_write(target_path, upscaled_image, file_extension)


# ==== CORE PROCESSING SECTION ====

def check_upscale_steps() -> None:
    """Monitorea el estado del proceso de escalado en un hilo separado."""
    global stop_thread_flag
    sleep(1)

    while not stop_thread_flag.is_set():
        try:
            actual_step = read_process_status()

            if actual_step == COMPLETED_STATUS:
                info_message.set(f"All files completed!")
                stop_upscale_process()
                stop_thread_flag.set()  # Señaliza la finalización del hilo
                break  # Sal del bucle

            elif actual_step == STOP_STATUS:
                info_message.set(f"Magic stopped")
                stop_upscale_process()
                stop_thread_flag.set()  # Señaliza la finalización del hilo
                break  # Sal del bucle

            elif ERROR_STATUS in actual_step:
                info_message.set(f"Error while upscaling :(")
                error_to_show = actual_step.replace(ERROR_STATUS, "")
                show_error_message(error_to_show.strip())
                stop_thread_flag.set()  # Señaliza la finalización del hilo
                break  # Sal del bucle
            else:
                info_message.set(actual_step)

            sleep(1)
        except Exception as e:
            # Si hay un error al leer la cola, el proceso principal probablemente murió.
            print(f"[MONITOR] Error reading process status: {str(e)}")
            # Sal del bucle para terminar el hilo.
            break

    # Se asegura de que el botón de re-inicio aparezca al final
    place_upscale_button()


def read_process_status() -> str:
    return process_status_q.get()


def write_process_status(process_status_q: multiprocessing_Queue, step: str) -> None:

    print(f"{step}")
    while not process_status_q.empty():
        process_status_q.get()
    process_status_q.put(f"{step}")


def stop_upscale_process() -> None:
    global process_upscale_orchestrator
    try:
        process_upscale_orchestrator
    except NameError:
        pass
    else:
        # Fix 4.1: Use terminate() instead of kill() for safer process termination
        process_upscale_orchestrator.terminate()


def stop_button_command() -> None:
    stop_upscale_process()
    write_process_status(process_status_q, f"{STOP_STATUS}")


def upscale_button_command() -> None:
    # --- Unified upscaling/interpolation pipeline: FluidFrames integration ---
    global selected_file_list
    global selected_AI_model
    global selected_gpu
    global selected_keep_frames
    global selected_AI_multithreading
    global selected_blending_factor
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global tiles_resolution
    global input_resize_factor
    global output_resize_factor
    global selected_frame_generation_option
    global process_upscale_orchestrator
    global stop_thread_flag

    # Fix 2.2: Clear stop_thread_flag at the beginning of each execution
    stop_thread_flag.clear()

    if user_input_checks():
        info_message.set("Loading")
        cpu_number = int(os_cpu_count()/2)
        print("=" * 50)
        print(f"> Starting:")
        print(f"  Files to process: {len(selected_file_list)}")
        print(f"  Output path: {(selected_output_path.get())}")
        print(f"  Selected AI model: {selected_AI_model}")
        print(
            f"  Selected frame generation option: {selected_frame_generation_option}")
        print(f"  Selected GPU: {selected_gpu}")
        print(f"  AI multithreading: {selected_AI_multithreading}")
        print(f"  Blending/factor: {selected_blending_factor}")
        print(f"  Selected image output extension: {selected_image_extension}")
        print(f"  Selected video output extension: {selected_video_extension}")
        print(f"  Selected video output codec: {selected_video_codec}")
        print(
            f"  Tiles resolution (for GPU): {tiles_resolution}x{tiles_resolution}px")
        print(f"  Input resize: {int(input_resize_factor * 100)}%")
        print(f"  Output resize: {int(output_resize_factor * 100)}%")
        print(f"  CPU threads: {cpu_number}")
        print(f"  Save frames: {selected_keep_frames}")
        print("=" * 50)
        place_stop_button()

        # Use FluidFrames' RIFE-based pipeline when relevant
        if selected_AI_model in RIFE_models_list:
            process_upscale_orchestrator = Process(
                target=fluidframes_interpolation_pipeline,
                args=(process_status_q, selected_file_list, selected_output_path.get(), selected_AI_model, selected_gpu,
                      selected_frame_generation_option, selected_image_extension, selected_video_extension, selected_video_codec,
                      input_resize_factor, output_resize_factor, cpu_number, selected_keep_frames)
            )
            process_upscale_orchestrator.start()
        else:
            process_upscale_orchestrator = Process(
                target=upscale_orchestrator,
                args=(process_status_q, selected_file_list, selected_output_path.get(), selected_AI_model, selected_AI_multithreading,
                      input_resize_factor, output_resize_factor, selected_gpu, tiles_resolution, selected_blending_factor,
                      selected_keep_frames, selected_image_extension, selected_video_extension, selected_video_codec, cpu_number,)
            )
            process_upscale_orchestrator.start()

        thread_wait = Thread(target=check_upscale_steps)
        thread_wait.start()

# --- Inserted: FluidFrames orchestration (minimal, reusing classes/logic copied from FluidFrames.py) ---


def fluidframes_interpolation_pipeline(
        process_status_q, selected_file_list, selected_output_path, selected_AI_model, selected_gpu,
        selected_generation_option, selected_image_extension, selected_video_extension, selected_video_codec,
        input_resize_factor, output_resize_factor, cpu_number, selected_keep_frames):
    '''
    This function runs all the FluidFrames video/image interpolation generation logic in one go for Warlock Studio.
    '''
    try:
        frame_gen_factor, slowmotion = check_frame_generation_option(
            selected_generation_option)
        write_process_status(process_status_q, "Loading AI model")
        AI_instance = AI_interpolation(
            selected_AI_model, frame_gen_factor, selected_gpu, input_resize_factor, output_resize_factor)
        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            file_path = selected_file_list[file_number]
            current_file_number = file_number + 1
            # Branch between video and image: only video gets interpolation
            if check_if_file_is_video(file_path):
                try:
                    fluidframes_video_interpolate(
                        process_status_q, file_path, current_file_number, selected_output_path, AI_instance,
                        selected_AI_model, frame_gen_factor, slowmotion, selected_image_extension, selected_video_extension,
                        selected_video_codec, input_resize_factor, output_resize_factor, cpu_number, selected_keep_frames
                    )
                except Exception as file_error:
                    error_msg = f"Error processing {os_path_basename(file_path)}: {str(file_error)}"
                    log_and_report_error(error_msg)
                    write_process_status(
                        process_status_q, f"{ERROR_STATUS}{error_msg}")
                    continue  # Continue with next file
            else:
                # If an image, just no-op/fail, or could add image interpolation, but that's not FluidFrames
                write_process_status(
                    process_status_q, f"{current_file_number}. File is not a video; skipping interpolation for image files.")
        write_process_status(process_status_q, f"{COMPLETED_STATUS}")
    except Exception as exception:
        error_msg = str(exception)
        print(f"Error in FluidFrames interpolation pipeline: {error_msg}")
        log_and_report_error(f"Interpolation error: {error_msg}")
        write_process_status(
            process_status_q, f"{ERROR_STATUS}Interpolation error: {error_msg}")

# Helper for generation options string -> factor/slowmotion
# (straight copy from FluidFrames.py, rename as needed)


def check_frame_generation_option(selected_generation_option):
    slowmotion = False
    frame_gen_factor = 0
    if "Slowmotion" in selected_generation_option:
        slowmotion = True
    if "2" in selected_generation_option:
        frame_gen_factor = 2
    elif "4" in selected_generation_option:
        frame_gen_factor = 4
    elif "8" in selected_generation_option:
        frame_gen_factor = 8
    return frame_gen_factor, slowmotion

# Adapter: orchestration logic -- this wraps the full FluidFrames video flow
# (fluidframes_video_interpolate = mostly rename of video_frame_generation() + encoding etc; minimal adaptation)


def prepare_generated_frames_paths(
        base_path: str,
        selected_AI_model: str,
        selected_image_extension: str,
        frame_gen_factor: int
) -> list[str]:
    generated_frames_paths = [
        f"{base_path}_{selected_AI_model}_{i}{selected_image_extension}" for i in range(frame_gen_factor-1)]
    return generated_frames_paths


def prepare_output_video_frame_filenames(
        extracted_frames_paths: list[str],
        selected_AI_model: str,
        frame_gen_factor: int,
        selected_image_extension: str,
) -> list[str]:
    total_frames_paths = []
    how_many_frames = len(extracted_frames_paths)
    for index in range(how_many_frames - 1):
        frame_path = extracted_frames_paths[index]
        base_path = os_path_splitext(frame_path)[0]
        generated_frames_paths = prepare_generated_frames_paths(
            base_path, selected_AI_model, selected_image_extension, frame_gen_factor)
        total_frames_paths.append(frame_path)
        total_frames_paths.extend(generated_frames_paths)
    total_frames_paths.append(extracted_frames_paths[-1])
    return total_frames_paths


def prepare_output_video_frame_to_generate_filenames(
        extracted_frames_paths: list[str],
        selected_AI_model: str,
        frame_gen_factor: int,
        selected_image_extension: str,
) -> list[str]:
    only_generated_frames_paths = []
    how_many_frames = len(extracted_frames_paths)
    for index in range(how_many_frames - 1):
        frame_path = extracted_frames_paths[index]
        base_path = os_path_splitext(frame_path)[0]
        generated_frames_paths = prepare_generated_frames_paths(
            base_path, selected_AI_model, selected_image_extension, frame_gen_factor)
        only_generated_frames_paths.extend(generated_frames_paths)
    return only_generated_frames_paths


def fluidframes_video_interpolate(
        process_status_q, video_path, file_number, selected_output_path, AI_instance,
        selected_AI_model, frame_gen_factor, slowmotion, selected_image_extension,
        selected_video_extension, selected_video_codec, input_resize_factor, output_resize_factor, cpu_number, selected_keep_frames):
    # Step 1. Setup output dirs
    target_directory = prepare_output_video_directory_name(
        video_path, selected_output_path, selected_AI_model, frame_gen_factor, slowmotion, input_resize_factor, output_resize_factor)
    video_output_path = prepare_output_video_filename(
        video_path, selected_output_path, selected_AI_model, frame_gen_factor, slowmotion, input_resize_factor, output_resize_factor, selected_video_extension)
    # Step 2. Extract video frames
    write_process_status(
        process_status_q, f"{file_number}. Extracting video frames")
    extracted_frames_paths = extract_video_frames(
        process_status_q, file_number, target_directory, AI_instance, video_path, cpu_number, selected_image_extension)
    # Step 3. Prepare output/gen frame names
    total_frames_paths = prepare_output_video_frame_filenames(
        extracted_frames_paths, selected_AI_model, frame_gen_factor, selected_image_extension)
    only_generated_frames_paths = prepare_output_video_frame_to_generate_filenames(
        extracted_frames_paths, selected_AI_model, frame_gen_factor, selected_image_extension)
    # Step 4. Interpolated frames generation (calls AI orchestration)
    write_process_status(
        process_status_q, f"{file_number}. Video frame generation")
    global global_processing_times_list
    global_processing_times_list = []
    for frame_index in range(len(extracted_frames_paths)-1):
        frame_1_path = extracted_frames_paths[frame_index]
        frame_2_path = extracted_frames_paths[frame_index+1]
        frame_1 = image_read(frame_1_path)
        frame_2 = image_read(frame_2_path)
        start_timer = timer()
        generated_frames = AI_instance.AI_orchestration(frame_1, frame_2)
        # Save generated frames
        generated_frames_paths = prepare_generated_frames_paths(
            os_path_splitext(frame_1_path)[0], selected_AI_model, selected_image_extension, frame_gen_factor)
        for i, gen_frame in enumerate(generated_frames):
            image_write(generated_frames_paths[i], gen_frame)
        end_timer = timer()
        processing_time = end_timer - start_timer
        global_processing_times_list.append(processing_time)
    # Step 5. Save/copy/cleanup - cleanup handled at end of process
    # Step 6. Video encoding
    write_process_status(
        process_status_q, f"{file_number}. Encoding frame-generated video")
    video_encoding(
        process_status_q, video_path, video_output_path, total_frames_paths, selected_video_codec)
    copy_file_metadata(video_path, video_output_path)

    # Step 7. Cleanup after video interpolation processing
    if not selected_keep_frames and os_path_exists(target_directory):
        try:
            remove_directory(target_directory)
        except Exception as e:
            print(
                f"Warning: Could not remove directory {target_directory}: {str(e)}")

# ==== ORCHESTRATOR SECTION ====


def upscale_orchestrator(
        process_status_q: multiprocessing_Queue,
        selected_file_list: list,
        selected_output_path: str,
        selected_AI_model: str,
        selected_AI_multithreading: int,
        input_resize_factor: int,
        output_resize_factor: int,
        selected_gpu: str,
        tiles_resolution: int,
        selected_blending_factor: float,
        selected_keep_frames: bool,
        selected_image_extension: str,
        selected_video_extension: str,
        selected_video_codec: str,
        cpu_number: int,
) -> None:

    global global_status_lock
    global_status_lock = Lock()

    try:
        write_process_status(process_status_q, f"Loading AI model")

        AI_upscale_instance_list = [
            AI_upscale(selected_AI_model, selected_gpu,
                       input_resize_factor, output_resize_factor, tiles_resolution)
            for _ in range(selected_AI_multithreading)
        ]

        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            file_path = selected_file_list[file_number]
            file_number = file_number + 1

            if check_if_file_is_video(file_path):
                upscale_video(
                    process_status_q,
                    file_path,
                    file_number,
                    selected_output_path,
                    AI_upscale_instance_list,
                    selected_AI_model,
                    input_resize_factor,
                    output_resize_factor,
                    cpu_number,
                    selected_video_extension,
                    selected_blending_factor,
                    selected_AI_multithreading,
                    selected_keep_frames,
                    selected_video_codec
                )
            else:
                upscale_image(
                    process_status_q,
                    file_path,
                    file_number,
                    selected_output_path,
                    AI_upscale_instance_list[0],
                    selected_AI_model,
                    selected_image_extension,
                    input_resize_factor,
                    output_resize_factor,
                    selected_blending_factor
                )

        write_process_status(process_status_q, f"{COMPLETED_STATUS}")

    except Exception as exception:
        error_message = str(exception)

        if "cannot convert float NaN to integer" in error_message:
            write_process_status(
                process_status_q,
                f"{ERROR_STATUS}An error occurred during video upscaling, likely due to a GPU driver timeout.\n"
                "Restart the process without deleting the upscaled frames to resume and complete the upscaling."
            )
        else:
            log_and_report_error(error_message)
            write_process_status(
                process_status_q, f"{ERROR_STATUS} {error_message}")

# ==== IMAGE PROCESSING SECTION ====


def upscale_image(
        process_status_q: multiprocessing_Queue,
        image_path: str,
        file_number: int,
        selected_output_path: str,
        AI_instance: AI_upscale,
        selected_AI_model: str,
        selected_image_extension: str,
        input_resize_factor: int,
        output_resize_factor: int,
        selected_blending_factor: float
) -> None:

    starting_image = image_read(image_path)
    upscaled_image_path = prepare_output_image_filename(
        image_path, selected_output_path, selected_AI_model, input_resize_factor, output_resize_factor, selected_image_extension, selected_blending_factor)

    write_process_status(
        process_status_q, f"{file_number}. Enchanting your image. Be patient...")
    upscaled_image = AI_instance.AI_orchestration(starting_image)

    if selected_blending_factor > 0:
        blend_images_and_save(
            upscaled_image_path,
            starting_image,
            upscaled_image,
            selected_blending_factor,
            selected_image_extension
        )
    else:
        image_write(upscaled_image_path, upscaled_image,
                    selected_image_extension)

    copy_file_metadata(image_path, upscaled_image_path)

# ==== VIDEO PROCESSING SECTION ====


def upscale_video(
        process_status_q: multiprocessing_Queue,
        video_path: str,
        file_number: int,
        selected_output_path: str,
        AI_upscale_instance_list: list[AI_upscale],
        selected_AI_model: str,
        input_resize_factor: int,
        output_resize_factor: int,
        cpu_number: int,
        selected_video_extension: str,
        selected_blending_factor: float,
        selected_AI_multithreading: int,
        selected_keep_frames: bool,
        selected_video_codec: str
) -> None:

    # Internal functions

    def update_process_status_videos(
            process_status_q: multiprocessing_Queue,
            file_number: int,
    ) -> None:

        global global_upscaled_frames_paths
        global global_processing_times_list

        # Remaining frames
        total_frames_counter = len(global_upscaled_frames_paths)
        frames_already_upscaled_counter = len(
            [path for path in global_upscaled_frames_paths if os_path_exists(path)])
        frames_to_upscale_counter = len(
            [path for path in global_upscaled_frames_paths if not os_path_exists(path)])

        try:
            average_processing_time = numpy_mean(global_processing_times_list)
        except Exception:
            average_processing_time = 0.0

        remaining_frames = frames_to_upscale_counter
        remaining_time = calculate_time_to_complete_video(
            average_processing_time, remaining_frames)
        if remaining_time != "":
            percent_complete = (
                frames_already_upscaled_counter / total_frames_counter) * 100
            write_process_status(
                process_status_q, f"{file_number}.Enchanting your video. Be patient... {percent_complete:.2f}% ({remaining_time})")

    def save_multiple_upscaled_frame_async(
        starting_frames_to_save: list[numpy_ndarray],
        upscaled_frames_to_save: list[numpy_ndarray],
        upscaled_frame_paths_to_save: list[str],
        selected_blending_factor: float
    ) -> None:

        for frame_index, _ in enumerate(upscaled_frames_to_save):
            starting_frame = starting_frames_to_save[frame_index]
            upscaled_frame = upscaled_frames_to_save[frame_index]
            upscaled_frame_path = upscaled_frame_paths_to_save[frame_index]

            if selected_blending_factor > 0:
                blend_images_and_save(
                    upscaled_frame_path, starting_frame, upscaled_frame, selected_blending_factor)
            else:
                image_write(upscaled_frame_path, upscaled_frame)

    def save_frames_on_disk(
        starting_frames_to_save: list[numpy_ndarray],
        upscaled_frames_to_save: list[numpy_ndarray],
        upscaled_frame_paths_to_save: list[str],
        selected_blending_factor: float
    ) -> None:
        nonlocal writer_threads  # Access the outer scope variable

        # Fix 2.1: Track writer threads to ensure all frames are written before encoding
        t = Thread(
            target=save_multiple_upscaled_frame_async,
            args=(
                starting_frames_to_save,
                upscaled_frames_to_save,
                upscaled_frame_paths_to_save,
                selected_blending_factor
            )
        )
        writer_threads.append(t)
        t.start()

    def upscale_video_frames_async(
            process_status_q: multiprocessing_Queue,
            file_number: int,
            threads_number: int,
            AI_instance: AI_upscale,
            extracted_frames_paths: list[str],
            upscaled_frame_paths: list[str],
            selected_blending_factor: float,
    ) -> None:

        global global_processing_times_list
        global global_can_i_update_status
        global global_status_lock  # Fix 2.3: Add thread lock for safe status updates

        starting_frames_to_save = []
        upscaled_frames_to_save = []
        upscaled_frame_paths_to_save = []

        for frame_index in range(len(extracted_frames_paths)):
            frame_path = extracted_frames_paths[frame_index]
            upscaled_frame_path = upscaled_frame_paths[frame_index]
            already_upscaled = os_path_exists(upscaled_frame_path)

            if already_upscaled == False:
                start_timer = timer()

                # Upscale frame with memory optimization
                starting_frame = image_read(frame_path)
                try:
                    upscaled_frame = AI_instance.AI_orchestration(
                        starting_frame)
                except Exception as e:
                    # Fix 3.2: Handle GPU memory errors with retry logic
                    if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                        print(
                            f"[GPU] Memory error detected, reducing tiles resolution and retrying...")
                        original_tiles = AI_instance.max_resolution
                        AI_instance.max_resolution = max(
                            128, AI_instance.max_resolution // 2)
                        try:
                            upscaled_frame = AI_instance.AI_orchestration(
                                starting_frame)
                            print(
                                f"[GPU] Retry successful with reduced tiles resolution: {AI_instance.max_resolution}")
                        except Exception as retry_error:
                            AI_instance.max_resolution = original_tiles  # Restore original
                            raise retry_error
                    else:
                        raise e

                # Adding frames in list to save
                starting_frames_to_save.append(starting_frame)
                upscaled_frames_to_save.append(upscaled_frame)
                upscaled_frame_paths_to_save.append(upscaled_frame_path)

                # Calculate processing time and update process status
                end_timer = timer()
                processing_time = (end_timer - start_timer)/threads_number
                global_processing_times_list.append(processing_time)

                # Fix 3.1: Write frames immediately to disk to reduce memory usage
                if (frame_index + 1) % MULTIPLE_FRAMES_TO_SAVE == 0:
                    # Save frames present in RAM on disk
                    save_frames_on_disk(starting_frames_to_save, upscaled_frames_to_save,
                                        upscaled_frame_paths_to_save, selected_blending_factor)
                    # Clear frame lists to free memory
                    starting_frames_to_save = []
                    upscaled_frames_to_save = []
                    upscaled_frame_paths_to_save = []
                    # Optimize memory usage
                    optimize_memory_usage()

                    # Fix 2.3: Use thread lock to safely modify status flag
                    with global_status_lock:
                        global_can_i_update_status = not global_can_i_update_status
                        if global_can_i_update_status:
                            update_process_status_videos(
                                process_status_q, file_number)
                            if len(global_processing_times_list) >= 100:
                                global_processing_times_list = []

        if len(upscaled_frame_paths_to_save) > 0:
            # Save frames still present in RAM on disk
            save_frames_on_disk(starting_frames_to_save, upscaled_frames_to_save,
                                upscaled_frame_paths_to_save, selected_blending_factor)
            starting_frames_to_save = []
            upscaled_frames_to_save = []
            upscaled_frame_paths_to_save = []
            # Final memory optimization
            optimize_memory_usage()

    def upscale_video_frames(
            process_status_q: multiprocessing_Queue,
            file_number: int,
            AI_upscale_instance_list: list[AI_upscale],
            extracted_frames_paths: list[str],
            upscaled_frame_paths: list[str],
            threads_number: int,
            selected_blending_factor: float,
    ) -> None:

        global global_upscaled_frames_paths
        global global_processing_times_list
        global global_can_i_update_status

        global_upscaled_frames_paths = upscaled_frame_paths
        global_processing_times_list = []
        global_can_i_update_status = False

        chunk_size = len(extracted_frames_paths) // threads_number
        extracted_frame_list_chunks = [extracted_frames_paths[i:i + chunk_size]
                                       for i in range(0, len(extracted_frames_paths), chunk_size)]
        upscaled_frame_list_chunks = [upscaled_frame_paths[i:i + chunk_size]
                                      for i in range(0, len(upscaled_frame_paths), chunk_size)]

        write_process_status(
            process_status_q, f"{file_number}. Upscaling video. Be patient ({threads_number} threads)")
        with ThreadPool(threads_number) as pool:
            pool.starmap(
                upscale_video_frames_async,
                zip(
                    repeat(process_status_q),
                    repeat(file_number),
                    repeat(threads_number),
                    AI_upscale_instance_list,
                    extracted_frame_list_chunks,
                    upscaled_frame_list_chunks,
                    repeat(selected_blending_factor),
                )
            )

    def check_forgotten_video_frames(
            process_status_q: multiprocessing_Queue,
            file_number: int,
            AI_upscale_instance_list: AI_upscale,
            extracted_frames_paths: list[str],
            upscaled_frame_paths: list[str],
            selected_blending_factor: float,
            threads_number: int = 1,
    ):

        sleep(1)

        # Check if all the upscaled frames exist
        frame_path_todo_list = []
        upscaled_frame_path_todo_list = []

        for frame_index in range(len(upscaled_frame_paths)):
            extracted_frames_path = extracted_frames_paths[frame_index]
            upscaled_frame_path = upscaled_frame_paths[frame_index]

            if not os_path_exists(upscaled_frame_path):
                frame_path_todo_list.append(extracted_frames_path)
                upscaled_frame_path_todo_list.append(upscaled_frame_path)

        if len(upscaled_frame_path_todo_list) > 0:
            upscale_video_frames(
                process_status_q,
                file_number,
                AI_upscale_instance_list,
                extracted_frames_paths,
                upscaled_frame_paths,
                threads_number,
                selected_blending_factor
            )

    # Main function

    # Fix 2.1: Initialize writer_threads list to track frame writing threads
    writer_threads = []

    # 1.Preparation
    target_directory = prepare_output_video_directory_name(
        video_path, selected_output_path, selected_AI_model, 1, False, input_resize_factor, output_resize_factor)
    video_output_path = prepare_output_video_filename(video_path, selected_output_path, selected_AI_model,
                                                      1, False, input_resize_factor, output_resize_factor, selected_video_extension)

    # 2. Resume upscaling OR Extract video frames
    video_upscale_continue = check_video_upscaling_resume(
        target_directory, selected_AI_model)
    if video_upscale_continue:
        write_process_status(
            process_status_q, f"{file_number}. Resume video upscaling")
        extracted_frames_paths = get_video_frames_for_upscaling_resume(
            target_directory, selected_AI_model)
    else:
        write_process_status(
            process_status_q, f"{file_number}. Extracting video frames")
        extracted_frames_paths = extract_video_frames(
            process_status_q, file_number, target_directory, AI_upscale_instance_list[0], video_path, cpu_number, ".jpg")

    upscaled_frame_paths = [prepare_output_video_frame_filename(
        frame_path, selected_AI_model, input_resize_factor, output_resize_factor, selected_blending_factor) for frame_path in extracted_frames_paths]

    # 3. Check if video need tiles OR video multithreading upscale
    multiframes_supported_by_gpu = AI_upscale_instance_list[0].calculate_multiframes_supported_by_gpu(
        extracted_frames_paths[0])
    threads_number = min(multiframes_supported_by_gpu,
                         selected_AI_multithreading)
    if threads_number <= 0:
        threads_number = 1

    # 4. Upscaling video frames
    write_process_status(process_status_q, f"{file_number}. Upscaling video")
    upscale_video_frames(process_status_q, file_number, AI_upscale_instance_list,
                         extracted_frames_paths, upscaled_frame_paths, threads_number, selected_blending_factor)

    # 5. Check for forgotten video frames
    check_forgotten_video_frames(process_status_q, file_number, AI_upscale_instance_list,
                                 extracted_frames_paths, upscaled_frame_paths, selected_blending_factor)

    # Fix 2.1: Wait for all writer threads to complete before encoding
    for t in writer_threads:
        t.join()

    # 6. Video encoding
    write_process_status(
        process_status_q, f"{file_number}. Encoding upscaled video")
    video_encoding(process_status_q, video_path, video_output_path,
                   upscaled_frame_paths, selected_video_codec)
    copy_file_metadata(video_path, video_output_path)

    # 7. Delete frames folder
    if selected_keep_frames == False:
        if os_path_exists(target_directory):
            try:
                remove_directory(target_directory)
            except Exception as e:
                print(
                    f"Warning: Could not remove directory {target_directory}: {str(e)}")


# ==== GUI UTILITIES SECTION ====

def check_if_file_is_video(file: str) -> bool:
    return any(video_extension in file for video_extension in supported_video_extensions)


def validate_configuration() -> bool:
    """Comprehensive configuration validation."""
    errors = []

    # Check AI model compatibility
    if selected_AI_model == MENU_LIST_SEPARATOR[0]:
        errors.append("Invalid AI model selected")

    # Check frame generation compatibility
    if selected_AI_model in RIFE_models_list and selected_frame_generation_option == "OFF":
        errors.append("Frame generation option required for RIFE models")

    # Check system requirements
    if not validate_system_requirements():
        errors.append("System requirements not met")

    if errors:
        for error in errors:
            log_and_report_error(error)
        return False
    return True


def user_input_checks() -> bool:
    global selected_file_list
    global selected_AI_model
    global selected_image_extension
    global tiles_resolution
    global input_resize_factor
    global output_resize_factor

    # Enhanced file validation
    try:
        selected_file_list = file_widget.get_selected_file_list()
    except Exception:
        info_message.set("Please select a file")
        return False

    if len(selected_file_list) <= 0:
        info_message.set("Please select a file")
        return False

    # Validate file paths and accessibility
    if not validate_file_paths(selected_file_list):
        info_message.set("File validation failed. Check log for details.")
        return False

    # Validate output path
    if not validate_output_path(selected_output_path.get()):
        info_message.set("Output path validation failed")
        return False

    # Additional configuration validation
    if not validate_configuration():
        info_message.set("Configuration validation failed")
        return False

    # AI model
    if selected_AI_model == MENU_LIST_SEPARATOR[0]:
        info_message.set("Please select the AI model")
        return False

    # Input resize factor
    try:
        input_resize_factor = int(
            float(str(selected_input_resize_factor.get())))
    except (ValueError, TypeError):
        info_message.set("Input resolution % must be a number")
        return False

    if input_resize_factor > 0:
        input_resize_factor = input_resize_factor/100
    else:
        info_message.set("Input resolution % must be a value > 0")
        return False

    # Output resize factor
    try:
        output_resize_factor = int(
            float(str(selected_output_resize_factor.get())))
    except (ValueError, TypeError):
        info_message.set("Output resolution % must be a number")
        return False

    if output_resize_factor > 0:
        output_resize_factor = output_resize_factor/100
    else:
        info_message.set("Output resolution % must be a value > 0")
        return False

# VRAM limiter
    try:
        vram_gb = int(float(str(selected_VRAM_limiter.get())))
        if vram_gb <= 0:
            info_message.set("GPU VRAM value must be a value > 0")
            return False

        vram_multiplier = VRAM_model_usage.get(selected_AI_model)
        if vram_multiplier is None:
            vram_multiplier = 1  # Default for interpolation models or unknowns

        # El cálculo original parece confuso. Esta es una interpretación más clara:
        # Se asume que el VRAM Limiter es la VRAM en GB y se multiplica por un factor y 100.
        # Si el modelo 'RealESR_Gx4' (factor 2.2) y VRAM es 4GB, tiles_resolution sería ~880.
        selected_vram_factor = vram_multiplier * vram_gb
        tiles_resolution = int(selected_vram_factor * 100)

    except (ValueError, TypeError):
        info_message.set("GPU VRAM value must be a number")
        return False

    return True


def show_error_message(exception: str) -> None:
    try:
        messageBox_title = "Upscale error"
        messageBox_subtitle = "Please report the error on Github, SourceForge or write to us on negroayub97@gmail.com."
        messageBox_text = f"\n {str(exception)} \n"

        MessageBox(
            messageType="error",
            title=messageBox_title,
            subtitle=messageBox_subtitle,
            default_value=None,
            option_list=[messageBox_text]
        )
    except Exception as e:
        print(f"[ERROR] Could not show error message: {str(e)}")
        print(f"[ERROR] Original error was: {exception}")


def get_upscale_factor() -> int:
    global selected_AI_model
    upscale_factor = 1  # Default value for most models
    if MENU_LIST_SEPARATOR[0] in selected_AI_model:
        upscale_factor = 0
    elif 'x1' in selected_AI_model:
        upscale_factor = 1
    elif 'x2' in selected_AI_model:
        upscale_factor = 2
    elif 'x4' in selected_AI_model:
        upscale_factor = 4
    elif selected_AI_model in RIFE_models_list:
        # RIFE interpolation models do not use upscaling; fallback to 1, not used
        upscale_factor = 1
    return upscale_factor


def open_files_action():

    def check_supported_selected_files(uploaded_file_list: list) -> list:
        return [file for file in uploaded_file_list if any(supported_extension in file for supported_extension in supported_file_extensions)]

    info_message.set("Selecting files")

    uploaded_files_list = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)

    print("> Uploaded files: " + str(uploaded_files_counter) +
          " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:

        upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()

        global file_widget
        file_widget = FileWidget(
            master=window,
            selected_file_list=supported_files_list,
            upscale_factor=upscale_factor,
            input_resize_factor=input_resize_factor,
            output_resize_factor=output_resize_factor,
            fg_color=background_color,
            bg_color=background_color
        )
        file_widget.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)
        info_message.set("Ready to be being enchanted!")
    else:
        info_message.set("Not supported files :(")


def open_output_path_action():
    asked_selected_output_path = filedialog.askdirectory()
    if asked_selected_output_path == "":
        selected_output_path.set(OUTPUT_PATH_CODED)
    else:
        selected_output_path.set(asked_selected_output_path)


# ==== GUI MENU SELECTION SECTION ====

def select_AI_from_menu(selected_option: str) -> None:
    global selected_AI_model
    selected_AI_model = selected_option
    update_file_widget(1, 2, 3)

    # --- Improved: instant dynamic refresh for conditional FluidFrames menus ---
    clear_dynamic_menus()

    # FluidFrames/RIFE: Show frame generation menu, otherwise show blending
    if selected_AI_model in RIFE_models_list:
        place_frame_generation_menu()
    else:
        place_AI_blending_menu()
    # Always restore other key controls
    place_AI_multithreading_menu()
    place_input_output_resolution_textboxs()
    place_gpu_gpuVRAM_menus()
    place_video_codec_keep_frames_menus()
    place_image_video_output_menus()
    place_output_path_textbox()
    place_message_label()
    place_upscale_button()


def clear_dynamic_menus() -> None:
    """Clear any existing dynamic menus from the interface"""
    # This will be called to clear menus before placing new ones
    try:
        for widget in window.winfo_children():
            widget_info = widget.place_info()
            if widget_info and float(widget_info.get('rely', 0)) == row2:
                widget.place_forget()
    except Exception:
        pass


def select_AI_multithreading_from_menu(selected_option: str) -> None:
    global selected_AI_multithreading
    if selected_option == "OFF":
        selected_AI_multithreading = 1
    else:
        selected_AI_multithreading = int(selected_option.split()[0])


def select_blending_from_menu(selected_option: str) -> None:
    global selected_blending_factor

    match selected_option:
        case "OFF": selected_blending_factor = 0
        case "Low":      selected_blending_factor = 0.3
        case "Medium":   selected_blending_factor = 0.5
        case "High":     selected_blending_factor = 0.7


def select_gpu_from_menu(selected_option: str) -> None:
    global selected_gpu
    selected_gpu = selected_option


def select_save_frame_from_menu(selected_option: str):
    global selected_keep_frames
    if selected_option == "ON":
        selected_keep_frames = True
    elif selected_option == "OFF":
        selected_keep_frames = False


def select_image_extension_from_menu(selected_option: str) -> None:
    global selected_image_extension
    selected_image_extension = selected_option


def select_video_extension_from_menu(selected_option: str) -> None:
    global selected_video_extension
    selected_video_extension = selected_option


def select_video_codec_from_menu(selected_option: str) -> None:
    global selected_video_codec
    selected_video_codec = selected_option


def select_frame_generation_from_menu(selected_option: str) -> None:
    global selected_frame_generation_option
    selected_frame_generation_option = selected_option

# ==== GUI LAYOUT SECTION ====

# --- FLUIDFRAMES: Handle Interpolator menus/logic ---


def is_rife_model_selected():
    global selected_AI_model
    return selected_AI_model in RIFE_models_list


def get_generation_options_list():
    # Only show on RIFE-based
    if is_rife_model_selected():
        return frame_generation_options_list
    return ["OFF"]


def place_dynamic_rife_interpolator():
    clear_dynamic_menus()
    if is_rife_model_selected():
        place_frame_generation_menu()
    else:
        place_AI_blending_menu()

# END FLUIDFRAMES


def place_loadFile_section():
    background = CTkFrame(
        master=window, fg_color=background_color, corner_radius=1)

    text_drop = (" SUPPORTED FILES \n\n "
                 + "IMAGES • jpg png tif bmp webp heic \n "
                 + "VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp ")

    input_file_text = CTkLabel(
        master=window,
        text=text_drop,
        fg_color=background_color,
        bg_color=background_color,
        text_color=text_color,
        width=300,
        height=150,
        font=bold13,
        anchor="center"
    )

    input_file_button = CTkButton(
        master=window,
        command=open_files_action,
        text="SELECT FILES",
        width=140,
        height=30,
        font=bold12,
        border_width=1,
        corner_radius=1,
        fg_color="#282828",
        text_color="#E0E0E0",
        border_color="#ECD125"
    )

    background.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)
    input_file_text.place(relx=0.25, rely=0.4,  anchor="center")
    input_file_button.place(relx=0.25, rely=0.5, anchor="center")


def place_app_name():
    background = CTkFrame(
        master=window, fg_color=background_color, corner_radius=1)
    app_name_label = CTkLabel(
        master=window,
        text=app_name + " " + version,
        fg_color=background_color,
        text_color=app_name_color,
        font=bold20,
        anchor="w"
    )
    background.place(relx=0.5, rely=0.0, relwidth=0.5, relheight=1.0)
    app_name_label.place(relx=column_1 - 0.05, rely=0.04, anchor="center")


def place_AI_menu():

    def open_info_AI_model():
        option_list = [
            "\n IRCNN_Mx1 | IRCNN_Lx1 \n"
            "\n • Simple and lightweight AI models\n"
            " • Year: 2017\n"
            " • Function: Denoising\n",

            "\n RealESR_Gx4 | RealESR_Animex4 \n"
            "\n • Fast and lightweight AI models\n"
            " • Year: 2022\n"
            " • Function: Upscaling\n",

            "\n BSRGANx2 | BSRGANx4 | RealESRGANx4 | RealESRNetx4 \n"
            "\n • Complex and heavy AI models\n"
            " • Year: 2020\n"
            " • Function: High-quality upscaling\n",

            "\n RIFE | RIFE Lite\n" +
            "   • The complete RIFE AI model & Lite version\n" +
            "   • Excellent frame generation quality\n" +
            "   • Lite is 10% faster than full model\n" +
            "   • Recommended for GPUs with VRAM < 4GB \n",
        ]

        MessageBox(
            messageType="info",
            title="AI model",
            subtitle="This widget allows to choose between different AI models for upscaling",
            default_value=None,
            option_list=option_list
        )

    widget_row = row1
    background = create_option_background()
    background.place(relx=0.75, rely=widget_row,
                     relwidth=0.48, anchor="center")

    info_button = create_info_button(open_info_AI_model, "AI model")
    option_menu = create_option_menu(
        select_AI_from_menu, AI_models_list, default_AI_model)

    info_button.place(relx=column_info1, rely=widget_row -
                      0.003, anchor="center")
    option_menu.place(relx=column_3_5,   rely=widget_row,
                      anchor="center")


def place_frame_generation_menu():

    def open_info_frame_generation():
        option_list = [
            "\n FRAME GENERATION\n" +
            "   • x2 - doubles video framerate • 30fps => 60fps\n" +
            "   • x4 - quadruples video framerate • 30fps => 120fps\n" +
            "   • x8 - octuplicate video framerate • 30fps => 240fps\n",

            "\n SLOWMOTION (no audio)\n" +
            "   • Slowmotion x2 - slowmotion effect by a factor of 2\n" +
            "   • Slowmotion x4 - slowmotion effect by a factor of 4\n" +
            "   • Slowmotion x8 - slowmotion effect by a factor of 8\n"
        ]

        MessageBox(
            messageType="info",
            title="AI frame generation",
            subtitle=" This widget allows to choose between different AI frame generation option",
            default_value=None,
            option_list=option_list
        )

    widget_row = row2
    background = create_option_background()
    background.place(relx=0.75, rely=widget_row,
                     relwidth=0.48, anchor="center")

    info_button = create_info_button(
        open_info_frame_generation, "Frame generation")
    option_menu = create_option_menu(
        select_frame_generation_from_menu, frame_generation_options_list, "OFF")

    info_button.place(relx=column_info1, rely=widget_row -
                      0.003, anchor="center")
    option_menu.place(relx=column_3_5,   rely=widget_row, anchor="center")


def place_AI_blending_menu():

    def open_info_AI_blending():
        option_list = [
            " Blending combines the upscaled image produced by AI with the original image",

            " \n BLENDING OPTIONS\n" +
            "  • [OFF] No blending is applied\n" +
            "  • [Low] The result favors the upscaled image, with a slight touch of the original\n" +
            "  • [Medium] A balanced blend of the original and upscaled images\n" +
            "  • [High] The result favors the original image, with subtle enhancements from the upscaled version\n",

            " \n NOTES\n" +
            "  • Can enhance the quality of the final result\n" +
            "  • Especially effective when using the tiling/merging function (useful for low VRAM)\n" +
            "  • Particularly helpful at low input resolution percentages (<50%)\n",
        ]

        MessageBox(
            messageType="info",
            title="AI blending",
            subtitle="This widget allows you to choose the blending between the upscaled and original image/frame",
            default_value=None,
            option_list=option_list
        )

    widget_row = row2

    background = create_option_background()
    background.place(relx=0.75, rely=widget_row,
                     relwidth=0.48, anchor="center")

    info_button = create_info_button(open_info_AI_blending, "AI blending")
    option_menu = create_option_menu(
        select_blending_from_menu, blending_list, default_blending)

    info_button.place(relx=column_info1, rely=widget_row -
                      0.003, anchor="center")
    option_menu.place(relx=column_3_5,   rely=widget_row,
                      anchor="center")


def place_AI_multithreading_menu():

    def open_info_AI_multithreading():
        option_list = [
            " This option can enhance video upscaling performance, especially on powerful GPUs.",

            " \n AI MULTITHREADING OPTIONS\n"
            + "  • OFF - Processes one frame at a time.\n"
            + "  • 2 threads - Processes two frames simultaneously.\n"
            + "  • 4 threads - Processes four frames simultaneously.\n"
            + "  • 6 threads - Processes six frames simultaneously.\n"
            + "  • 8 threads - Processes eight frames simultaneously.\n",

            " \n NOTES\n"
            + "  • Higher thread counts increase CPU, GPU, and RAM usage.\n"
            + "  • The GPU may be heavily stressed, potentially reaching high temperatures.\n"
            + "  • Monitor your system's temperature to prevent overheating.\n"
            + "  • If the chosen thread count exceeds GPU capacity, the app automatically selects an optimal value.\n",
        ]

        MessageBox(
            messageType="info",
            title="AI multithreading (EXPERIMENTAL)",
            subtitle="This widget allows to choose how many video frames are upscaled simultaneously",
            default_value=None,
            option_list=option_list
        )

    widget_row = row3
    background = create_option_background()
    background.place(relx=0.75, rely=widget_row,
                     relwidth=0.48, anchor="center")

    info_button = create_info_button(
        open_info_AI_multithreading, "AI multithreading")
    option_menu = create_option_menu(
        select_AI_multithreading_from_menu, AI_multithreading_list, default_AI_multithreading)

    info_button.place(relx=column_info1, rely=widget_row -
                      0.003, anchor="center")
    option_menu.place(relx=column_3_5,   rely=widget_row,
                      anchor="center")


def place_input_output_resolution_textboxs():

    def open_info_input_resolution():
        option_list = [
            " A high value (>70%) will create high quality photos/videos but will be slower",
            " While a low value (<40%) will create good quality photos/videos but will much faster",

            " \n For example, for a 1080p (1920x1080) image/video\n" +
            " • Input resolution 25% => input to AI 270p (480x270)\n" +
            " • Input resolution 50% => input to AI 540p (960x540)\n" +
            " • Input resolution 75% => input to AI 810p (1440x810)\n" +
            " • Input resolution 100% => input to AI 1080p (1920x1080) \n",
        ]

        MessageBox(
            messageType="info",
            title="Input resolution %",
            subtitle="This widget allows to choose the resolution input to the AI",
            default_value=None,
            option_list=option_list
        )

    def open_info_output_resolution():
        option_list = [
            " TBD ",
        ]

        MessageBox(
            messageType="info",
            title="Output resolution %",
            subtitle="This widget allows to choose upscaled files resolution",
            default_value=None,
            option_list=option_list
        )

    widget_row = row4

    background = create_option_background()
    background.place(relx=0.75, rely=widget_row,
                     relwidth=0.48, anchor="center")

    # Input resolution %
    info_button = create_info_button(
        open_info_input_resolution, "Input resolution")
    option_menu = create_text_box(
        selected_input_resize_factor, width=little_textbox_width)

    info_button.place(relx=column_info1, rely=widget_row -
                      0.003, anchor="center")
    option_menu.place(relx=column_1_5,   rely=widget_row,
                      anchor="center")

    # Output resolution %
    info_button = create_info_button(
        open_info_output_resolution, "Output resolution")
    option_menu = create_text_box(
        selected_output_resize_factor, width=little_textbox_width)

    info_button.place(relx=column_info2, rely=widget_row -
                      0.003, anchor="center")
    option_menu.place(relx=column_3,     rely=widget_row,
                      anchor="center")


def place_gpu_gpuVRAM_menus():

    def open_info_gpu():
        option_list = [
            "\n It is possible to select up to 4 GPUs for AI processing\n" +
            "  • Auto (the app will select the most powerful GPU)\n" +
            "  • GPU 1 (GPU 0 in Task manager)\n" +
            "  • GPU 2 (GPU 1 in Task manager)\n" +
            "  • GPU 3 (GPU 2 in Task manager)\n" +
            "  • GPU 4 (GPU 3 in Task manager)\n",

            "\n NOTES\n" +
            "  • Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be\n" +
            "  • For optimal performance, it is essential to regularly update your GPUs drivers\n" +
            "  • Selecting a GPU not present in the PC will cause the app to use the CPU for AI processing\n"
        ]

        MessageBox(
            messageType="info",
            title="GPU",
            subtitle="This widget allows to select the GPU for AI upscale",
            default_value=None,
            option_list=option_list
        )

    def open_info_vram_limiter():
        option_list = [
            " Make sure to enter the correct value based on the selected GPU's VRAM",
            " Setting a value higher than the available VRAM may cause upscale failure",
            " For integrated GPUs (Intel HD series • Vega 3, 5, 7), select 2 GB to avoid issues",
        ]

        MessageBox(
            messageType="info",
            title="GPU VRAM (GB)",
            subtitle="This widget allows to set a limit on the GPU VRAM memory usage",
            default_value=None,
            option_list=option_list
        )

    widget_row = row5

    background = create_option_background()
    background.place(relx=0.75, rely=widget_row,
                     relwidth=0.48, anchor="center")

    # GPU
    info_button = create_info_button(open_info_gpu, "GPU")
    option_menu = create_option_menu(
        select_gpu_from_menu, gpus_list, default_gpu, width=little_menu_width)

    info_button.place(relx=column_info1,
                      rely=widget_row - 0.003, anchor="center")
    option_menu.place(relx=column_1_4, rely=widget_row,  anchor="center")

    # GPU VRAM
    info_button = create_info_button(open_info_vram_limiter, "GPU VRAM (GB)")
    option_menu = create_text_box(
        selected_VRAM_limiter, width=little_textbox_width)

    info_button.place(relx=column_info2, rely=widget_row -
                      0.003, anchor="center")
    option_menu.place(relx=column_3,     rely=widget_row,
                      anchor="center")


def place_image_video_output_menus():

    def open_info_image_output():
        option_list = [
            " \n PNG\n"
            " • Very good quality\n"
            " • Slow and heavy file\n"
            " • Supports transparent images\n"
            " • Lossless compression (no quality loss)\n"
            " • Ideal for graphics, web images, and screenshots\n",

            " \n JPG\n"
            " • Good quality\n"
            " • Fast and lightweight file\n"
            " • Lossy compression (some quality loss)\n"
            " • Ideal for photos and web images\n"
            " • Does not support transparency\n",

            " \n BMP\n"
            " • Highest quality\n"
            " • Slow and heavy file\n"
            " • Uncompressed format (large file size)\n"
            " • Ideal for raw images and high-detail graphics\n"
            " • Does not support transparency\n",

            " \n TIFF\n"
            " • Highest quality\n"
            " • Very slow and heavy file\n"
            " • Supports both lossless and lossy compression\n"
            " • Often used in professional photography and printing\n"
            " • Supports multiple layers and transparency\n",
        ]

        MessageBox(
            messageType="info",
            title="Image output",
            subtitle="This widget allows to choose the extension of upscaled images",
            default_value=None,
            option_list=option_list
        )

    def open_info_video_extension():
        option_list = [
            " \n MP4\n"
            " • Most widely supported format\n"
            " • Good quality with efficient compression\n"
            " • Fast and lightweight file\n"
            " • Ideal for streaming and general use\n",

            " \n MKV\n"
            " • High-quality format with multiple audio and subtitle tracks support\n"
            " • Larger file size compared to MP4\n"
            " • Supports almost any codec\n"
            " • Ideal for high-quality videos and archiving\n",

            " \n AVI\n"
            " • Older format with high compatibility\n"
            " • Larger file size due to less efficient compression\n"
            " • Supports multiple codecs but lacks modern features\n"
            " • Ideal for older devices and raw video storage\n",

            " \n MOV\n"
            " • High-quality format developed by Apple\n"
            " • Large file size due to less compression\n"
            " • Best suited for editing and high-quality playback\n"
            " • Compatible mainly with macOS and iOS devices\n",
        ]

        MessageBox(
            messageType="info",
            title="Video output",
            subtitle="This widget allows to choose the extension of the upscaled video",
            default_value=None,
            option_list=option_list
        )

    widget_row = row6

    background = create_option_background()
    background.place(relx=0.75, rely=widget_row,
                     relwidth=0.48, anchor="center")

    # Image output
    info_button = create_info_button(open_info_image_output, "Image output")
    option_menu = create_option_menu(select_image_extension_from_menu,
                                     image_extension_list, default_image_extension, width=little_menu_width)
    info_button.place(relx=column_info1,
                      rely=widget_row - 0.003, anchor="center")
    option_menu.place(relx=column_1_4, rely=widget_row,
                      anchor="center")

    # Video output
    info_button = create_info_button(open_info_video_extension, "Video output")
    option_menu = create_option_menu(select_video_extension_from_menu,
                                     video_extension_list, default_video_extension, width=little_menu_width)
    info_button.place(relx=column_info2,
                      rely=widget_row - 0.003, anchor="center")
    option_menu.place(relx=column_2_9, rely=widget_row,
                      anchor="center")


def place_video_codec_keep_frames_menus():

    def open_info_video_codec():
        option_list = [
            " \n SOFTWARE ENCODING (CPU)\n"
            " • x264 | H.264 software encoding\n"
            " • x265 | HEVC (H.265) software encoding\n",

            " \n NVIDIA GPU ENCODING (NVENC - Optimized for NVIDIA GPU)\n"
            " • h264_nvenc | H.264 hardware encoding\n"
            " • hevc_nvenc | HEVC (H.265) hardware encoding\n",

            " \n AMD GPU ENCODING (AMF - Optimized for AMD GPU)\n"
            " • h264_amf | H.264 hardware encoding\n"
            " • hevc_amf | HEVC (H.265) hardware encoding\n",

            " \n INTEL GPU ENCODING (QSV - Optimized for Intel GPU)\n"
            " • h264_qsv | H.264 hardware encoding\n"
            " • hevc_qsv | HEVC (H.265) hardware encoding\n"
        ]

        MessageBox(
            messageType="info",
            title="Video codec",
            subtitle="This widget allows to choose video codec for upscaled video",
            default_value=None,
            option_list=option_list
        )

    def open_info_keep_frames():
        option_list = [
            "\n ON \n" +
            " The app does NOT delete the video frames after creating the upscaled video \n",

            "\n OFF \n" +
            " The app deletes the video frames after creating the upscaled video \n"
        ]

        MessageBox(
            messageType="info",
            title="Keep video frames",
            subtitle="This widget allows to choose to keep video frames",
            default_value=None,
            option_list=option_list
        )

    widget_row = row7

    background = create_option_background()
    background.place(relx=0.75, rely=widget_row,
                     relwidth=0.48, anchor="center")

    # Video codec
    info_button = create_info_button(open_info_video_codec, "Video codec")
    option_menu = create_option_menu(
        select_video_codec_from_menu, video_codec_list, default_video_codec, width=little_menu_width)
    info_button.place(relx=column_info1,
                      rely=widget_row - 0.003, anchor="center")
    option_menu.place(relx=column_1_4, rely=widget_row,
                      anchor="center")

    # Keep frames
    info_button = create_info_button(open_info_keep_frames, "Keep frames")
    option_menu = create_option_menu(
        select_save_frame_from_menu, keep_frames_list, default_keep_frames, width=little_menu_width)
    info_button.place(relx=column_info2,
                      rely=widget_row - 0.003, anchor="center")
    option_menu.place(relx=column_2_9, rely=widget_row,
                      anchor="center")


def place_output_path_textbox():

    def open_info_output_path():
        option_list = [
            "\n The default path is defined by the input files."
            + "\n For example: selecting a file from the Download folder,"
            + "\n the app will save upscaled files in the Download folder \n",

            " Otherwise it is possible to select the desired path using the SELECT button",
        ]

        MessageBox(
            messageType="info",
            title="Output path",
            subtitle="This widget allows to choose upscaled files path",
            default_value=None,
            option_list=option_list
        )

    background = create_option_background()
    info_button = create_info_button(open_info_output_path, "Output path")
    option_menu = create_text_box_output_path(selected_output_path)
    active_button = create_active_button(
        command=open_output_path_action, text="SELECT", width=60, height=25)

    background.place(relx=0.75,                 rely=row10,
                     relwidth=0.48, anchor="center")
    info_button.place(relx=column_info1,         rely=row10 -
                      0.003,           anchor="center")
    active_button.place(relx=column_info1 + 0.052,
                        rely=row10,                   anchor="center")
    option_menu.place(relx=column_2 - 0.008,     rely=row10,
                      anchor="center")


def place_message_label():
    message_label = CTkLabel(
        master=window,
        textvariable=info_message,
        height=26,
        width=200,
        font=bold11,
        fg_color="#ffbf00",
        text_color="#000000",
        anchor="center",
        corner_radius=1
    )
    message_label.place(relx=0.83, rely=0.9495, anchor="center")


def place_stop_button():
    stop_button = create_active_button(
        command=stop_button_command,
        text="STOP",
        icon=stop_icon,
        width=140,
        height=30,
        border_color="#EC1D1D"
    )
    stop_button.place(relx=0.75 - 0.1, rely=0.95, anchor="center")


def place_upscale_button():
    upscale_button = create_active_button(
        command=upscale_button_command,
        text="Make Magic",
        icon=upscale_icon,
        width=140,
        height=30
    )
    upscale_button.place(relx=0.75 - 0.1, rely=0.95, anchor="center")


# ==== MAIN APPLICATION SECTION ====

def on_app_close() -> None:
    # Clean up logger
    logging.shutdown()
    window.grab_release()
    window.destroy()

    global selected_AI_model
    global selected_AI_multithreading
    global selected_gpu
    global selected_blending_factor
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global tiles_resolution
    global input_resize_factor

    AI_model_to_save = f"{selected_AI_model}"
    gpu_to_save = selected_gpu
    image_extension_to_save = selected_image_extension
    video_extension_to_save = selected_video_extension
    video_codec_to_save = selected_video_codec
    blending_to_save = {0: "OFF", 0.3: "Low", 0.5: "Medium",
                        0.7: "High"}.get(selected_blending_factor)

    if selected_keep_frames == True:
        keep_frames_to_save = "ON"
    else:
        keep_frames_to_save = "OFF"

    if selected_AI_multithreading == 1:
        AI_multithreading_to_save = "OFF"
    else:
        AI_multithreading_to_save = f"{selected_AI_multithreading} threads"

    user_preference = {
        "default_AI_model":             AI_model_to_save,
        "default_AI_multithreading":    AI_multithreading_to_save,
        "default_gpu":                  gpu_to_save,
        "default_keep_frames":          keep_frames_to_save,
        "default_image_extension":      image_extension_to_save,
        "default_video_extension":      video_extension_to_save,
        "default_video_codec":          video_codec_to_save,
        "default_blending":             blending_to_save,
        "default_output_path":          selected_output_path.get(),
        "default_input_resize_factor":  str(selected_input_resize_factor.get()),
        "default_output_resize_factor": str(selected_output_resize_factor.get()),
        "default_VRAM_limiter":         str(selected_VRAM_limiter.get()),
    }
    user_preference_json = json_dumps(user_preference)
    with open(USER_PREFERENCE_PATH, "w") as preference_file:
        preference_file.write(user_preference_json)

    stop_upscale_process()


class App():
    def __init__(self, window):
        self.toplevel_window = None
        window.protocol("WM_DELETE_WINDOW", on_app_close)

        window.title('')
        # Get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        # Set to 80% of the screen by default, centered
        default_width = int(screen_width * 0.8)
        default_height = int(screen_height * 0.8)
        x_position = (screen_width - default_width) // 2
        y_position = (screen_height - default_height) // 2
        window.geometry(
            f"{default_width}x{default_height}+{x_position}+{y_position}")
        window.resizable(True, True)
        window.iconbitmap(find_by_relative_path(
            "Assets" + os_separator + "logo.ico"))

        place_loadFile_section()

        place_app_name()
        place_output_path_textbox()

        place_AI_menu()
        place_AI_multithreading_menu()

        # Show appropriate menu based on default AI model
        if default_AI_model in RIFE_models_list:
            place_frame_generation_menu()
        else:
            place_AI_blending_menu()

        place_input_output_resolution_textboxs()

        place_gpu_gpuVRAM_menus()
        place_video_codec_keep_frames_menus()

        place_image_video_output_menus()

        place_message_label()
        place_upscale_button()


# Splash Screen class for application startup
class SplashScreen(CTkToplevel):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("")
        self.overrideredirect(True)  # Remove window decorations
        self.attributes('-topmost', True)

        # Calculate window position for center of screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        default_width = int(screen_width * 0.4)
        default_height = int(screen_height * 0.3)
        self.geometry(f"{default_width}x{default_height}")

        # Set default window size
        window_width = 500
        window_height = 300

        # Try to load banner image
        banner_path = find_by_relative_path(f"Assets{os_separator}banner.png")
        try:
            self.banner_image = CTkImage(
                pillow_image_open(banner_path),
                size=(450, 200)  # Adjust size as needed
            )
            has_banner = True
        except Exception as e:
            print(f"[SPLASH] Could not load splash banner: {e}")
            has_banner = False
            window_height = 200  # Smaller height if no banner

        # Center window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Configure appearance to match app
        self.configure(fg_color="#212325")  # background_color

        # Create banner or title
        if has_banner:
            self.banner_label = CTkLabel(
                self,
                image=self.banner_image,
                text=""
            )
            self.banner_label.pack(pady=(30, 15))
        else:
            # Fallback to text title if image not found
            title_label = CTkLabel(
                self,
                text="Warlock Studio",
                font=CTkFont(family="Segoe UI", size=28, weight="bold"),
                text_color="#ECD125"  # app_name_color
            )
            title_label.pack(pady=(50, 20))

        # Create status frame with progress messages
        status_frame = CTkFrame(
            self,
            fg_color="#343638",  # widget_background_color
            corner_radius=10
        )
        status_frame.pack(pady=10, padx=20, fill="x")

        self.status_label = CTkLabel(
            status_frame,
            text="Loading AI-ONNX models...",
            font=CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="white"  # text_color
        )
        self.status_label.pack(pady=10, padx=10)

        # Define enough messages to fill 15 seconds (~1.5s por mensaje)
        self.messages = [
            "Preparing environment...",
            "Loading AI-ONNX models...",
            "Initializing FFmpeg...",
            "Almost ready..."
        ]

        # Start loading animation
        self._loading_step = 0
        self.update_loading_text()

        # Splash duration: 10 seconds
        self.after(10000, self.start_fade_out)

    def update_loading_text(self):
        """Update the loading message every 1.5 seconds"""
        if self._loading_step < len(self.messages):
            self.status_label.configure(text=self.messages[self._loading_step])
            self._loading_step += 1
            self.after(1500, self.update_loading_text)

    def start_fade_out(self):
        """Start the fade out animation"""
        self._fade_step = 1.0
        self.fade_out()

    def fade_out(self):
        """Smoothly fade out the splash screen"""
        if self._fade_step > 0:
            # Use cosine for smooth fade
            opacity = cos((1.0 - self._fade_step) * pi/2)
            self.attributes('-alpha', opacity)
            self._fade_step -= 0.05
            self.after(40, self.fade_out)
        else:
            self.destroy()


if __name__ == "__main__":
    multiprocessing_freeze_support()
    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    process_status_q = multiprocessing_Queue(maxsize=1)

    # Create main window but keep it hidden initially
    window = CTk()
    window.withdraw()  # Hide main window temporarily

    # Create and show splash screen
    splash = SplashScreen()

    # Schedule showing the main window after splash finishes
    window.after(11000, window.deiconify)  # 10s + fade time

    info_message = StringVar()
    selected_output_path = StringVar()
    selected_input_resize_factor = StringVar()
    selected_output_resize_factor = StringVar()
    selected_VRAM_limiter = StringVar()

    global selected_file_list
    global selected_AI_model
    global selected_gpu
    global selected_keep_frames
    global selected_AI_multithreading
    global selected_image_extension
    global selected_video_extension
    global selected_video_codec
    global selected_blending_factor
    global selected_frame_generation_option
    global tiles_resolution
    global input_resize_factor

    selected_file_list = []

    selected_AI_model = default_AI_model
    selected_gpu = default_gpu
    selected_image_extension = default_image_extension
    selected_video_extension = default_video_extension
    selected_video_codec = default_video_codec

    if default_AI_multithreading == "OFF":
        selected_AI_multithreading = 1
    else:
        selected_AI_multithreading = int(default_AI_multithreading.split()[0])

    if default_keep_frames == "ON":
        selected_keep_frames = True
    else:
        selected_keep_frames = False

    selected_blending_factor = {"OFF": 0, "Low": 0.3,
                                "Medium": 0.5, "High": 0.7}.get(default_blending)

    selected_frame_generation_option = "OFF"  # Initialize frame generation option

    # Initialize global variables that are used in video processing
    global stop_thread_flag
    global global_processing_times_list
    global global_upscaled_frames_paths
    global global_can_i_update_status
    global output_resize_factor
    global tiles_resolution

    stop_thread_flag = Event()
    global_processing_times_list = []
    global_upscaled_frames_paths = []
    global_can_i_update_status = False
    output_resize_factor = 1.0
    tiles_resolution = 800  # Default value

    selected_input_resize_factor.set(default_input_resize_factor)
    selected_output_resize_factor.set(default_output_resize_factor)
    selected_VRAM_limiter.set(default_VRAM_limiter)
    selected_output_path.set(default_output_path)

    info_message.set("Ready for the wonderful show!")
    selected_input_resize_factor.trace_add('write', update_file_widget)
    selected_output_resize_factor.trace_add('write', update_file_widget)

    font = "Segoe UI"
    bold8 = CTkFont(family=font, size=8, weight="bold")
    bold9 = CTkFont(family=font, size=9, weight="bold")
    bold10 = CTkFont(family=font, size=10, weight="bold")
    bold11 = CTkFont(family=font, size=11, weight="bold")
    bold12 = CTkFont(family=font, size=12, weight="bold")
    bold13 = CTkFont(family=font, size=13, weight="bold")
    bold14 = CTkFont(family=font, size=14, weight="bold")
    bold16 = CTkFont(family=font, size=16, weight="bold")
    bold17 = CTkFont(family=font, size=17, weight="bold")
    bold18 = CTkFont(family=font, size=18, weight="bold")
    bold19 = CTkFont(family=font, size=19, weight="bold")
    bold20 = CTkFont(family=font, size=20, weight="bold")
    bold21 = CTkFont(family=font, size=21, weight="bold")
    bold22 = CTkFont(family=font, size=22, weight="bold")
    bold23 = CTkFont(family=font, size=23, weight="bold")
    bold24 = CTkFont(family=font, size=24, weight="bold")

    stop_icon = CTkImage(pillow_image_open(find_by_relative_path(
        f"Assets{os_separator}stop_icon.png")),      size=(15, 15))
    upscale_icon = CTkImage(pillow_image_open(find_by_relative_path(
        f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    clear_icon = CTkImage(pillow_image_open(find_by_relative_path(
        f"Assets{os_separator}clear_icon.png")),     size=(15, 15))
    info_icon = CTkImage(pillow_image_open(find_by_relative_path(
        f"Assets{os_separator}info_icon.png")),      size=(18, 18))

    app = App(window)
    window.update()
    window.mainloop()

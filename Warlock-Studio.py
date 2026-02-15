# Standard library imports
import atexit
import gc
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime
from functools import cache
from json import JSONDecodeError
from json import dumps as json_dumps
from json import load as json_load
from math import cos, pi
from multiprocessing import Process
from multiprocessing import Queue as multiprocessing_Queue
from multiprocessing import freeze_support
from multiprocessing.pool import ThreadPool
from os import cpu_count as os_cpu_count
from os import devnull as os_devnull
from os import listdir as os_listdir
from os import makedirs as os_makedirs
from os import path as os_path
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
from shutil import copy2
from shutil import move as shutil_move
from shutil import rmtree as remove_directory
from subprocess import CalledProcessError
from subprocess import run as subprocess_run
from threading import Event, Lock, Thread
from time import sleep
from timeit import default_timer as timer
from tkinter import DISABLED, StringVar, messagebox
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from webbrowser import open as open_browser

import customtkinter as ctk
import cv2
import numpy as np
# ONNX Runtime imports
import onnxruntime
# GUI imports (CustomTkinter & TkinterDnD)
from customtkinter import (CTk, CTkButton, CTkEntry, CTkFont, CTkFrame,
                           CTkImage, CTkLabel, CTkOptionMenu, CTkProgressBar,
                           CTkScrollableFrame, CTkToplevel, filedialog,
                           set_appearance_mode, set_default_color_theme)
# OpenCV imports
from cv2 import (CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT,
                 CAP_PROP_FRAME_WIDTH, COLOR_BGR2RGB, COLOR_BGR2RGBA,
                 COLOR_BGRA2BGR, COLOR_GRAY2RGB, COLOR_RGB2GRAY,
                 IMREAD_UNCHANGED, INTER_AREA, INTER_CUBIC)
from cv2 import VideoCapture as opencv_VideoCapture
from cv2 import addWeighted as opencv_addWeighted
from cv2 import cvtColor as opencv_cvtColor
from cv2 import imdecode as opencv_imdecode
from cv2 import imencode as opencv_imencode
from cv2 import imread as image_read
from cv2 import resize as opencv_resize
# MoviePy imports
from moviepy.video.io import ImageSequenceClip
# Third-party library imports
from natsort import natsorted
# NumPy imports
from numpy import ascontiguousarray as numpy_ascontiguousarray
from numpy import clip as numpy_clip
from numpy import concatenate as numpy_concatenate
from numpy import expand_dims as numpy_expand_dims
from numpy import float16, float32
from numpy import frombuffer as numpy_frombuffer
from numpy import full as numpy_full
from numpy import max as numpy_max
from numpy import mean as numpy_mean
from numpy import min as numpy_min
from numpy import ndarray as numpy_ndarray
from numpy import repeat as numpy_repeat
from numpy import squeeze as numpy_squeeze
from numpy import stack as numpy_stack
from numpy import transpose as numpy_transpose
from numpy import uint8
from numpy import zeros as numpy_zeros
from onnxruntime import InferenceSession
# Necesitarás PIL para cargar el icono de limpieza que pide el constructor
from PIL import Image
from PIL.Image import fromarray as pillow_image_fromarray
from PIL.Image import open as pillow_image_open
from tkinterdnd2 import DND_ALL, TkinterDnD

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# INITIALIZE CONSOLE REDIRECTION EARLY
# -----------------------------------------------------------------------------
from console import IntegratedConsole, console
# Local imports
from drag_drop import DnDCTk, enable_drag_and_drop
# Importa la clase de tu archivo (asumiendo que se llama file_queue_manager.py)
from file_queue_manager import FileQueueManager
from processing_chain import ChainManager, ProcessingStep
from splash_screen import SplashScreen
from warlock_preferences import ConfigManager, PreferencesButton

# Redirigir inmediatamente para capturar logs de importación
console.setup_redirection()

# Suppress specific warnings to keep console clean
warnings.filterwarnings("ignore", category=UserWarning)

# Variable global para mantener la instancia (o la lista)
chain_window = None
current_chain_steps = []  # Aquí se guardarán los pasos si el usuario usa la cadena

# Handle PyInstaller or Normal Execution Path


def find_by_relative_path(relative_path: str) -> str:
    """
    Resuelve rutas absolutas para recursos, funcionando tanto en desarrollo
    como cuando el script está empaquetado con PyInstaller (--onefile).
    """
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(
        os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)


app_name = "Warlock-Studio"
version = "6.0"

# Supported File Extensions
supported_image_extensions = [".jpg", ".jpeg",
                              ".png", ".bmp", ".tiff", ".tif", ".webp"]
supported_video_extensions = [".mp4", ".avi",
                              ".mkv", ".mov", ".wmv", ".flv", ".webm"]
supported_file_extensions = supported_image_extensions + supported_video_extensions

# -----------------------------------------------------------------------------
# THEME & COLORS
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# THEME & COLORS
# -----------------------------------------------------------------------------

# Fondo: Negro casi puro, igual que el fondo del banner para máximo contraste
background_color = "#000000"
# Nombre de la app: Amarillo intenso para alto contraste
app_name_color = "#FFD700"
# Paneles: Gris oscuro neutro, permite que el rojo y dorado resalten sin competir
widget_background_color = "#1A1A1A"
# Texto principal: Blanco puro para legibilidad máxima
text_color = "#F5F5F5"
# Texto secundario: Dorado pálido/desaturado, para no cansar la vista pero mantener la identidad
secondary_text_color = "#9E9E9E"
# Acento: Amarillo/ámbar vibrante (alto contraste)
accent_color = "#FFC107"
# Hover de botones: El rojo vibrante del relleno del texto "WARLOCK"
button_hover_color = "#D32F2F"
# Bordes: Un dorado oscuro muy sutil, imitando el borde del logo sin ser chillón
border_color = "#2D2D2D"
# Botones info/secundarios: El rojo sangre oscuro del fondo del círculo del logo
info_button_color = "#7F1500"
# Advertencias: Naranja dorado, sacado del sombreado del sombrero
warning_color = "#FF6F00"
# Éxito: Verde brillante, necesario para contraste funcional
success_color = "#00C853"
# Error: Rojo carmesí intenso, similar al borde de las letras "WARLOCK"
error_color = "#B71C1C"
# Resaltado: Amarillo luz, como el centro de los destellos (estrellas)
highlight_color = "#FFF59D"
# Scrollbars: Rojo vino oscuro translúcido, para mantener la temática sin distraer
scrollbar_color = "#000000"

# -----------------------------------------------------------------------------
# AI MODEL LISTS & CONFIGURATION
# -----------------------------------------------------------------------------

VRAM_model_usage = {
    'RealESR_Gx4':     2.2,
    'RealESR_Animex4': 2.2,
    'RealESRNetx4':    2.2,
    'BSRGANx4':        0.6,
    'BSRGANx2':        0.7,
    'RealESRGANx4':    0.6,
    'IRCNN_Mx1':       4,
    'IRCNN_Lx1':       4,
    'GFPGAN':          1.8,
}

MENU_LIST_SEPARATOR = ["•••"]
SRVGGNetCompact_models_list = ["RealESR_Gx4", "RealESR_Animex4"]
BSRGAN_models_list = ["BSRGANx4", "BSRGANx2", "RealESRGANx4", "RealESRNetx4"]
IRCNN_models_list = ["IRCNN_Mx1", "IRCNN_Lx1"]
Face_restoration_models_list = ["GFPGAN"]
RIFE_models_list = ["RIFE", "RIFE_Lite"]

AI_models_list = (SRVGGNetCompact_models_list + MENU_LIST_SEPARATOR + BSRGAN_models_list +
                  MENU_LIST_SEPARATOR + IRCNN_models_list + MENU_LIST_SEPARATOR + Face_restoration_models_list +
                  MENU_LIST_SEPARATOR + RIFE_models_list)

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

# -----------------------------------------------------------------------------
# PATHS & USER PREFERENCES
# -----------------------------------------------------------------------------

OUTPUT_PATH_CODED = "Same path as input files"
DOCUMENT_PATH = os_path_join(os_path_expanduser('~'), 'Documents')
USER_PREFERENCE_PATH = find_by_relative_path(
    f"{DOCUMENT_PATH}{os_separator}{app_name}_{version}_UserPreference.json")

# --- INTEGRACIÓN DE PREFERENCIAS: RUTAS PERSONALIZADAS ---
_app_config = ConfigManager.load_config()
CORNER_RADIUS = _app_config.get("corner_radius", 10)

# Lógica FFmpeg
_custom_ffmpeg = _app_config.get("custom_ffmpeg_path", "")
if _custom_ffmpeg and os_path_exists(_custom_ffmpeg):
    FFMPEG_EXE_PATH = _custom_ffmpeg
else:
    FFMPEG_EXE_PATH = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")

# Lógica ExifTool
_custom_exiftool = _app_config.get("custom_exiftool_path", "")
if _custom_exiftool and os_path_exists(_custom_exiftool):
    EXIFTOOL_EXE_PATH = _custom_exiftool
else:
    EXIFTOOL_EXE_PATH = find_by_relative_path(
        f"Assets{os_separator}exiftool.exe")
# ---------------------------------------------------------

ECTRACTION_FRAMES_FOR_CPU = 30
MULTIPLE_FRAMES_TO_SAVE = 8

COMPLETED_STATUS = "Completed"
ERROR_STATUS = "Error"
STOP_STATUS = "Stop"

# Check External Tools
if os_path_exists(FFMPEG_EXE_PATH):
    print(f"[{app_name}] ffmpeg.exe found")
else:
    print(f"[{app_name}] WARNING: ffmpeg.exe not found. Video functionality will be limited.")

# Load User Preferences (with Error Handling)
# Default Values
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

if os_path_exists(USER_PREFERENCE_PATH):
    print(f"[{app_name}] Preference file exists")
    try:
        with open(USER_PREFERENCE_PATH, "r") as json_file:
            json_data = json_load(json_file)
            # Safe .get() calls to prevent KeyErrors if fields are missing in old config versions
            default_AI_model = json_data.get(
                "default_AI_model", default_AI_model)
            default_AI_multithreading = json_data.get(
                "default_AI_multithreading", default_AI_multithreading)
            default_gpu = json_data.get("default_gpu", default_gpu)
            default_keep_frames = json_data.get(
                "default_keep_frames", default_keep_frames)
            default_image_extension = json_data.get(
                "default_image_extension", default_image_extension)
            default_video_extension = json_data.get(
                "default_video_extension", default_video_extension)
            default_video_codec = json_data.get(
                "default_video_codec", default_video_codec)
            default_blending = json_data.get(
                "default_blending", default_blending)
            default_output_path = json_data.get(
                "default_output_path", default_output_path)
            default_input_resize_factor = json_data.get(
                "default_input_resize_factor", default_input_resize_factor)
            default_output_resize_factor = json_data.get(
                "default_output_resize_factor", default_output_resize_factor)
            default_VRAM_limiter = json_data.get(
                "default_VRAM_limiter", default_VRAM_limiter)
    except (JSONDecodeError, Exception) as e:
        print(f"[{app_name}] Error reading preference file ({e}). Using defaults.")
else:
    print(f"[{app_name}] Preference file does not exist, using default coded value")


# -----------------------------------------------------------------------------
# GUI LAYOUT CONSTANTS
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# ONNX SESSION HELPER
# -----------------------------------------------------------------------------

def create_onnx_session(model_path: str, selected_gpu: str) -> InferenceSession:
    """
    Creates an ONNX inference session respecting User Preferences for backend execution.
    """
    if not os_path_exists(model_path):
        raise FileNotFoundError(f"AI model file not found: {model_path}")

    # Cargar preferencias
    config = ConfigManager.load_config()
    provider_pref = config.get("onnx_provider_preference", "Auto")

    # Mapear selección de GUI a Device ID
    device_id_map = {'GPU 1': 0, 'GPU 2': 1, 'GPU 3': 2, 'GPU 4': 3}
    target_device_id = device_id_map.get(selected_gpu, 0)

    # Definir opciones de proveedores
    cuda_opts = {'device_id': target_device_id}
    dml_opts = {'device_id': target_device_id}

    # Construir lista de prioridad basada en preferencias
    providers_to_try = []

    if provider_pref == "CUDA":
        providers_to_try.append(('CUDAExecutionProvider', cuda_opts))
    elif provider_pref == "DirectML":
        providers_to_try.append(('DmlExecutionProvider', dml_opts))
    elif provider_pref == "CPU":
        providers_to_try.append(('CPUExecutionProvider', None))
    elif provider_pref == "OpenVINO":
        providers_to_try.append(('OpenVINOExecutionProvider', None))
    else:  # AUTO
        providers_to_try.append(('CUDAExecutionProvider', cuda_opts))
        providers_to_try.append(('DmlExecutionProvider', dml_opts))
        providers_to_try.append(('CPUExecutionProvider', None))

    # Asegurar que siempre hay fallbacks si falla la preferencia principal
    if ('CPUExecutionProvider', None) not in providers_to_try:
        providers_to_try.append(('CPUExecutionProvider', None))

    available_providers = onnxruntime.get_available_providers()

    for provider, options in providers_to_try:
        if provider in available_providers:
            try:
                session_options = [options] if options else None
                session = InferenceSession(
                    path_or_bytes=model_path,
                    providers=[provider],
                    provider_options=session_options
                )
                print(
                    f"[AI] Loaded model with provider: {provider} (Pref: {provider_pref})")
                return session
            except Exception as e:
                print(f"[AI WARNING] Failed to load {provider}: {e}")
                continue

    raise RuntimeError("Critical: Failed to load AI model with any provider.")
    """
    Creates an ONNX inference session by selecting the best available provider.
    Fixes: Correct type for device_id (int) and handles 'Auto' properly.
    """
    if not os_path_exists(model_path):
        raise FileNotFoundError(f"AI model file not found: {model_path}")

    # Map the GUI selection to the numerical device_id (INTEGER)
    # 'Auto' defaults to 0, but we handle it differently in logic
    device_id_map = {'GPU 1': 0, 'GPU 2': 1, 'GPU 3': 2, 'GPU 4': 3}

    target_device_id = device_id_map.get(selected_gpu, 0)
    is_auto = selected_gpu == "Auto"

    # Providers priority
    providers_to_try = []

    # 1. CUDA (NVIDIA)
    cuda_options = {'device_id': target_device_id}
    providers_to_try.append(('CUDAExecutionProvider', cuda_options))

    # 2. DirectML (AMD/Intel/Windows)
    # DirectML expects device_id as string in some versions, int in others.
    # We use int standard here, usually works with modern ort-dml.
    dml_options = {'device_id': target_device_id}
    providers_to_try.append(('DmlExecutionProvider', dml_options))

    # 3. CPU (Fallback)
    providers_to_try.append(('CPUExecutionProvider', None))

    available_providers = onnxruntime.get_available_providers()

    for provider, options in providers_to_try:
        if provider in available_providers:
            try:
                # Si es Auto y estamos en CUDA, intentamos sin forzar device_id
                # a menos que sea explícitamente necesario, pero usualmente ID 0 es seguro.
                # La corrección clave aquí es pasar options como diccionario, no lista de diccionario.

                session_options = [options] if options else None

                session = InferenceSession(
                    path_or_bytes=model_path,
                    providers=[provider],
                    provider_options=session_options
                )
                print(
                    f"[AI] Loaded model with provider: {provider} (Device ID: {target_device_id})")
                return session
            except Exception as e:
                print(f"[AI WARNING] Failed to load {provider}: {e}")
                continue

    # Final fallback attempt without options (let ONNX decide)
    try:
        return InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        raise RuntimeError(f"Critical: Failed to load AI model. Error: {e}")

# Enhanced Model Utilization and Error Handling


class AI_upscale:
    # -------------------------------------------------------------------------
    # CLASS INIT
    # -------------------------------------------------------------------------
    def __init__(
            self,
            AI_model_name: str,
            directml_gpu: str,
            input_resize_factor: float,
            output_resize_factor: float,
            max_resolution: int
    ):
        # Parámetros recibidos
        self.AI_model_name = AI_model_name
        self.directml_gpu = directml_gpu
        self.input_resize_factor = input_resize_factor
        self.output_resize_factor = output_resize_factor
        self.max_resolution = max_resolution

        # --- OPTIMIZACIÓN: Padding y Batch Size ---
        # Padding: Píxeles extra alrededor del tile para evitar cortes visibles (seamless).
        # 32 es un buen equilibrio para la mayoría de modelos SRGAN/ESRGAN.
        self.tile_padding = 32

        # Batch Size: Cantidad de tiles procesados simultáneamente.
        # Aumentar esto acelera el proceso pero consume más VRAM.
        # 4 es un valor conservador y rápido para la mayoría de GPUs modernas.
        self.batch_size = 4

        # --- CORRECCIÓN DE RUTAS (FP16 / FP32) ---
        # Definimos las posibles rutas del modelo
        path_fp16 = find_by_relative_path(
            f"AI-onnx{os_separator}{self.AI_model_name}_fp16.onnx")
        path_fp32 = find_by_relative_path(
            f"AI-onnx{os_separator}{self.AI_model_name}_fp32.onnx")
        path_clean = find_by_relative_path(
            f"AI-onnx{os_separator}{self.AI_model_name}.onnx")

        # Verificamos cuál existe y asignamos
        if os_path_exists(path_fp16):
            self.AI_model_path = path_fp16
        elif os_path_exists(path_fp32):
            self.AI_model_path = path_fp32
        elif os_path_exists(path_clean):
            self.AI_model_path = path_clean
        else:
            # Si no encuentra ninguno, dejamos el por defecto (fp16)
            self.AI_model_path = path_fp16

        self.upscale_factor = self._get_upscale_factor()

        # La sesión se carga bajo demanda
        self.inferenceSession = None

    def _get_upscale_factor(self) -> int:
        """Determina el factor de escala basado en el nombre del modelo."""
        if "x1" in self.AI_model_name:
            return 1
        elif "x2" in self.AI_model_name:
            return 2
        elif "x4" in self.AI_model_name:
            return 4
        return 1

    def _load_inferenceSession(self) -> None:
        """Carga la sesión de inferencia utilizando la función centralizada."""
        if self.inferenceSession is not None:
            return
        try:
            self.inferenceSession = create_onnx_session(
                self.AI_model_path, self.directml_gpu)
        except Exception as e:
            error_msg = f"Failed to load AI model {os_path_basename(self.AI_model_path)}: {str(e)}"
            logging.error(f"[AI ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    # -------------------------------------------------------------------------
    # IMAGE UTILS
    # -------------------------------------------------------------------------
    def get_image_mode(self, image: numpy_ndarray) -> str:
        if image is None:
            raise ValueError("Image is None")
        if image.ndim == 2:
            return "Grayscale"
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 3:
                return "RGB"
            elif channels == 4:
                return "RGBA"
            elif channels == 1:
                return "Grayscale"
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def get_image_resolution(self, image: numpy_ndarray) -> tuple:
        return image.shape[0], image.shape[1]  # Height, Width

    def calculate_target_resolution(self, image: numpy_ndarray) -> tuple:
        h, w = self.get_image_resolution(image)
        return h * self.upscale_factor, w * self.upscale_factor

    def _ensure_even_dimensions(self, dim: int) -> int:
        return dim if dim % 2 == 0 else dim + 1

    def resize_with_input_factor(self, image: numpy_ndarray) -> numpy_ndarray:
        old_h, old_w = self.get_image_resolution(image)
        new_w = int(old_w * self.input_resize_factor)
        new_h = int(old_h * self.input_resize_factor)

        new_w = max(2, self._ensure_even_dimensions(new_w))
        new_h = max(2, self._ensure_even_dimensions(new_h))

        if self.input_resize_factor == 1.0 and (new_w == old_w and new_h == old_h):
            return image

        # Optimización: Lanczos es mejor para reducir (downscale), Cubic para ampliar
        interpolation = INTER_CUBIC if self.input_resize_factor > 1 else cv2.INTER_LANCZOS4
        return opencv_resize(image, (new_w, new_h), interpolation=interpolation)

    def resize_with_output_factor(self, image: numpy_ndarray) -> numpy_ndarray:
        old_h, old_w = self.get_image_resolution(image)
        new_w = int(old_w * self.output_resize_factor)
        new_h = int(old_h * self.output_resize_factor)

        new_w = max(2, self._ensure_even_dimensions(new_w))
        new_h = max(2, self._ensure_even_dimensions(new_h))

        if self.output_resize_factor == 1.0 and (new_w == old_w and new_h == old_h):
            return image

        interpolation = INTER_CUBIC if self.output_resize_factor > 1 else cv2.INTER_LANCZOS4
        return opencv_resize(image, (new_w, new_h), interpolation=interpolation)

    def calculate_multiframes_supported_by_gpu(self, video_frame_path: str) -> int:
        try:
            frame = image_read(video_frame_path)
            # Usar una estimación rápida si la imagen es gigante
            if frame is None:
                return 1

            # Simular el resize de entrada
            h, w = frame.shape[:2]
            input_h = int(h * self.input_resize_factor)
            input_w = int(w * self.input_resize_factor)

            image_pixels = input_h * input_w
            max_pixels = self.max_resolution * self.max_resolution

            if image_pixels == 0:
                return 1

            # Cálculo conservador
            frames = max_pixels // image_pixels
            return max(1, min(frames, 16))
        except:
            return 1

    # -------------------------------------------------------------------------
    # CORE PROCESSING (Optimized Batch & Tiling)
    # -------------------------------------------------------------------------
    def normalize_image(self, image: numpy_ndarray) -> tuple:
        # Optimización: Vectorización directa
        if image.dtype == uint8:
            return (image.astype(float32) / 255.0), 255.0
        elif numpy_max(image) > 1.0:
            return (image.astype(float32) / 255.0), 255.0
        else:
            return image.astype(float32), 1.0

    def de_normalize_image(self, image: numpy_ndarray, range_val: float) -> numpy_ndarray:
        image = image * range_val
        return numpy_clip(image, 0, 255).astype(uint8)

    def preprocess_image_batch(self, images_list: list) -> numpy_ndarray:
        """Convierte una lista de imágenes HWC a un batch NCHW."""
        # Stack convierte lista de arrays (H,W,C) en (N, H, W, C)
        batch = numpy_stack(images_list, axis=0)
        # Transponer a (N, C, H, W)
        batch = numpy_transpose(batch, (0, 3, 1, 2))
        return numpy_ascontiguousarray(batch)

    def onnx_inference_batch(self, batch_input: numpy_ndarray) -> numpy_ndarray:
        if self.inferenceSession is None:
            self._load_inferenceSession()

        input_name = self.inferenceSession.get_inputs()[0].name
        # Ejecutar inferencia
        try:
            results = self.inferenceSession.run(
                None, {input_name: batch_input})[0]
        except Exception as e:
            # Fallback a CPU si falla GPU por memoria en lote grande
            logging.warning(
                f"Batch inference failed: {e}. Retrying individually might be needed.")
            raise e

        return results

    def _process_tile_batch(self, batch_images: list, range_val: float) -> list:
        """Helper para procesar un lote de tiles RGB."""
        if not batch_images:
            return []

        # 1. Normalizar (asumiendo que ya vienen en float32 0-1 o procesar aqui)
        # Para velocidad, asumiremos que batch_images ya son recortes del 'image_norm' (float32)

        # 2. Preprocesar (HWC -> NCHW)
        batch_nchw = self.preprocess_image_batch(batch_images)

        # 3. Inferencia
        output_nchw = self.onnx_inference_batch(batch_nchw)

        # 4. Post-proceso (NCHW -> HWC)
        # Transponer de vuelta a (N, H, W, C)
        output_nhwc = numpy_transpose(output_nchw, (0, 2, 3, 1))

        # 5. Denormalizar y convertir a lista
        processed_batch = []
        for i in range(output_nhwc.shape[0]):
            img_out = self.de_normalize_image(output_nhwc[i], range_val)
            processed_batch.append(img_out)

        return processed_batch

    # -------------------------------------------------------------------------
    # MAIN UPSCALE LOGIC
    # -------------------------------------------------------------------------
    def AI_upscale(self, image: numpy_ndarray) -> numpy_ndarray:
        try:
            if image is None or image.size == 0:
                raise ValueError("Input image is empty")

            # Manejo de memoria y formato
            image = numpy_ascontiguousarray(image)
            mode = self.get_image_mode(image)

            # --- SEPARACIÓN DE CANAL ALPHA ---
            has_alpha = False
            channel_alpha = None
            rgb_image = image

            if mode == "RGBA":
                has_alpha = True
                channel_alpha = image[:, :, 3]
                rgb_image = image[:, :, :3]  # Extraer RGB
            elif mode == "Grayscale":
                # Convertir a RGB para el modelo
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # --- PREPARACIÓN RGB ---
            # Normalizar imagen completa
            image_norm, range_val = self.normalize_image(rgb_image)

            h, w = image_norm.shape[:2]

            # Comprobar Tiling
            # Usamos max_resolution - padding * 2 para asegurar que el contenido útil cabe
            if (h * w) > (self.max_resolution ** 2):
                upscaled_rgb = self._upscale_with_tiling_optimized(
                    image_norm, range_val)
            else:
                # Proceso directo (Single Image)
                upscaled_rgb = self._process_tile_batch(
                    [image_norm], range_val)[0]

            # --- RECONSTRUCCIÓN ALPHA Y SALIDA ---
            if has_alpha:
                # Escalar Alpha con Bicubic/Lanczos (muy rápido y buena calidad para máscaras)
                target_h, target_w = upscaled_rgb.shape[:2]
                upscaled_alpha = cv2.resize(
                    channel_alpha, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                # Unir
                if upscaled_alpha.ndim == 2:
                    upscaled_alpha = numpy_expand_dims(upscaled_alpha, axis=2)

                final_image = numpy_concatenate(
                    (upscaled_rgb, upscaled_alpha), axis=2)

            elif mode == "Grayscale":
                final_image = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2GRAY)
            else:
                final_image = upscaled_rgb

            return final_image

        except Exception as e:
            logging.error(f"AI Upscale Error: {str(e)}")
            raise RuntimeError(f"Upscaling failed: {str(e)}")

    def _upscale_with_tiling_optimized(self, image_norm: numpy_ndarray, range_val: float) -> numpy_ndarray:
        """
        Sistema de Tiling Avanzado con Overlap (Superposición) y Batch Processing.
        Elimina las costuras visibles (seams) y acelera con lotes.
        """
        h, w = image_norm.shape[:2]
        tile_size = self.max_resolution
        padding = self.tile_padding
        scale = self.upscale_factor

        # Coordenadas de los tiles (sin padding todavía)
        # Se generan pasos de (tile_size - 2*padding) para asegurar cobertura limpia
        step = tile_size - (2 * padding)
        if step <= 0:
            step = tile_size // 2  # Fallback si el padding es enorme

        y_points = list(range(0, h, step))
        x_points = list(range(0, w, step))

        # Calcular tamaño final
        target_h, target_w = h * scale, w * scale

        # Lienzo final pre-alocado (Más rápido que append)
        final_image = numpy_zeros((target_h, target_w, 3), dtype=uint8)

        batch_tiles = []
        # (y_start_out, y_end_out, x_start_out, x_end_out, crop_y1, crop_y2, crop_x1, crop_x2)
        batch_coords = []

        print(
            f"[AI] Advanced Tiling: Processing grid {len(y_points)}x{len(x_points)} with Batch Size {self.batch_size}")

        for y in y_points:
            for x in x_points:
                # 1. Definir coordenadas del tile de entrada con padding
                # Intentamos tomar padding extra, pero clampamos a los bordes de la imagen
                y_start_in = max(0, y - padding)
                y_end_in = min(h, y + step + padding)
                x_start_in = max(0, x - padding)
                x_end_in = min(w, x + step + padding)

                # Extraer tile
                tile = image_norm[y_start_in:y_end_in, x_start_in:x_end_in]

                # Guardar tile para batch
                batch_tiles.append(tile)

                # 2. Calcular dónde va este tile en la imagen de SALIDA
                # La zona "valid" es la que no es padding (excepto en los bordes de la imagen original)

                # Offsets relativos dentro del tile extraído (para recortar el padding después del upscale)
                res_y_start = (y - y_start_in) * scale
                res_x_start = (x - x_start_in) * scale

                # Dimensiones de la zona válida a pegar en el canvas final
                # El ancho/alto "útil" es min(step, lo que quede de imagen) * scale
                valid_h = min(step, h - y) * scale
                valid_w = min(step, w - x) * scale

                res_y_end = res_y_start + valid_h
                res_x_end = res_x_start + valid_w

                # Coordenadas absolutas en el canvas final
                abs_y = y * scale
                abs_x = x * scale

                # Guardamos las coordenadas de recorte y pegado
                batch_coords.append({
                    "crop": (int(res_y_start), int(res_y_end), int(res_x_start), int(res_x_end)),
                    "place": (int(abs_y), int(abs_y + valid_h), int(abs_x), int(abs_x + valid_w))
                })

                # --- PROCESAR BATCH SI ESTÁ LLENO ---
                if len(batch_tiles) >= self.batch_size:
                    self._process_and_stitch_batch(
                        batch_tiles, batch_coords, final_image, range_val)
                    batch_tiles = []
                    batch_coords = []
                    gc.collect()  # Mantener RAM a raya

        # Procesar remanentes
        if batch_tiles:
            self._process_and_stitch_batch(
                batch_tiles, batch_coords, final_image, range_val)
            gc.collect()

        return final_image

    def _process_and_stitch_batch(self, tiles, coords, final_image, range_val):
        """Procesa un batch y pega los resultados en el canvas global."""
        # Nota: Los tiles pueden tener tamaños distintos en los bordes.
        # Si los tamaños son distintos, no se puede hacer batching directo en un tensor único NCHW.
        # Verificamos si todos tienen mismo tamaño. Si no, procesamos 1 a 1 (fallback del batch).

        shapes = [t.shape for t in tiles]
        if len(set(shapes)) > 1:
            # Tamaños mixtos (pasa en bordes), procesar uno a uno
            upscaled_tiles = []
            for t in tiles:
                upscaled_tiles.extend(self._process_tile_batch([t], range_val))
        else:
            # Todos iguales, batch real
            upscaled_tiles = self._process_tile_batch(tiles, range_val)

        # Pegado (Stitching)
        for img_out, coord in zip(upscaled_tiles, coords):
            cy1, cy2, cx1, cx2 = coord["crop"]
            py1, py2, px1, px2 = coord["place"]

            # Recortar zona válida (quitando padding escalado)
            valid_content = img_out[cy1:cy2, cx1:cx2]

            # Pegar en imagen final
            # Asegurar dimensiones exactas (por redondeos)
            fh, fw = final_image[py1:py2, px1:px2].shape[:2]
            if valid_content.shape[0] != fh or valid_content.shape[1] != fw:
                valid_content = valid_content[:fh, :fw]

            final_image[py1:py2, px1:px2] = valid_content

    # -------------------------------------------------------------------------
    # ORCHESTRATION (PUBLIC API)
    # -------------------------------------------------------------------------
    def AI_orchestration(self, image: numpy_ndarray) -> numpy_ndarray:
        """Punto de entrada principal para procesar una imagen."""
        if self.inferenceSession is None:
            self._load_inferenceSession()

        # 1. Redimensionar Entrada (Input Resize)
        image = self.resize_with_input_factor(image)

        # 2. Upscaling (Core AI)
        upscaled_image = self.AI_upscale(image)

        # 3. Redimensionar Salida (Output Resize)
        final_image = self.resize_with_output_factor(upscaled_image)

        return final_image


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
        """Carga la sesión de inferencia utilizando la función centralizada."""
        try:
            return create_onnx_session(self.AI_model_path, self.directml_gpu)
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
        # Optimización: Normalizar in-place para reducir uso de memoria
        image1 = numpy_ascontiguousarray(image1, dtype=float32) / 255.0
        image2 = numpy_ascontiguousarray(image2, dtype=float32) / 255.0
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

    def AI_orchestration(self, image1: numpy_ndarray, image2: numpy_ndarray) -> List[numpy_ndarray]:
        """Generate interpolated frames between two input images."""
        generated_images = []

        # Optimización: Usar memoria contigua para las imágenes de entrada
        image1 = numpy_ascontiguousarray(image1)
        image2 = numpy_ascontiguousarray(image2)

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


# AI FACE RESTORATION for face enhancement -----------------

class AI_face_restoration:
    """
    Clase para restauración facial (GFPGAN u otros modelos ONNX de face-restoration).
    Reemplaza/actualiza la implementación anterior con:
    - Preprocesado seguro (float32 por defecto)
    - Conversión a float16 solo si la sesión ONNX realmente lo requiere
    - Manejo de alpha channel y reescalados
    - Logs diagnósticos y manejo robusto de errores
    """

    def __init__(
        self,
        AI_model_name: str,
        directml_gpu: str,
        input_resize_factor: float,
        output_resize_factor: float,
        max_resolution: int
    ):
        # Parámetros pasados
        self.AI_model_name = AI_model_name
        self.directml_gpu = directml_gpu
        self.input_resize_factor = input_resize_factor
        self.output_resize_factor = output_resize_factor
        self.max_resolution = max_resolution

        # Configuración por modelo (ajustable)
        # GFPGAN suele usar 512x512; ajustar según tu modelo real
        self.model_configs = {
            "GFPGAN": {
                "input_size": (512, 512),
                "scale_factor": 1,
                "description": "GFPGAN v1.4 for face restoration",
                "fp16": True  # indica que hay una variante fp16, pero no forzamos su uso
            }
        }

        # Rutas y estado
        self.AI_model_path = self._get_model_path()
        self.model_config = self.model_configs.get(
            AI_model_name, self.model_configs["GFPGAN"])
        self.inferenceSession = None

    # -------------------
    # CARGA Y SESIÓN ONNX
    # -------------------
    def _get_model_path(self) -> str:
        """
        Construye la ruta al archivo ONNX del modelo.
        """
        # Prioriza la versión fp16 si nombre lo sugiere, si no existe cae en fp32
        candidate_fp16 = find_by_relative_path(
            f"AI-onnx{os_separator}{self.AI_model_name}_fp16.onnx")
        candidate_fp32 = find_by_relative_path(
            f"AI-onnx{os_separator}{self.AI_model_name}_fp32.onnx")
        candidate_default = find_by_relative_path(
            f"AI-onnx{os_separator}{self.AI_model_name}.onnx")

        if os_path_exists(candidate_fp16):
            return candidate_fp16
        if os_path_exists(candidate_fp32):
            return candidate_fp32
        if os_path_exists(candidate_default):
            return candidate_default

        # Si no existe, retornamos la ruta esperada (la carga fallará y se informará)
        return candidate_default

    def _load_inferenceSession(self) -> None:
        """
        Carga la sesión ONNX usando la función centralizada create_onnx_session.
        Levanta RuntimeError si falla.
        """
        try:
            if not os_path_exists(self.AI_model_path):
                raise FileNotFoundError(
                    f"AI model not found: {self.AI_model_path}")
            self.inferenceSession = create_onnx_session(
                self.AI_model_path, self.directml_gpu)
            print(
                f"[GFPGAN] Modelo cargado: {os_path_basename(self.AI_model_path)}")
        except Exception as e:
            error_msg = f"Failed to load face restoration model {os_path_basename(self.AI_model_path)}: {str(e)}"
            print(f"[AI ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    # -------------------
    # UTILIDADES INTERNAS
    # -------------------
    def get_image_mode(self, image: numpy_ndarray) -> str:
        """
        Devuelve 'Grayscale', 'RGB' o 'RGBA' según la forma del array.
        """
        if image is None:
            raise ValueError("Image is None")
        shape = image.shape
        if len(shape) == 2:
            return "Grayscale"
        elif len(shape) == 3 and shape[2] == 3:
            return "RGB"
        elif len(shape) == 3 and shape[2] == 4:
            return "RGBA"
        else:
            raise ValueError(f"Unsupported image shape: {shape}")

    def get_image_resolution(self, image: numpy_ndarray) -> tuple:
        height = image.shape[0]
        width = image.shape[1]
        return height, width

    def resize_with_input_factor(self, image: numpy_ndarray) -> numpy_ndarray:
        """
        Redimensiona la imagen según input_resize_factor y garantiza dimensiones pares.
        """
        old_h, old_w = self.get_image_resolution(image)
        new_w = int(old_w * self.input_resize_factor)
        new_h = int(old_h * self.input_resize_factor)
        new_w = new_w if new_w % 2 == 0 else new_w + 1
        new_h = new_h if new_h % 2 == 0 else new_h + 1

        if self.input_resize_factor > 1:
            return opencv_resize(image, (new_w, new_h), interpolation=INTER_CUBIC)
        elif self.input_resize_factor < 1:
            return opencv_resize(image, (new_w, new_h), interpolation=INTER_AREA)
        else:
            return image

    def resize_with_output_factor(self, image: numpy_ndarray) -> numpy_ndarray:
        """
        Redimensiona la imagen según output_resize_factor y garantiza dimensiones pares.
        """
        old_h, old_w = self.get_image_resolution(image)
        new_w = int(old_w * self.output_resize_factor)
        new_h = int(old_h * self.output_resize_factor)
        new_w = new_w if new_w % 2 == 0 else new_w + 1
        new_h = new_h if new_h % 2 == 0 else new_h + 1

        if self.output_resize_factor > 1:
            return opencv_resize(image, (new_w, new_h), interpolation=INTER_CUBIC)
        elif self.output_resize_factor < 1:
            return opencv_resize(image, (new_w, new_h), interpolation=INTER_AREA)
        else:
            return image

    def add_alpha_channel(self, image: numpy_ndarray) -> numpy_ndarray:
        """
        Asegura que la imagen tenga canal alpha (lo añade opaco si no).
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            alpha = numpy_full(
                (image.shape[0], image.shape[1], 1), 255, dtype=uint8)
            image = numpy_concatenate((image, alpha), axis=2)
        return image

    # -------------------
    # PRE / POST PROCESS
    # -------------------
    def preprocess_face_image(self, image: numpy_ndarray) -> tuple[numpy_ndarray, bool]:
        """
        Prepara la imagen para la inferencia de restauración facial.
        Devuelve (preprocessed_image_float32, had_alpha_bool).
        - Siempre devuelve float32 por defecto.
        - El caller decidirá convertir a float16 justo antes de la inferencia si la sesión lo requiere.
        """
        # Asegurar memoria contigua
        image = numpy_ascontiguousarray(image)

        # Detectar alpha
        had_alpha = False
        if len(image.shape) == 3 and image.shape[2] == 4:
            had_alpha = True
            # Guardamos alpha pero procesaremos solo BGR
            # Convertir BGRA -> BGR para el modelo
            try:
                image = opencv_cvtColor(image, COLOR_BGRA2BGR)
            except Exception:
                # Fallback: eliminar canal alpha si cvtColor falla
                image = image[:, :, :3]

        # Asegurar que la imagen tenga 3 canales
        if len(image.shape) == 2:
            image = opencv_cvtColor(image, COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] != 3:
            # si hay más canales, recortar a 3
            image = image[:, :, :3]

        # Redimensionar a tamaño del modelo (input_size)
        target_w, target_h = self.model_config["input_size"][1], self.model_config["input_size"][0]
        try:
            image_resized = opencv_resize(
                image, (target_w, target_h), interpolation=INTER_AREA)
        except Exception as e:
            print(
                f"[GFPGAN] Warning: resize failed: {e}. Using original size.")
            image_resized = image

        # Normalizar a float32 en rango [0,1]
        preprocessed = numpy_ascontiguousarray(
            image_resized, dtype=float32) / 255.0

        # Transpose a NCHW
        preprocessed = numpy_transpose(preprocessed, (2, 0, 1))
        preprocessed = numpy_expand_dims(preprocessed, axis=0)  # batch dim

        return preprocessed, had_alpha

    def postprocess_face_image(self, output: numpy_ndarray, original_size: tuple) -> numpy_ndarray:
        """
        Postprocesa la salida del modelo:
         - squeeze batch
         - clamp [0,1]
         - transpose a HWC
         - convertir a uint8 y redimensionar a tamaño original
        """
        # Squeeze batch
        output = numpy_squeeze(output, axis=0)

        # Clamp y asegurar tipo float32
        output = numpy_clip(output, 0.0, 1.0)

        # Transpose a HWC
        output = numpy_transpose(output, (1, 2, 0))

        # Convertir a uint8
        output_uint8 = (output * 255.0).round().astype(uint8)

        # Redimensionar a tamaño original (original_size es (h, w))
        try:
            if (original_size[0], original_size[1]) != (self.model_config["input_size"][0], self.model_config["input_size"][1]):
                # opencv resize espera (width, height)
                output_uint8 = opencv_resize(
                    output_uint8, (original_size[1], original_size[0]), interpolation=INTER_CUBIC)
        except Exception as e:
            print(f"[GFPGAN] Warning: postprocess resize failed: {e}")

        return output_uint8

    # -------------------
    # LÓGICA PRINCIPAL
    # -------------------
    def face_restoration(self, image: numpy_ndarray) -> numpy_ndarray:
        """
        Aplica restauración facial.
        CORREGIDO: Usa float32 estándar para máxima compatibilidad y estabilidad.
        Eliminada la detección frágil de fp16.
        """
        if self.inferenceSession is None:
            self._load_inferenceSession()

        # Guardar datos originales
        original_h, original_w = self.get_image_resolution(image)

        # Manejo de Alpha (similar a la corrección #7)
        original_alpha = None
        if len(image.shape) == 3 and image.shape[2] == 4:
            original_alpha = image[:, :, 3]
            image = image[:, :, :3]  # Quedarse solo con RGB

        # 1. Redimensionar entrada si el usuario lo pidió
        resized_input = self.resize_with_input_factor(image)

        # 2. Preprocesar (Normalizar a 0-1 float32 y HWC->NCHW)
        preprocessed, _ = self.preprocess_face_image(resized_input)

        # 3. Inferencia (Directa en float32)
        try:
            input_name = self.inferenceSession.get_inputs()[0].name
            output_name = self.inferenceSession.get_outputs()[0].name

            # Ejecutar sin castings extraños
            output = self.inferenceSession.run(
                [output_name], {input_name: preprocessed})[0]

        except Exception as e:
            raise RuntimeError(f"GFPGAN inference failed: {e}")

        # 4. Postprocesar
        # Nota: pasamos el tamaño del resized_input para que el modelo devuelva
        # la cara restaurada en la escala de trabajo actual
        current_h, current_w = resized_input.shape[:2]
        restored_face = self.postprocess_face_image(
            output, (current_h, current_w))

        # 5. Restaurar Alpha si existía
        if original_alpha is not None:
            target_h, target_w = restored_face.shape[:2]
            upscaled_alpha = opencv_resize(
                original_alpha, (target_w, target_h), interpolation=INTER_CUBIC
            )
            if upscaled_alpha.ndim == 2:
                upscaled_alpha = numpy_expand_dims(upscaled_alpha, axis=2)
            restored_face = numpy_concatenate(
                (restored_face, upscaled_alpha), axis=2)

        # 6. Redimensionar salida final si el usuario lo pidió
        final_image = self.resize_with_output_factor(restored_face)

        return final_image
        """
        Orquestador principal: aplica restauración facial usando el modelo ONNX.
        Retorna la imagen restaurada (preservando alpha cuando sea necesario).
        """
        if self.inferenceSession is None:
            self._load_inferenceSession()

        # Guardar tamaño original y alpha si existe
        original_h, original_w = self.get_image_resolution(image)
        original_alpha = None
        if len(image.shape) == 3 and image.shape[2] == 4:
            original_alpha = image[:, :, 3]

        # Aplicar factor de input (si corresponde)
        try:
            resized_input = self.resize_with_input_factor(image)
        except Exception as e:
            print(f"[GFPGAN] Warning: resize_with_input_factor failed: {e}")
            resized_input = image

        # Preprocess -> float32
        preprocessed, had_alpha = self.preprocess_face_image(resized_input)

        # Detectar el dtype esperado por la sesión ONNX (si es posible)
        session_input = None
        input_type_str = None
        try:
            session_input = self.inferenceSession.get_inputs()[0]
            # Algunos objetos tienen .type o .dtype, algunos no; usamos str() como fallback
            if hasattr(session_input, 'type') and session_input.type:
                input_type_str = str(session_input.type)
            elif hasattr(session_input, 'dtype') and session_input.dtype:
                input_type_str = str(session_input.dtype)
            else:
                # Intentar inspeccionar la información de la firma
                try:
                    input_type_str = str(session_input)  # puede contener info
                except Exception:
                    input_type_str = None
        except Exception:
            input_type_str = None

        print(
            f"[GFPGAN] Pre-infer dtype(preprocessed)={preprocessed.dtype}, session_input_type={input_type_str}")

        # Convertir a float16 SOLO si la sesión lo requiere explícitamente
        run_input = preprocessed
        try:
            requires_fp16 = False
            if input_type_str:
                if 'float16' in input_type_str.lower() or 'fp16' in input_type_str.lower():
                    requires_fp16 = True
            if requires_fp16:
                # Convertimos sólo aquí, antes de pasar al modelo
                run_input = preprocessed.astype(float16)
                print(
                    "[GFPGAN] Convirtiendo input a float16 para la inferencia (según sesión).")
        except Exception as e:
            print(
                f"[GFPGAN] Warning: no se pudo convertir a float16: {e}. Manteniendo float32.")

        # Ejecutar la inferencia
        try:
            input_name = self.inferenceSession.get_inputs()[0].name
            output_name = self.inferenceSession.get_outputs()[0].name
            # Ejecutar la sesión (pasamos run_input)
            output = self.inferenceSession.run(
                [output_name], {input_name: run_input})[0]
        except Exception as e:
            raise RuntimeError(f"GFPGAN inference failed: {e}")

        # Postprocess
        restored_face = self.postprocess_face_image(
            output, (resized_input.shape[0], resized_input.shape[1]))

        # Restaurar canal alpha si era necesario
        if had_alpha and original_alpha is not None:
            try:
                alpha_resized = opencv_resize(
                    original_alpha, (restored_face.shape[1], restored_face.shape[0]), interpolation=INTER_CUBIC)
                if len(alpha_resized.shape) == 2:
                    alpha_resized = numpy_expand_dims(alpha_resized, axis=-1)
                restored_face = numpy_concatenate(
                    (restored_face, alpha_resized), axis=2)
            except Exception as e:
                print(
                    f"[GFPGAN] Warning: failed to restore alpha channel: {e}")

        # Aplicar factor de output (si corresponde)
        try:
            restored_face = self.resize_with_output_factor(restored_face)
        except Exception as e:
            print(f"[GFPGAN] Warning: resize_with_output_factor failed: {e}")

        return restored_face

    # -------------------
    # Orquestador público
    # -------------------

    def AI_orchestration(self, image: numpy_ndarray) -> numpy_ndarray:
        """
        Método público que otros módulos llaman para aplicar restauración facial.
        Maneja errores críticos de memoria re-lanzándolos.
        """
        try:
            return self.face_restoration(image)
        except Exception as e:
            error_str = str(e).lower()
            # --- CAMBIO CRÍTICO: Detectar errores de memoria ---
            # Si es un error de memoria (CUDA OOM), lo relanzamos para que el
            # orquestador principal active la reducción de tiles/resolución.
            if any(k in error_str for k in ['memory', 'cuda', 'allocation', 'resource']):
                raise e

            # Si es otro error (ej. datos corruptos), logueamos y devolvemos original
            print(f"[FACE RESTORATION ERROR] {str(e)}")
            return image


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

        self.attributes("-topmost", True)    # stay on top

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # create widgets with slight delay, to avoid white flickering of background
        self.after(10, self._create_widgets)
        self.grab_set()                # make other windows not clickable

        # Set minimum and maximum window sizes for better scrolling
        self.minsize(650, 500)
        self.maxsize(900, 800)

        # Set initial window size based on content
        self.geometry("600x400")

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
            title_subtitle_text_color = accent_color  # Amarillo dorado
        elif self._messageType == "error":
            title_subtitle_text_color = error_color  # Rojo brillante

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
                text_color=accent_color,  # Amarillo dorado
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
        # Create a scrollable frame for the options
        from customtkinter import CTkScrollableFrame

        self.scrollable_frame = CTkScrollableFrame(
            master=self,
            width=600,
            height=300,  # Fixed height to enable scrolling
            fg_color="transparent",
            corner_radius=10,
            scrollbar_button_color=border_color,
            scrollbar_button_hover_color=button_hover_color
        )

        self._ctkwidgets_index += 1
        self.scrollable_frame.grid(row=self._ctkwidgets_index, column=0,
                                   columnspan=2, padx=25, pady=10, sticky="ew")

        # Add options to the scrollable frame
        for i, option_text in enumerate(self._option_list):
            optionLabel = CTkLabel(
                master=self.scrollable_frame,
                width=550,  # Slightly smaller to account for scrollbar
                anchor='w',
                justify="left",
                text_color=text_color,
                fg_color=widget_background_color,
                bg_color="transparent",
                font=bold13,
                text=option_text,
                corner_radius=10,
                wraplength=530  # Enable text wrapping
            )

            optionLabel.grid(row=i, column=0, padx=10, pady=4, sticky="ew")

        # Configure grid weight for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

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
            fg_color=widget_background_color,
            text_color=secondary_text_color,
            border_color=accent_color,
        hover_color=button_hover_color,
        corner_radius=CORNER_RADIUS
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
    # Si el widget no existe o no tiene archivos, no hacemos nada crítico,
    # pero actualizamos los valores internos para cuando lleguen archivos.
    if not file_widget:
        return

    upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()

    # Pasar valores al manager
    file_widget.set_upscale_factor(upscale_factor)
    file_widget.set_input_resize_factor(input_resize_factor)
    file_widget.set_output_resize_factor(output_resize_factor)

    # Regenerar textos de info en la lista si es necesario
    if file_widget.queue_items:
        file_widget.regenerate_all_info()


def create_option_background():
    return CTkFrame(
        master=window,
        bg_color=background_color,
        fg_color=widget_background_color,
        height=46,
        corner_radius=CORNER_RADIUS
    )


def create_info_button(command: Callable, text: str, width: int = 200) -> CTkFrame:

    frame = CTkFrame(
        master=window, fg_color=widget_background_color, height=25, corner_radius=CORNER_RADIUS)

    button = CTkButton(
        master=frame,
        command=command,
        font=bold12,
        text="?",
        border_color=accent_color,
        border_width=1,
        fg_color=info_button_color,
        hover_color=button_hover_color,
        text_color=text_color,
        width=23,
        height=15,
        corner_radius=CORNER_RADIUS
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
    border_color: str = None,
    border_width: int = 1,
    width: int = 159
) -> CTkFrame:

    width = width
    height = 28

    total_width = (width + 2 * border_width)
    total_height = (height + 2 * border_width)

    # Use default border color if none provided
    if border_color is None:
        border_color = accent_color

    frame = CTkFrame(
        master=window,
        fg_color=border_color,
        width=total_width,
        height=total_height,
        border_width=0,
        corner_radius=CORNER_RADIUS,
    )

    option_menu = CTkOptionMenu(
        master=frame,
        command=command,
        values=values,
        width=width,
        height=height,
        corner_radius=max(CORNER_RADIUS - 2, 0),
        dropdown_font=bold12,
        font=bold11,
        anchor="center",
        text_color=text_color,
        fg_color=widget_background_color,
        button_color=widget_background_color,
        button_hover_color=button_hover_color,
        dropdown_fg_color=widget_background_color,
        dropdown_text_color=text_color,
        dropdown_hover_color=button_hover_color
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
        corner_radius=CORNER_RADIUS,
        width=width,
        height=28,
        font=bold11,
        justify="center",
        text_color=text_color,
        fg_color=widget_background_color,
        border_width=1,
        border_color=accent_color,
        placeholder_text_color=secondary_text_color
    )


def create_text_box_output_path(textvariable: StringVar) -> CTkEntry:
    return CTkEntry(
        master=window,
        textvariable=textvariable,
        corner_radius=CORNER_RADIUS,
        width=250,
        height=28,
        font=bold11,
        justify="center",
        text_color=secondary_text_color,
        fg_color=widget_background_color,
        border_width=1,
        border_color=border_color,
        state=DISABLED
    )


def create_active_button(
        command: Callable,
        text: str,
        icon: CTkImage = None,
        width: int = 140,
        height: int = 30,
        border_color: str = None
) -> CTkButton:

    # Use default border color if none provided
    if border_color is None:
        border_color = accent_color

    return CTkButton(
        master=window,
        command=command,
        text=text,
        image=icon,
        width=width,
        height=height,
        font=bold11,
        border_width=1,
        corner_radius=CORNER_RADIUS,
        fg_color=widget_background_color,
        text_color=text_color,
        border_color=border_color,
        hover_color=button_hover_color
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
    """Optimize memory usage by triggering garbage collection and clearing caches."""
    try:
        import gc
        import sys

        # Force garbage collection
        gc.collect()

        # Clear any cached frames or temporary data
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()

        # Additional memory optimization for Windows
        try:
            import ctypes
            if hasattr(ctypes, 'windll'):
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except Exception:
            pass

    except Exception as e:
        logging.debug(f"Memory optimization warning: {str(e)}")


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
            'frame_') and f.endswith('.png')]

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
        # Extract number from patterns like "frame_001.png"
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
    """
    Crea un directorio si no existe.
    CORREGIDO: Ya no borra el directorio si existe (evita pérdida de datos).
    """
    try:
        if not os_path_exists(name_dir):
            os_makedirs(name_dir, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Could not create directory {name_dir}: {e}")


def stop_thread() -> None:
    """Notifica al hilo de monitoreo que debe detenerse de forma segura."""
    global stop_thread_flag
    stop_thread_flag.set()


def image_read(file_path: str) -> numpy_ndarray:
    """Enhanced image reading with comprehensive error handling and validation."""
    try:
        if not os_path_exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Check file size
        file_size = os_path_getsize(file_path)
        if file_size == 0:
            raise ValueError(f"Image file is empty: {file_path}")

        # Limit maximum file size (500MB) to prevent memory issues
        max_size = 500 * 1024 * 1024  # 500MB
        if file_size > max_size:
            raise ValueError(
                f"Image file too large ({file_size / (1024*1024):.1f}MB > 500MB): {file_path}")

        with open(file_path, 'rb') as file:
            file_data = file.read()

        # Validate file data
        if len(file_data) == 0:
            raise ValueError(f"Could not read image data: {file_path}")

        # Decode image
        buffer = numpy_ascontiguousarray(numpy_frombuffer(file_data, uint8))
        image = opencv_imdecode(buffer, IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(
                f"Could not decode image (corrupted or unsupported format): {file_path}")

        # Validate image properties
        if image.size == 0:
            raise ValueError(f"Decoded image has zero size: {file_path}")

        # Check for reasonable dimensions
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid image dimensions ({width}x{height}): {file_path}")

        # Check for extremely large dimensions that could cause memory issues
        max_dimension = 32768  # 32K pixels per dimension
        if height > max_dimension or width > max_dimension:
            raise ValueError(
                f"Image dimensions too large ({width}x{height} > {max_dimension}x{max_dimension}): {file_path}")

        # Validate channel count
        channels = len(image.shape) if len(
            image.shape) == 2 else image.shape[2]
        if len(image.shape) == 3 and channels not in [1, 3, 4]:
            logging.warning(
                f"Unusual channel count ({channels}) in image: {file_path}")

        print(
            f"[IMAGE] Successfully loaded: {width}x{height}x{channels if len(image.shape) > 2 else 1} - {file_size / 1024:.1f}KB")
        return image

    except Exception as e:
        error_msg = f"Failed to read image {os_path_basename(file_path)}: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)


def image_write(file_path: str, file_data: numpy_ndarray, file_extension: str = ".png") -> None:
    """
    Escribe la imagen en disco de forma robusta, manejando rutas Unicode y buffer de escritura.
    """
    try:
        # 1. Codificar la imagen a memoria (buffer) usando la extensión correcta
        success, buffer = opencv_imencode(file_extension, file_data)

        if not success:
            raise RuntimeError(
                f"Could not encode image format: {file_extension}")

        # 2. Escribir el buffer al disco usando manejo estándar de archivos
        # Esto evita problemas con rutas non-ASCII que tiene cv2.imwrite
        with open(file_path, "wb") as f:
            f.write(buffer.tobytes())

    except Exception as e:
        # Loguear el error pero permitir que el proceso principal lo maneje si es necesario
        logging.error(f"Failed to write image to {file_path}: {str(e)}")
        raise RuntimeError(f"Failed to write image: {str(e)}")


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

        # CORRECCIÓN: Agregar errors='replace' y encoding='utf-8' (o dejar encoding default pero con replace)
        result = subprocess_run(exiftool_cmd, check=True,
                                shell=False, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')  # <--- CAMBIO AQUÍ
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
    to_append += f".png"

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
    """Get video frame rate with proper validation."""
    video_capture = opencv_VideoCapture(video_path)

    if not video_capture.isOpened():
        video_capture.release()
        raise ValueError(f"Could not open video file: {video_path}")

    frame_rate = video_capture.get(CAP_PROP_FPS)
    video_capture.release()

    if frame_rate <= 0 or frame_rate > 1000:  # Sanity check
        raise ValueError(
            f"Invalid frame rate: {frame_rate} for video: {video_path}")

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
) -> List[str]:
    """Extract frames from video with proper error handling."""
    try:
        create_dir(target_directory)

        # Check if video file exists
        if not os_path_exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames_number_to_save = cpu_number * ECTRACTION_FRAMES_FOR_CPU
        video_capture = opencv_VideoCapture(video_path)

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
    fps_multiplier: int = 1,
) -> None:
    """
    Video encoding function for Warlock-Studio.
    INCLUDES AUTOMATIC FALLBACK TO X264.
    """

    try:
        # ... (El código de preparación de rutas y FPS se mantiene igual) ...
        # Copia todo el inicio de la función hasta llegar a la parte de subprocess

        # --- Preparación de rutas temporales ---
        base_name = os_path_splitext(video_output_path)[0]
        txt_path = f"{base_name}_frames.txt"
        no_audio_path = f"{base_name}_no_audio{os_path_splitext(video_output_path)[1]}"

        # Eliminar residuos previos
        for temp_file in [txt_path, no_audio_path]:
            if os_path_exists(temp_file):
                try:
                    os_remove(temp_file)
                except Exception as e:
                    print(
                        f"[WARNING] Temporary file could not be deleted {temp_file}: {e}")

        # --- Obtener FPS ---
        try:
            video_fps = get_video_fps(video_path)
            if video_fps <= 0 or video_fps > 1000:
                raise ValueError(f"FPS inválido: {video_fps}")
            final_fps = video_fps * fps_multiplier
            video_fps_str = f"{final_fps:.6f}"
        except Exception as e:
            print(
                f"[WARNING] Could not obtain FPS: {e}, using 30.0 by default")
            video_fps_str = "30.000000"

        # --- Crear lista de frames ---
        if not create_frame_list_file(upscaled_frame_paths, txt_path):
            raise RuntimeError("Error creating the frames list file")

        # ==============================================================================
        #  LOGICA DE FALLBACK (INTENTOS DE CODIFICACIÓN)
        # ==============================================================================
        encoding_success = False

        for attempt in range(2):
            try:
                current_codec = selected_video_codec
                if attempt == 1:
                    process_status_q.put(
                        f"[LOG] [WARNING] Hardware encoding failed. Switching to CPU fallback (x264)...")
                    current_codec = 'x264'

                codec_settings = get_video_codec_settings(
                    current_codec, {'fps': video_fps_str})
                encoding_command = build_encoding_command(
                    video_path, txt_path, no_audio_path, codec_settings, video_fps_str)

                process_status_q.put(
                    f"[LOG] [FFMPEG] Attempt {attempt+1}: Encoding with {codec_settings['codec']}")

                # --- Ejecutar FFmpeg (CORREGIDO CON ERRORS='REPLACE') ---
                process = subprocess.Popen(
                    encoding_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',  # <--- ESTO YA ESTABA, PERO ASEGÚRATE QUE ESTÉ PRESENTE
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )

                for line in process.stdout:
                    line = line.strip()
                    if line:
                        process_status_q.put(f"[LOG] {line}")

                process.wait()

                if process.returncode != 0:
                    if attempt == 0:
                        raise RuntimeError(
                            f"FFmpeg failed with code {process.returncode}")
                    else:
                        raise RuntimeError("FFmpeg CPU Fallback also failed.")

                if not os_path_exists(no_audio_path) or os_path_getsize(no_audio_path) < 1024:
                    if attempt == 0:
                        raise RuntimeError("Output file missing or too small")
                    else:
                        raise RuntimeError(
                            "Output file missing or too small after fallback")

                process_status_q.put(f"[LOG] [FFMPEG] Encoding complete")
                encoding_success = True
                break

            except Exception as e:
                if attempt == 0:
                    process_status_q.put(
                        f"[LOG] [WARNING] Primary encoding failed: {e}. Preparing fallback...")
                    if os_path_exists(no_audio_path):
                        try:
                            os_remove(no_audio_path)
                        except:
                            pass
                    continue
                else:
                    raise e

        if not encoding_success:
            raise RuntimeError("Encoding failed after all attempts.")

        # ==============================================================================
        #  DETECTAR AUDIO (AQUÍ ES DONDE SUELE FALLAR EL 0xe1)
        # ==============================================================================
        process_status_q.put(
            "[LOG] [FFMPEG] Checking audio track of original video...")

        # ... (Código de búsqueda de ffprobe se mantiene igual) ...
        ffprobe_path = None
        try:
            ffprobe_guess = FFMPEG_EXE_PATH.replace(
                "ffmpeg.exe", "ffprobe.exe")
            if os_path_exists(ffprobe_guess):
                ffprobe_path = ffprobe_guess
            else:
                ffprobe_guess2 = FFMPEG_EXE_PATH.replace(
                    "ffmpeg.exe", "ffprobe")
                if os_path_exists(ffprobe_guess2):
                    ffprobe_path = ffprobe_guess2
        except:
            ffprobe_path = None

        has_audio = False
        audio_codec = ""

        if ffprobe_path:
            try:
                probe_cmd = [
                    ffprobe_path, "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=noprint_wrappers=1:nokey=1", video_path
                ]
                # CORRECCIÓN: Agregar errors='replace'
                probe = subprocess_run(probe_cmd, capture_output=True, text=True,
                                       encoding='utf-8', errors='replace', timeout=30)  # <--- CAMBIO
                audio_codec = probe.stdout.strip()
                has_audio = bool(audio_codec)
            except Exception as e:
                print(f"[WARNING] Audio probe failed: {e}")
                has_audio = False

        if not ffprobe_path or not has_audio:
            try:
                probe_cmd = [FFMPEG_EXE_PATH, "-i", video_path]
                # CORRECCIÓN: Agregar errors='replace'
                probe = subprocess_run(probe_cmd, capture_output=True, text=True,
                                       encoding='utf-8', errors='replace', timeout=20)  # <--- CAMBIO
                stderr_output = probe.stderr or probe.stdout or ""
                if "Audio:" in stderr_output:
                    has_audio = True
            except Exception:
                pass

        process_status_q.put(
            f"[LOG] [FFMPEG] has_audio={has_audio}, codec={audio_codec}")

        if has_audio:
            audio_copy_cmd = [
                FFMPEG_EXE_PATH, "-y", "-loglevel", "error",
                "-i", video_path, "-i", no_audio_path,
                "-c:v", "copy", "-c:a", "copy",
                "-map", "1:v:0", "-map", "0:a",
                "-shortest", video_output_path
            ]
            try:
                process_status_q.put(
                    "[LOG] [FFMPEG] Trying to copy audio track...")
                # CORRECCIÓN: Agregar errors='replace'
                subprocess_run(audio_copy_cmd, check=True, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')  # <--- CAMBIO

                if os_path_exists(no_audio_path):
                    os_remove(no_audio_path)
                process_status_q.put(
                    "[LOG] [FFMPEG] Audio copy completed successfully.")
                return
            except Exception as e:
                process_status_q.put(
                    f"[LOG] [WARNING] Audio copy failed, re-encoding... ({e})")

            audio_reencode_cmd = [
                FFMPEG_EXE_PATH, "-y", "-loglevel", "error",
                "-i", video_path, "-i", no_audio_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "1:v:0", "-map", "0:a",
                "-shortest", video_output_path
            ]
            try:
                # CORRECCIÓN: Agregar errors='replace'
                subprocess_run(audio_reencode_cmd, check=True, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')  # <--- CAMBIO

                if os_path_exists(no_audio_path):
                    os_remove(no_audio_path)
                process_status_q.put(
                    "[LOG] [FFMPEG] Audio re-encoding completed.")
                return
            except Exception as audio_error:
                process_status_q.put(
                    f"[LOG] [WARNING] Audio re-encoding failed: {audio_error}")

            try:
                if os_path_exists(no_audio_path):
                    shutil_move(no_audio_path, video_output_path)
                    process_status_q.put(
                        "[LOG] [FFMPEG] Final video saved WITHOUT audio (fallback).")
                    return
            except Exception as move_error:
                raise RuntimeError(f"Could not move final file: {move_error}")

        else:
            try:
                shutil_move(no_audio_path, video_output_path)
                process_status_q.put(
                    "[LOG] [FFMPEG] Video saved (original was muted).")
                return
            except Exception as move_error:
                raise RuntimeError(f"Video could not be saved: {move_error}")

    except Exception as e:
        error_msg = f"General error in video_encoding: {str(e)}"
        log_and_report_error(error_msg)
        write_process_status(process_status_q, f"{ERROR_STATUS}{error_msg}")
        if 'txt_path' in locals() and os_path_exists(txt_path):
            try:
                os_remove(txt_path)
            except:
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
        file for file in directory_files if file.endswith('.png')]
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
        file_extension: str = ".png"
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
        else:
            return "Unknown"

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
        starting_mode = get_image_mode(starting_image)
        upscaled_mode = get_image_mode(upscaled_image)

        if starting_mode == "RGBA" or upscaled_mode == "RGBA":
            if starting_mode != "RGBA":
                starting_image = add_alpha_channel(starting_image)
            if upscaled_mode != "RGBA":
                upscaled_image = add_alpha_channel(upscaled_image)
        elif starting_mode == "RGB" and upscaled_mode != "RGB":
            if upscaled_mode == "Grayscale":
                upscaled_image = opencv_cvtColor(
                    upscaled_image, COLOR_GRAY2RGB)
        elif upscaled_mode == "RGB" and starting_mode != "RGB":
            if starting_mode == "Grayscale":
                starting_image = opencv_cvtColor(
                    starting_image, COLOR_GRAY2RGB)

        if starting_image.dtype != upscaled_image.dtype:
            upscaled_image = upscaled_image.astype(starting_image.dtype)

        interpolated_image = opencv_addWeighted(
            starting_image, starting_image_importance, upscaled_image, upscaled_image_importance, 0)
        image_write(target_path, interpolated_image, file_extension)

    except Exception as e:
        print(
            f"[BLEND] Blending failed, saving original upscaled image: {str(e)}")
        image_write(target_path, upscaled_image, file_extension)


# ==== CORE PROCESSING SECTION (CORREGIDO) ====

def check_upscale_steps() -> None:
    global stop_thread_flag, app

    # Limpiar flag al iniciar
    stop_thread_flag.clear()

    while not stop_thread_flag.is_set():
        try:
            # Lectura no bloqueante
            actual_step = read_process_status()

            if actual_step is None:
                # Si no hay mensajes, verificar si el proceso sigue vivo
                # Si el proceso murió inesperadamente sin mandar "Completed" o "Stop", salimos
                global process_upscale_orchestrator
                if process_upscale_orchestrator and not process_upscale_orchestrator.is_alive() and not stop_thread_flag.is_set():
                    # El proceso murió silenciosamente, forzamos el stop
                    actual_step = STOP_STATUS
                else:
                    # Simplemente esperar y reintentar
                    sleep(0.1)
                    continue

            # --- DETECTAR LOGS ---
            if actual_step.startswith("[LOG]"):
                log_text = actual_step.replace("[LOG] ", "")
                console.write_log(log_text)
                continue

            # --- LÓGICA DE FINALIZACIÓN (Completado, Stop o Error) ---
            if actual_step in [COMPLETED_STATUS, STOP_STATUS] or ERROR_STATUS in actual_step:

                if actual_step == COMPLETED_STATUS:
                    info_message.set("All files completed!")
                    console.write_log(
                        "Process Completed Successfully", "SUCCESS")
                elif actual_step == STOP_STATUS:
                    info_message.set("Process Stopped")
                    console.write_log("Process stopped by user", "WARNING")
                elif ERROR_STATUS in actual_step:
                    err_msg = actual_step.replace(ERROR_STATUS, "")
                    info_message.set("Error occurred")
                    console.write_log(f"Error: {err_msg}", "ERROR")
                    show_error_message(err_msg)

                # 1. Detener proceso físico
                stop_upscale_process()

                # 2. Romper el bucle
                stop_thread_flag.set()

                # 3. RESTAURAR UI (Regresar botón a "Make Magic")
                # Usamos .after para asegurar que corra en el hilo principal de la UI
                window.after(100, place_upscale_button)
                break

            # --- ESTADOS INTERMEDIOS ---
            else:
                info_message.set(actual_step)
                # Opcional, puede ser mucho spam
                console.write_log(f"Status: {actual_step}", "INFO")

        except Exception as e:
            print(f"[MONITOR ERROR] {e}")
            stop_thread_flag.set()
            window.after(100, place_upscale_button)
            break


def read_process_status() -> str:
    try:
        # Intentar leer con un timeout muy corto (0.05s)
        # Esto evita que la GUI se congele si no hay mensajes
        return process_status_q.get(timeout=0.05)
    except Exception:
        # Si la cola está vacía (TimeOut), devolvemos None
        return None


def write_process_status(process_status_q: multiprocessing_Queue, step: str) -> None:
    # CORRECCIÓN CRITICA: NO VACIAR LA COLA ANTES DE ESCRIBIR
    # Esto borraba los logs antes de que pudieran leerse.
    print(f"[QUEUE] Put: {step}")
    process_status_q.put(f"{step}")


def stop_upscale_process() -> None:
    global process_upscale_orchestrator
    try:
        if 'process_upscale_orchestrator' in globals() and process_upscale_orchestrator:
            if process_upscale_orchestrator.is_alive():
                print("Terminating process...")
                process_upscale_orchestrator.terminate()
                # Esperar máx 1 segundo a que muera, si no, continuar
                process_upscale_orchestrator.join(timeout=1.0)
                if process_upscale_orchestrator.is_alive():
                    # Si sigue vivo (zombie), kill forzado (Python 3.7+)
                    try:
                        process_upscale_orchestrator.kill()
                    except:
                        pass

            process_upscale_orchestrator = None
    except Exception as e:
        print(f"Error stopping process: {e}")


def stop_button_command() -> None:
    # 1. Notificar visualmente inmediato
    info_message.set("Stopping...")

    # 2. Matar el proceso inmediatamente
    stop_upscale_process()

    # 3. Enviar señal a la cola para que el hilo de monitoreo (check_upscale_steps)
    # sepa que debe salir y restaurar la UI.
    write_process_status(process_status_q, STOP_STATUS)


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
    global chain_window  # Necesario para acceder al gestor de cadenas

    # 1. Confirmación de seguridad
    if not messagebox.askyesno("Start Processing", "Do you want to start the AI processing?"):
        return

    # 2. Limpiar bandera de parada
    stop_thread_flag.clear()

    # 3. Validar entradas del usuario
    if user_input_checks():
        info_message.set("Loading")
        cpu_number = int(os_cpu_count() / 2)

        # --- LÓGICA DE ENCADENAMIENTO (CHAINING) ---
        active_chain = []

        # Verificar si la ventana de cadena existe y tiene pasos
        if 'chain_window' in globals() and chain_window is not None:
            try:
                # Verificar si la ventana no ha sido destruida
                if chain_window.winfo_exists():
                    active_chain = chain_window.get_chain()
            except Exception:
                # Si la ventana fue cerrada, chain_window puede quedar con referencia rota
                pass

        # Determinar si vamos a usar cadena o proceso simple
        is_chain_active = len(active_chain) > 0

        # --- VISUALIZACIÓN DE LOGS (Consola) ---
        print("=" * 50)
        print(f"> Starting Process:")
        print(f"  Files to process: {len(selected_file_list)}")
        print(f"  Output path: {selected_output_path.get()}")
        print(f"  CPU threads: {cpu_number}")

        if is_chain_active:
            print(
                f"  [MODE] CHAIN PROCESSING ACTIVE ({len(active_chain)} Steps)")
            for i, step in enumerate(active_chain):
                print(f"    Step {i+1}: {step}")
        else:
            print(f"  [MODE] SINGLE PROCESSING")
            print(f"  Selected AI model: {selected_AI_model}")
            print(f"  Selected GPU: {selected_gpu}")
            print(f"  Input resize: {int(input_resize_factor * 100)}%")
            print(f"  Output resize: {int(output_resize_factor * 100)}%")
            print(f"  Frame Gen: {selected_frame_generation_option}")
            print(f"  VRAM/Tiles: {tiles_resolution}x{tiles_resolution}px")

        print("=" * 50)

        # Colocar botón de STOP
        place_stop_button()

        # --- SELECCIÓN DE PROCESO (ORCHESTRATOR) ---

        if is_chain_active:
            # CASO A: CADENA ACTIVA
            # Usamos el upscale_orchestrator actualizado, pasando la lista de pasos
            process_upscale_orchestrator = Process(
                target=upscale_orchestrator,
                args=(
                    process_status_q,
                    selected_file_list,
                    selected_output_path.get(),
                    # Se ignora si hay chain, pero se pasa por compatibilidad posicional
                    selected_AI_model,
                    selected_AI_multithreading,
                    input_resize_factor,        # Se ignora si hay chain
                    output_resize_factor,       # Se ignora si hay chain
                    selected_gpu,               # Se ignora si hay chain
                    tiles_resolution,
                    selected_blending_factor,   # Se ignora si hay chain
                    selected_keep_frames,
                    selected_image_extension,
                    selected_video_extension,
                    selected_video_codec,
                    cpu_number,
                    active_chain                # <--- AQUÍ PASAMOS LA CADENA
                )
            )
            process_upscale_orchestrator.start()

        elif selected_AI_model in RIFE_models_list:
            # CASO B: RIFE / INTERPOLACIÓN (Sin Cadena)
            # FluidFrames usa su propio pipeline específico
            process_upscale_orchestrator = Process(
                target=fluidframes_interpolation_pipeline,
                args=(
                    process_status_q,
                    selected_file_list,
                    selected_output_path.get(),
                    selected_AI_model,
                    selected_gpu,
                    selected_frame_generation_option,
                    selected_image_extension,
                    selected_video_extension,
                    selected_video_codec,
                    input_resize_factor,
                    output_resize_factor,
                    cpu_number,
                    selected_keep_frames
                )
            )
            process_upscale_orchestrator.start()

        else:
            # CASO C: UPSCALE / RESTAURACIÓN NORMAL (Sin Cadena)
            # Pasamos None en el último argumento para que el orquestador cree el paso único
            process_upscale_orchestrator = Process(
                target=upscale_orchestrator,
                args=(
                    process_status_q,
                    selected_file_list,
                    selected_output_path.get(),
                    selected_AI_model,
                    selected_AI_multithreading,
                    input_resize_factor,
                    output_resize_factor,
                    selected_gpu,
                    tiles_resolution,
                    selected_blending_factor,
                    selected_keep_frames,
                    selected_image_extension,
                    selected_video_extension,
                    selected_video_codec,
                    cpu_number,
                    None  # <--- SIN CADENA
                )
            )
            process_upscale_orchestrator.start()

        # Iniciar hilo de monitoreo de la interfaz
        thread_wait = Thread(target=check_upscale_steps)
        thread_wait.start()

# --- Inserted: FluidFrames orchestration (minimal, reusing classes/logic copied from FluidFrames.py) ---


def fluidframes_interpolation_pipeline(
        process_status_q, selected_file_list, selected_output_path, selected_AI_model, selected_gpu,
        selected_generation_option, selected_image_extension, selected_video_extension, selected_video_codec,
        input_resize_factor, output_resize_factor, cpu_number, selected_keep_frames):
    '''
    This function runs all the FluidFrames video/image interpolation generation logic in one go for Warlock-Studio.
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

    # Forzar .png para extracción temporal
    temp_extraction_ext = ".png"

    extracted_frames_paths = extract_video_frames(
        process_status_q, file_number, target_directory, AI_instance, video_path, cpu_number, temp_extraction_ext)

    # Step 3. Prepare output/gen frame names
    total_frames_paths = prepare_output_video_frame_filenames(
        extracted_frames_paths, selected_AI_model, frame_gen_factor, temp_extraction_ext)

    # Step 4. Interpolated frames generation
    write_process_status(
        process_status_q, f"{file_number}. Video frame generation initializing...")

    # --- LOGICA DE PROGRESO AÑADIDA ---
    global global_processing_times_list
    global_processing_times_list = []

    total_pairs = len(extracted_frames_paths) - 1

    for frame_index in range(total_pairs):
        frame_1_path = extracted_frames_paths[frame_index]
        frame_2_path = extracted_frames_paths[frame_index+1]

        # Medir tiempo de carga e inferencia
        start_timer = timer()

        frame_1 = image_read(frame_1_path)
        frame_2 = image_read(frame_2_path)

        generated_frames = AI_instance.AI_orchestration(frame_1, frame_2)

        # Save generated frames
        generated_frames_paths = prepare_generated_frames_paths(
            os_path_splitext(frame_1_path)[0], selected_AI_model, temp_extraction_ext, frame_gen_factor)

        for i, gen_frame in enumerate(generated_frames):
            image_write(generated_frames_paths[i], gen_frame)

        end_timer = timer()

        # --- CÁLCULO DE TIEMPO Y ACTUALIZACIÓN DE ESTADO ---
        step_time = end_timer - start_timer
        global_processing_times_list.append(step_time)

        # Limitamos el tamaño de la lista de tiempos para mantener el promedio reciente
        if len(global_processing_times_list) > 100:
            global_processing_times_list.pop(0)

        # Actualizar la interfaz cada 1% o cada frame si son pocos (evita saturar la GUI)
        if total_pairs > 0 and (frame_index % max(1, int(total_pairs / 100)) == 0 or frame_index == total_pairs - 1):
            avg_time = numpy_mean(global_processing_times_list)
            remaining_frames = total_pairs - (frame_index + 1)
            time_left = calculate_time_to_complete_video(
                avg_time, remaining_frames)
            percent_complete = ((frame_index + 1) / total_pairs) * 100

            status_msg = f"{file_number}. Interpolating frames: {percent_complete:.1f}% completed ({time_left})"
            write_process_status(process_status_q, status_msg)

    # Step 6. Video encoding
    write_process_status(
        process_status_q, f"{file_number}. Encoding frame-generated video")

    fps_multiplier = 1 if slowmotion else frame_gen_factor

    video_encoding(
        process_status_q, video_path, video_output_path, total_frames_paths, selected_video_codec, fps_multiplier)

    # Step 7. Cleanup
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
    input_resize_factor: float,
    output_resize_factor: float,
    selected_gpu: str,
    tiles_resolution: int,
    selected_blending_factor: float,
    selected_keep_frames: bool,
    selected_image_extension: str,
    selected_video_extension: str,
    selected_video_codec: str,
    cpu_number: int,
    # NUEVO ARGUMENTO: Lista de pasos para encadenamiento
    chain_steps: list = None
) -> None:

    global global_status_lock
    global_status_lock = Lock()

    # Lista para rastrear carpetas temporales y limpiarlas al final
    temp_folders_created = []

    try:
        # ---------------------------------------------------------------------
        # 1. NORMALIZACIÓN DE LA CADENA
        # Si no hay cadena (uso normal), creamos un paso único con los args recibidos.
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # 1. NORMALIZACIÓN DE LA CADENA
        # ---------------------------------------------------------------------
        if not chain_steps:
            # Crea un paso único si no se usa el gestor de cadenas
            # Nota: vram_limit se pasa tal cual, el orquestador calculará los tiles
            try:
                vram_val = float(selected_VRAM_limiter.get())
            except:
                vram_val = 0.0

            initial_step = ProcessingStep(
                model_name=selected_AI_model,
                input_resize=input_resize_factor,
                output_resize=output_resize_factor,
                blending=selected_blending_factor,
                vram_limit=vram_val,
                extension=selected_image_extension if not check_if_file_is_video(
                    selected_file_list[0]) else selected_video_extension,
                video_codec=selected_video_codec,
                frame_gen="OFF",
                keep_frames=selected_keep_frames,
                gpu=selected_gpu
            )
            chain_steps = [initial_step]

        how_many_files = len(selected_file_list)

        # ---------------------------------------------------------------------
        # 2. BUCLE PRINCIPAL DE ARCHIVOS
        # ---------------------------------------------------------------------
        for file_number in range(how_many_files):
            # El archivo de entrada inicial
            original_input_path = selected_file_list[file_number]
            current_input_path = original_input_path

            # Nombre base para logs
            base_filename = os_path_basename(original_input_path)
            display_number = file_number + 1

            # -----------------------------------------------------------------
            # 3. BUCLE DE LA CADENA DE PROCESOS (STEPS)
            # -----------------------------------------------------------------
            for step_index, step in enumerate(chain_steps):

                # Identificar si es el último paso
                is_last_step = (step_index == len(chain_steps) - 1)
                step_display = f"[Step {step_index + 1}/{len(chain_steps)}]"

                # --- CONFIGURACIÓN DEL PASO ACTUAL ---
                current_model = step.model_name
                current_input_resize = step.input_resize
                current_output_resize = step.output_resize
                current_blending = step.blending
                current_gpu = step.gpu

                # Determinar extensión de salida para este paso
                is_video = check_if_file_is_video(current_input_path)
                if is_video:
                    # Temp video container
                    current_extension = selected_video_extension if is_last_step else ".mp4"
                    current_codec = step.video_codec if step.video_codec else selected_video_codec
                else:
                    current_extension = step.extension if step.extension else selected_image_extension
                try:
                    if is_video and current_extension.lower() not in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
                        current_extension = ".mp4"
                    if (not is_video) and current_extension.lower() not in [".png", ".jpg", ".bmp", ".tiff", ".webp"]:
                        current_extension = ".png"
                except Exception:
                    pass

                # --- GESTIÓN DE RUTAS DE SALIDA ---
                if is_last_step:
                    # El último paso va a la carpeta seleccionada por el usuario
                    current_output_dir = selected_output_path
                else:
                    # Pasos intermedios van a una carpeta temporal única
                    current_output_dir = os_path_join(
                        os_path_dirname(original_input_path),
                        f"warlock_temp_step_{step_index}_{int(time.time())}"
                    )
                    if not os_path_exists(current_output_dir):
                        create_dir(current_output_dir)
                        temp_folders_created.append(current_output_dir)

                # --- LOG DE ESTADO ---
                write_process_status(
                    process_status_q,
                    f"{display_number}. {step_display} Loading {current_model}..."
                )

                is_interpolation_step = (current_model in RIFE_models_list) or (step.frame_gen and step.frame_gen != "OFF")

                if is_interpolation_step:
                    # Validar que la entrada actual sea video
                    if not is_video:
                        write_process_status(
                            process_status_q,
                            f"{display_number}. {step_display} Skipping: Interpolation requires a video input"
                        )
                        # Mantener current_input_path sin cambios y continuar con el siguiente paso
                        continue

                    # Resolver factor de generación y modo slowmotion a partir de frame_gen del paso
                    frame_gen_factor, slowmotion = check_frame_generation_option(step.frame_gen)

                    try:
                        AI_interp = AI_interpolation(
                            current_model,
                            frame_gen_factor,
                            current_gpu,
                            current_input_resize,
                            current_output_resize
                        )
                    except Exception as e:
                        raise RuntimeError(f"Error loading interpolation model {current_model}: {e}")

                    write_process_status(
                        process_status_q,
                        f"{display_number}. {step_display} Interpolating video..."
                    )

                    video_container_ext = current_extension if is_last_step else ".mp4"
                    codec_to_use = current_codec

                    try:
                        fluidframes_video_interpolate(
                            process_status_q,
                            current_input_path,
                            display_number,
                            current_output_dir,
                            AI_interp,
                            current_model,
                            frame_gen_factor,
                            slowmotion,
                            ".png",                     # extracción temporal siempre PNG
                            video_container_ext,
                            codec_to_use,
                            current_input_resize,
                            current_output_resize,
                            cpu_number,
                            step.keep_frames if is_last_step else False
                        )
                    except Exception as e:
                        raise RuntimeError(f"Interpolation failed: {e}")

                    expected_filename = prepare_output_video_filename(
                        current_input_path,
                        current_output_dir,
                        current_model,
                        frame_gen_factor,
                        slowmotion,
                        current_input_resize,
                        current_output_resize,
                        video_container_ext
                    )
                else:
                    # --- INSTANCIACIÓN DEL MODELO (UPSCALE/RESTORE) ---
                    # Recalcular tiles si hay límite de VRAM en el paso
                    current_tiles_resolution = tiles_resolution
                    if step.vram_limit > 0:
                        vram_multiplier = VRAM_model_usage.get(current_model, 1.0)
                        current_tiles_resolution = int(
                            (vram_multiplier * step.vram_limit) * 100)

                    # Crear instancias
                    AI_instances = []
                    try:
                        # Determinar clase (Face Restore vs Upscale Genérico)
                        if current_model in Face_restoration_models_list:
                            AI_instances = [
                                AI_face_restoration(
                                    current_model,
                                    current_gpu,
                                    current_input_resize,
                                    current_output_resize,
                                    current_tiles_resolution
                                ) for _ in range(selected_AI_multithreading)
                            ]
                        else:
                            AI_instances = [
                                AI_upscale(
                                    current_model,
                                    current_gpu,
                                    current_input_resize,
                                    current_output_resize,
                                    current_tiles_resolution
                                ) for _ in range(selected_AI_multithreading)
                            ]
                    except Exception as e:
                        raise RuntimeError(
                            f"Error loading model {current_model}: {e}")

                # --- EJECUCIÓN DEL PROCESAMIENTO ---
                write_process_status(
                    process_status_q,
                    f"{display_number}. {step_display} Processing {base_filename}..."
                )

                # Predecir el nombre del archivo de salida para poder pasarlo al siguiente paso
                if not is_interpolation_step:
                    if is_video:
                        # Lógica para Video
                        upscale_video(
                            process_status_q,
                            current_input_path,
                            display_number,  # Mantiene el número original
                            current_output_dir,
                            AI_instances,
                            current_model,
                            current_input_resize,
                            current_output_resize,
                            cpu_number,
                            current_extension,
                            current_blending,
                            selected_AI_multithreading,
                            # Solo guardar frames si es el último y el usuario quiere
                            step.keep_frames if is_last_step else False,
                            current_codec
                        )

                        # Calcular cuál fue el archivo resultante para usarlo de input en el siguiente paso
                        expected_filename = prepare_output_video_filename(
                            current_input_path,
                            # Importante: Buscar en el dir actual (temp o final)
                            current_output_dir,
                            current_model,
                            1,  # frame_gen factor (upscale es 1)
                            False,  # slowmo
                            current_input_resize,
                            current_output_resize,
                            current_extension
                        )

                    else:
                        # Lógica para Imagen
                        # Nota: upscale_image solo procesa una instancia, pasamos la primera
                        upscale_image(
                            process_status_q,
                            current_input_path,
                            display_number,
                            current_output_dir,
                            AI_instances[0],
                            current_model,
                            current_extension,
                            current_input_resize,
                            current_output_resize,
                            current_blending
                        )

                        # Calcular archivo resultante
                        expected_filename = prepare_output_image_filename(
                            current_input_path,
                            current_output_dir,
                            current_model,
                            current_input_resize,
                            current_output_resize,
                            current_extension,
                            current_blending
                        )

                # --- PREPARACIÓN PARA EL SIGUIENTE PASO ---
                # Verificar que el archivo se creó correctamente
                if not os_path_exists(expected_filename):
                    raise FileNotFoundError(
                        f"Step {step_index+1} failed. Output file not found: {expected_filename}")

                # El output de este paso es el input del siguiente
                current_input_path = expected_filename

                # Limpiar memoria VRAM/RAM entre pasos
                try:
                    del AI_instances
                except:
                    pass
                optimize_memory_usage()

            # Fin del bucle de pasos para este archivo

        # ---------------------------------------------------------------------
        # 4. LIMPIEZA FINAL
        # ---------------------------------------------------------------------

        # Eliminar carpetas temporales de encadenamiento
        for temp_dir in temp_folders_created:
            if os_path_exists(temp_dir):
                try:
                    # Usamos shutil.rmtree para borrar carpeta y contenido
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(
                        f"Warning: Could not remove temp chain dir {temp_dir}: {e}")

        write_process_status(process_status_q, f"{COMPLETED_STATUS}")

    except Exception as exception:
        error_message = str(exception)

        # Enviamos el error a la consola visual
        write_process_status(
            process_status_q, f"[LOG] [ERROR] Technical detail: {error_message}")

        if "cannot convert float NaN to integer" in error_message:
            friendly_msg = "GPU Driver Timeout. Try restarting without keeping frames."
            write_process_status(
                process_status_q, f"{ERROR_STATUS}{friendly_msg}")
        elif "memory" in error_message.lower():
            write_process_status(
                process_status_q, f"{ERROR_STATUS}Insufficient VRAM. Lower 'Tiles resolution' or 'Input %'.")
        else:
            write_process_status(
                process_status_q, f"{ERROR_STATUS}{error_message}")

        # Limpiar temporales en caso de error también
        for temp_dir in temp_folders_created:
            if os_path_exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass

        print(f"[ORCHESTRATOR ERROR] {error_message}")

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
        image_path,
        selected_output_path,
        selected_AI_model,
        input_resize_factor,
        output_resize_factor,
        selected_image_extension,
        selected_blending_factor
    )

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
        image_write(
            upscaled_image_path,
            upscaled_image,
            selected_image_extension
        )

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
        nonlocal writer_threads  # Accedemos a la lista de hilos externa

        # --- CORRECCIÓN: Limpieza de hilos muertos ---
        # Antes de crear uno nuevo, eliminamos de la lista los que ya terminaron
        writer_threads = [t for t in writer_threads if t.is_alive()]
        # ---------------------------------------------

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
        consecutive_memory_errors = 0
        max_memory_errors = 3

        for frame_index in range(len(extracted_frames_paths)):
            frame_path = extracted_frames_paths[frame_index]
            upscaled_frame_path = upscaled_frame_paths[frame_index]
            already_upscaled = os_path_exists(upscaled_frame_path)

            if not already_upscaled:
                start_timer = timer()
                starting_frame = None
                upscaled_frame = None

                try:
                    # Read frame with error handling
                    starting_frame = image_read(frame_path)
                    if starting_frame is None or starting_frame.size == 0:
                        print(
                            f"[WARNING] Invalid frame data at {frame_path}, skipping")
                        continue

                    # Upscale frame with enhanced memory error handling
                    upscaled_frame = AI_instance.AI_orchestration(
                        starting_frame)
                    consecutive_memory_errors = 0  # Reset counter on success

                except Exception as e:
                    error_msg = str(e).lower()

                    # Enhanced GPU memory error detection
                    if any(keyword in error_msg for keyword in ['memory', 'out of memory', 'allocation', 'cuda']):
                        consecutive_memory_errors += 1
                        print(
                            f"[GPU] Memory error #{consecutive_memory_errors} detected: {str(e)[:100]}...")

                        if consecutive_memory_errors >= max_memory_errors:
                            raise RuntimeError(
                                f"Too many consecutive GPU memory errors ({max_memory_errors}). Please reduce VRAM usage or batch size.")

                        # Progressive memory reduction strategy
                        original_tiles = AI_instance.max_resolution
                        # 2, 4, 8...
                        reduction_factor = 2 ** consecutive_memory_errors
                        new_resolution = max(
                            64, original_tiles // reduction_factor)

                        print(
                            f"[GPU] Reducing tiles resolution from {original_tiles} to {new_resolution} and retrying...")
                        AI_instance.max_resolution = new_resolution

                        # Force memory cleanup before retry
                        if starting_frame is not None:
                            del starting_frame
                        optimize_memory_usage()

                        try:
                            starting_frame = image_read(frame_path)
                            upscaled_frame = AI_instance.AI_orchestration(
                                starting_frame)
                            print(
                                f"[GPU] Retry successful with tiles resolution: {new_resolution}")
                            consecutive_memory_errors = 0  # Reset on successful retry
                        except Exception as retry_error:
                            # Restore original resolution if retry also fails
                            AI_instance.max_resolution = original_tiles
                            print(
                                f"[GPU] Retry failed: {str(retry_error)[:100]}...")
                            raise retry_error
                    else:
                        # Non-memory related error
                        logging.error(
                            f"Frame processing error at {frame_path}: {str(e)}")
                        raise e

                # Validate upscaled frame
                if upscaled_frame is None or upscaled_frame.size == 0:
                    print(
                        f"[WARNING] Upscaling produced invalid result for {frame_path}, skipping")
                    continue

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
                    [process_status_q] * threads_number,
                    [file_number] * threads_number,
                    [threads_number] * threads_number,
                    AI_upscale_instance_list,
                    extracted_frame_list_chunks,
                    upscaled_frame_list_chunks,
                    [selected_blending_factor] * threads_number,
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
            process_status_q, file_number, target_directory, AI_upscale_instance_list[0], video_path, cpu_number, ".png")

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

    # --- CORRECCIÓN: Argumento fps_multiplier ---
    # El upscaling no cambia FPS, por lo tanto multiplier es 1
    video_encoding(process_status_q, video_path, video_output_path,
                   upscaled_frame_paths, selected_video_codec, fps_multiplier=1)

    # 7. Delete frames folder
    if not selected_keep_frames:
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
        # Esto llama al método del nuevo FileQueueManager
        selected_file_list = file_widget.get_selected_file_list()
    except Exception:
        info_message.set("Please select a file")
        return False

    if not selected_file_list or len(selected_file_list) <= 0:
        info_message.set("Please select a file")
        return False

    # Enhanced file validation
    try:
        selected_file_list = file_widget.get_selected_file_list()
    except Exception:
        info_message.set("Please select a file")
        return False

    if not selected_file_list or len(selected_file_list) <= 0:
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

    # --- FIX: STRICT NORMALIZATION OF RESIZE FACTORS ---
    try:
        # Obtener valor crudo (ej: "50" o "100")
        raw_input = float(str(selected_input_resize_factor.get()))
        # Convertir estrictamente a factor (ej: 0.5 o 1.0)
        input_resize_factor = raw_input / 100.0

        if input_resize_factor <= 0:
            raise ValueError("Value must be > 0")
    except (ValueError, TypeError):
        info_message.set("Input resolution % must be a valid number > 0")
        return False

    try:
        raw_output = float(str(selected_output_resize_factor.get()))
        output_resize_factor = raw_output / 100.0

        if output_resize_factor <= 0:
            raise ValueError("Value must be > 0")
    except (ValueError, TypeError):
        info_message.set("Output resolution % must be a valid number > 0")
        return False
    # ---------------------------------------------------

    # VRAM limiter
    try:
        vram_gb = float(str(selected_VRAM_limiter.get()))
        if vram_gb <= 0:
            info_message.set("GPU VRAM value must be a value > 0")
            return False

        vram_multiplier = VRAM_model_usage.get(selected_AI_model, 1.0)

        # Cálculo corregido: VRAM (GB) * Multiplicador * 100 (Base tile size)
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


def open_files_action(files=None):
    # Función auxiliar interna
    def check_supported_selected_files(uploaded_file_list: list) -> list:
        return [file for file in uploaded_file_list if any(supported_extension in file for supported_extension in supported_file_extensions)]

    info_message.set("Processing files...")

    if files:
        # Caso A: Drag & Drop
        uploaded_files_list = list(files)
    else:
        # Caso B: Botón manual
        info_message.set("Selecting files")
        uploaded_files_list = list(filedialog.askopenfilenames())

    if not uploaded_files_list:
        return

    # Filtrar archivos
    supported_files_list = check_supported_selected_files(uploaded_files_list)

    if supported_files_list:
        # 1. Configurar los factores actuales en el widget antes de añadir
        upscale_factor, input_resize_factor, output_resize_factor = get_values_for_file_widget()
        file_widget.set_upscale_factor(upscale_factor)
        file_widget.set_input_resize_factor(input_resize_factor)
        file_widget.set_output_resize_factor(output_resize_factor)

        # 2. Añadir archivos (la carga pesada ocurre en segundo plano en el nuevo módulo)
        file_widget.add_files(supported_files_list)

        # 3. Cambiar vista
        show_file_manager()

        info_message.set("Ready to be enchanted!")
        print(f"> Added {len(supported_files_list)} files to queue.")
    else:
        info_message.set("No supported files selected")


def open_output_path_action():
    asked_selected_output_path = filedialog.askdirectory()
    if asked_selected_output_path == "":
        selected_output_path.set(OUTPUT_PATH_CODED)
    else:
        selected_output_path.set(asked_selected_output_path)

# ==== GUI MENU SELECTION SECTION ====


def select_AI_from_menu(selected_option: str) -> None:
    global selected_AI_model
    global selected_blending_factor       # <-- AÑADIR
    global selected_frame_generation_option  # <-- AÑADIR

    selected_AI_model = selected_option
    update_file_widget(1, 2, 3)

    # --- Improved: instant dynamic refresh for conditional FluidFrames menus ---
    clear_dynamic_menus()

    # FluidFrames/RIFE: Show frame generation menu, otherwise show blending
    if selected_AI_model in RIFE_models_list:
        place_frame_generation_menu()
        # Restablecer el factor de blending a OFF cuando RIFE es elegido
        selected_blending_factor = 0
    else:
        # Restablecer la generación de frames a OFF para cualquier modelo que no sea RIFE
        selected_frame_generation_option = "OFF"

        # Face restoration models don't need blending (they work differently)
        if selected_AI_model not in Face_restoration_models_list:
            place_AI_blending_menu()
        # Los modelos de restauración facial (GFPGAN) tampoco usan blending
        elif selected_AI_model in Face_restoration_models_list:
            # Asegurarse de que el blending esté en OFF para GFPGAN
            selected_blending_factor = 0

    # Always restore other key controls
    place_AI_multithreading_menu()
    place_input_output_resolution_textboxs()
    place_gpu_gpuVRAM_menus()
    place_video_codec_keep_frames_menus()
    place_image_video_output_menus()
    place_output_path_textbox()
    place_integrated_console()
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


# Variables globales para manejar las vistas
drop_zone_frame = None
file_widget = None  # Esta será la instancia de FileQueueManager


def show_drop_zone():
    """Oculta la lista de archivos y muestra la zona de carga."""
    if file_widget:
        file_widget.place_forget()
    if drop_zone_frame:
        drop_zone_frame.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)
    info_message.set("No files selected")


def show_file_manager():
    """Oculta la zona de carga y muestra la lista de archivos."""
    if drop_zone_frame:
        drop_zone_frame.place_forget()
    if file_widget:
        file_widget.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)


def place_loadFile_section():
    global drop_zone_frame, file_widget

    # --- 1. Crear el Frame de la Drop Zone (Inicialmente Visible) ---
    drop_zone_frame = CTkFrame(
        master=window, fg_color=background_color, corner_radius=CORNER_RADIUS)

    text_drop = (" SUPPORTED FILES \n\n "
                 + "IMAGES • jpg, jpeg, png, bmp, tiff, tif, webp \n "
                 + "VIDEOS • mp4, avi, mkv, mov, wmv, flv, webm ")

    input_file_text = CTkLabel(
        master=drop_zone_frame,
        text=text_drop,
        fg_color=widget_background_color,
        bg_color=background_color,
        text_color=secondary_text_color,
        width=300,
        height=150,
        font=bold13,
        anchor="center",
        corner_radius=CORNER_RADIUS
    )

    input_file_button = CTkButton(
        master=drop_zone_frame,
        command=open_files_action,  # Llama a la función modificada abajo
        text="Select Files or Drag & Drop",
        width=150,
        height=30,
        font=bold12,
        border_width=1,
        corner_radius=CORNER_RADIUS,
        fg_color=widget_background_color,
        text_color=text_color,
        border_color=accent_color,
        hover_color=button_hover_color
    )

    # Colocar elementos dentro del frame de Drop Zone
    input_file_text.place(relx=0.5, rely=0.4, anchor="center")
    input_file_button.place(relx=0.5, rely=0.5, anchor="center")

    # Mostrar Drop Zone por defecto
    drop_zone_frame.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)

    # --- 2. Instanciar FileQueueManager (Inicialmente Oculto) ---
    # Usamos show_drop_zone como callback para cuando el usuario limpie la lista
    file_widget = FileQueueManager(
        master=window,
        clear_icon=clear_icon,
        on_queue_empty_callback=show_drop_zone,
        width=300,  # <-- Esto es un kwargs
    )

    # Habilitar Drag & Drop en AMBOS componentes (Drop Zone y File Manager)
    # Esto permite arrastrar archivos incluso si ya hay una lista visible
    enable_drag_and_drop(window, [
                         drop_zone_frame, input_file_button, input_file_text, file_widget], open_files_action)


def place_app_name():
    background = CTkFrame(
        master=window, fg_color=background_color, corner_radius=CORNER_RADIUS)
    app_name_label = CTkLabel(
        master=window,
        text=app_name + " " + version,
        fg_color="transparent",
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

            "\n GFPGAN \n"
            "\n • Generative Face Prior GAN for face restoration\n"
            " • Year: 2021\n"
            " • Function: Face restoration and enhancement\n"
            " • Excellent for old/blurry photos\n",

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

# --- CAMBIO: Usamos 'row8' en lugar de 'row10' para subirlo ---
    background.place(relx=0.75,                 rely=row8,
                     relwidth=0.48, anchor="center")
    info_button.place(relx=column_info1,         rely=row8 -
                      0.003,           anchor="center")
    active_button.place(relx=column_info1 + 0.052,
                        rely=row8,                   anchor="center")
    option_menu.place(relx=column_2 - 0.008,     rely=row8,
                      anchor="center")


def place_integrated_console():
    """
    Crea y coloca la consola integrada, desplazada hacia abajo
    para dejar espacio a los botones superiores.
    """

    # 1. Crear el widget
    console_widget = IntegratedConsole(
        master=window,
        fg_color=widget_background_color,
        border_color=accent_color,
        border_width=1,
        corner_radius=5
    )

    # 2. Posicionamiento
    # rely=0.89: Bajamos la consola al fondo.
    # relheight=0.18: Altura ajustada para no salirse de la pantalla.
    console_widget.place(
        relx=0.75,
        rely=0.89,
        relwidth=0.48,
        relheight=0.18,
        anchor="center"
    )

    # 3. Conexión lógica
    console.set_widget(console_widget)

    # Mensaje inicial
    console.write_log(
        f"[{app_name}] System initialized. Console ready.", "SUCCESS")


def place_stop_button():
    stop_button = create_active_button(
        command=stop_button_command,
        text="STOP",
        icon=stop_icon,
        width=200,    # Ajustado a 200 (igual que el nuevo Make Magic)
        height=28,    # Altura delgada
        border_color=error_color
    )
    # Posición: rely=0.77 y relx=0.83 (Misma que el nuevo Make Magic)
    stop_button.place(relx=0.83, rely=0.77, anchor="center")

# En Warlock-Studio.py, busca donde creas los botones principales


def open_chain_manager():
    global chain_window

    # Callback para capturar la configuración actual de la GUI
    def capture_current_settings():
        # Capturamos las variables globales actuales de la GUI
        return {
            'model': selected_AI_model,
            'input_resize': float(selected_input_resize_factor.get()) / 100.0,
            'output_resize': float(selected_output_resize_factor.get()) / 100.0,
            'blending': selected_blending_factor,
            'vram': float(selected_VRAM_limiter.get()),
            'ext_img': selected_image_extension,
            'ext_vid': selected_video_extension,
            'codec': selected_video_codec,
            'frame_gen': selected_frame_generation_option,
            'keep_frames': selected_keep_frames,
            'gpu': selected_gpu
        }

    if chain_window is None or not chain_window.winfo_exists():
        chain_window = ChainManager(window, capture_current_settings)
    else:
        chain_window.lift()

# --- Añadir el botón en la GUI (ejemplo: al lado de settings) ---
# Puedes añadir esto en la función `place_upscale_button` o crear `place_chain_button`


def place_chain_button():
    """
    Coloca el botón Chain a la izquierda del botón Magic, en la misma fila.
    """
    btn_chain = create_active_button(
        command=open_chain_manager,
        text="⛓ Chain",
        width=110,   # Ancho suficiente para el texto
        height=28    # Misma altura que el botón Magic para consistencia
    )
    # POSICIÓN: A la izquierda (0.65)
    btn_chain.place(relx=0.65, rely=0.77, anchor="center")
    btn_chain.lift()


def place_upscale_button():
    """
    Coloca el botón Magic desplazado a la derecha para dar espacio al Chain.
    """
    upscale_button = create_active_button(
        command=upscale_button_command,
        text="Make Magic",
        icon=upscale_icon,
        width=200,   # Reducido ligeramente de 240 a 200 para equilibrar el espacio
        height=28
    )
    # POSICIÓN: Desplazado a la derecha (0.83)
    # Antes estaba en 0.75 (centro absoluto del panel derecho).
    upscale_button.place(relx=0.83, rely=0.77, anchor="center")


# ==== MAIN APPLICATION SECTION ====

def on_app_close() -> None:
    # 1. Confirmación de salida
    if not messagebox.askyesno("Exit Warlock-Studio", "Are you sure you want to close the application?"):
        return

    # 2. CAPTURAR ESTADO DE LA VENTANA (ANTES DE DESTRUIRLA)
    # Soluciona el error: application has been destroyed
    try:
        is_topmost = window.attributes("-topmost")
    except Exception:
        is_topmost = False

    # 3. Recopilar variables globales para guardar preferencias
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

    keep_frames_to_save = "ON" if selected_keep_frames else "OFF"

    if selected_AI_multithreading == 1:
        AI_multithreading_to_save = "OFF"
    else:
        AI_multithreading_to_save = f"{selected_AI_multithreading} threads"

    # 4. Construir diccionario de preferencias
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
        # Usamos la variable capturada al inicio
        "keep_window_on_top":           is_topmost
    }

    # 5. Guardar JSON en disco
    try:
        user_preference_json = json_dumps(user_preference)
        with open(USER_PREFERENCE_PATH, "w") as preference_file:
            preference_file.write(user_preference_json)
    except Exception as e:
        print(f"Error saving preferences: {e}")

    # 6. Limpieza de procesos y logs
    stop_upscale_process()
    logging.shutdown()

    # 7. DESTRUIR LA VENTANA (AL FINAL)
    try:
        window.grab_release()
        window.destroy()
    except Exception:
        pass


class App():
    def __init__(self, window):
        self.toplevel_window = None
        window.protocol("WM_DELETE_WINDOW", on_app_close)

        window.title(f"Warlock-Studio")
        # Get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # --- CAMBIO REALIZADO AQUÍ (Aumentado a 85% para ser más grande) ---
        # Set to 85% of the screen by default, centered
        default_width = int(screen_width * 0.8)
        default_height = int(screen_height * 0.87)
        # ------------------------------------------------------------------

        x_position = (screen_width - default_width) // 2
        y_position = (screen_height - default_height) // 2
        window.geometry(
            f"{default_width}x{default_height}+{x_position}+{y_position}")

        # --> AQUÍ ESTÁ EL CAMBIO
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

        place_integrated_console()
        place_upscale_button()


def log_startup_info():
    """
    Imprime la información de inicio una vez que la consola gráfica está activa.
    """
    # 1. Check FFmpeg
    if os_path_exists(FFMPEG_EXE_PATH):
        print(f"[{app_name}] ffmpeg.exe found")
    else:
        print(
            f"[{app_name}] WARNING: ffmpeg.exe not found. Video functionality will be limited.")

    # 2. Check Preferences
    if os_path_exists(USER_PREFERENCE_PATH):
        print(f"[{app_name}] Preference file exists")
    else:
        print(
            f"[{app_name}] Preference file does not exist, using default coded value")

    # 3. Check ONNX Providers
    try:
        from onnxruntime import get_available_providers
        providers = get_available_providers()
        print("Available ONNX Runtime Providers:")
        for p in providers:
            print(f"- {p}")
    except ImportError as e:
        print(f"Error: The onnxruntime library is not installed. {e}")

# --- CÓDIGO PARA EL BOTÓN DEL MANUAL (Pegar antes de if __name__ == "__main__":) ---


def open_manual_action():
    """Función para abrir el PDF del manual."""
    manual_path = find_by_relative_path(
        f"Assets{os_separator}Warlock-Studio_Manual.pdf")

    if os_path_exists(manual_path):
        try:
            # Abre el PDF con el visor predeterminado del sistema
            if os.name == 'nt':  # Windows
                os.startfile(manual_path)
            else:  # macOS / Linux
                subprocess_run(['open' if sys.platform ==
                               'darwin' else 'xdg-open', manual_path])
            console.write_log("Manual opened successfully", "SUCCESS")
        except Exception as e:
            log_and_report_error(f"Could not open manual: {e}")
    else:
        show_error_message("Manual file not found in Assets folder.")


class ManualButton(CTkButton):
    """Clase para el botón de ayuda/manual."""

    def __init__(self, master, **kwargs):
        # Intenta buscar el icono 'manual_icon' en las variables globales
        # Si no existe, usa texto
        icon_img = globals().get('manual_icon', None)
        text_val = "" if icon_img else "📖 Help"

        super().__init__(master, text=text_val, image=icon_img, width=40, height=28,
                         fg_color=widget_background_color, border_color=border_color, border_width=1,
                         hover_color=button_hover_color, text_color=text_color,
                         command=open_manual_action, **kwargs)

# -----------------------------------------------------------------------------------


if __name__ == "__main__":
    freeze_support()
    from warlock_preferences import ConfigManager, PreferencesButton
    set_appearance_mode("Dark")

    # Configurar tema visual
    import customtkinter
    customtkinter.set_default_color_theme("dark-blue")

    # Aplicar overrides de tema para consistencia visual
    try:
        customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"] = [
            widget_background_color, widget_background_color]
        customtkinter.ThemeManager.theme["CTkButton"]["fg_color"] = [
            widget_background_color, widget_background_color]
        customtkinter.ThemeManager.theme["CTkButton"]["hover_color"] = [
            button_hover_color, button_hover_color]
        customtkinter.ThemeManager.theme["CTkButton"]["text_color"] = [
            text_color, text_color]
        customtkinter.ThemeManager.theme["CTkButton"]["border_color"] = [
            accent_color, accent_color]
        customtkinter.ThemeManager.theme["CTkEntry"]["fg_color"] = [
            widget_background_color, widget_background_color]
        customtkinter.ThemeManager.theme["CTkEntry"]["text_color"] = [
            text_color, text_color]
        customtkinter.ThemeManager.theme["CTkEntry"]["border_color"] = [
            accent_color, accent_color]
        customtkinter.ThemeManager.theme["CTkOptionMenu"]["fg_color"] = [
            widget_background_color, widget_background_color]
        customtkinter.ThemeManager.theme["CTkOptionMenu"]["text_color"] = [
            text_color, text_color]
        customtkinter.ThemeManager.theme["CTkOptionMenu"]["button_hover_color"] = [
            button_hover_color, button_hover_color]
        customtkinter.ThemeManager.theme["CTkLabel"]["text_color"] = [
            text_color, text_color]
    except Exception as e:
        print(f"[THEME] Could not apply custom theme: {e}")

    process_status_q = multiprocessing_Queue(maxsize=1)

    # Inicializar ventana principal (oculta para mostrar el splash primero)
    window = DnDCTk()
    window.withdraw()

    # Imprimir la info de inicio (saldrá en la nueva consola integrada)
    log_startup_info()

    # ------------------------------------------------------------
    # CONFIGURACIÓN DEL SPLASH SCREEN (CORREGIDO)
    # ------------------------------------------------------------

    # 1. Empaquetar colores para el módulo externo
    splash_theme = {
        'bg': background_color,
        'widget_bg': widget_background_color,
        'accent': accent_color,
        'app_name': app_name_color,
        'text_sec': secondary_text_color
    }

    # 2. Instanciar SplashScreen pasando los argumentos requeridos
    # Esto evita el TypeError que estabas teniendo
    splash = SplashScreen(
        root_window=window,       # <--- CAMBIO AQUÍ (antes root_window=window)
        app_title=app_name,
        version=version,
        asset_loader=find_by_relative_path,  # Función para buscar assets
        theme_colors=splash_theme,          # Diccionario de colores
        duration_ms=6000                    # Duración: 6 segundos
    )

    # 3. Programar aparición de la ventana principal
    # Se añade un pequeño retardo extra (6500ms) sobre la duración del splash (6000ms)
    window.after(6500, window.deiconify)

    # ------------------------------------------------------------
    # Inicialización de Variables de UI
    info_message = StringVar()
    selected_output_path = StringVar()
    selected_input_resize_factor = StringVar()
    selected_output_resize_factor = StringVar()
    selected_VRAM_limiter = StringVar()

    # Inicializar variables globales seleccionadas con los defaults cargados
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
                                "Medium": 0.5, "High": 0.7}.get(default_blending, 0)
    selected_frame_generation_option = "OFF"

    # Inicializar variables de control global
    stop_thread_flag = Event()
    global_processing_times_list = []
    global_upscaled_frames_paths = []
    global_can_i_update_status = False
    output_resize_factor = 1.0
    tiles_resolution = 800

    # Asignar valores por defecto a los campos de texto
    selected_input_resize_factor.set(default_input_resize_factor)
    selected_output_resize_factor.set(default_output_resize_factor)
    selected_VRAM_limiter.set(default_VRAM_limiter)
    selected_output_path.set(default_output_path)

    info_message.set("Ready for the show!")
    # Añadir listeners para actualizar widgets cuando cambien los valores
    selected_input_resize_factor.trace_add('write', update_file_widget)
    selected_output_resize_factor.trace_add('write', update_file_widget)

    # Definición de Fuentes e Iconos
    font = "Consola"
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

# Cargar Iconos
    stop_icon = CTkImage(pillow_image_open(find_by_relative_path(
        f"Assets{os_separator}stop_icon.png")), size=(15, 15))
    upscale_icon = CTkImage(pillow_image_open(find_by_relative_path(
        f"Assets{os_separator}upscale_icon.png")), size=(15, 15))
    clear_icon = CTkImage(pillow_image_open(find_by_relative_path(
        f"Assets{os_separator}clear_icon.png")), size=(15, 15))
    info_icon = CTkImage(pillow_image_open(find_by_relative_path(
        f"Assets{os_separator}info_icon.png")), size=(18, 18))

    # --- NUEVO: Carga del icono del manual con seguridad ---
    try:
        manual_icon = CTkImage(pillow_image_open(find_by_relative_path(
            f"Assets{os_separator}manual_icon.png")), size=(20, 20))
    except:
        manual_icon = None  # Fallback si no existe la imagen
    # -------------------------------------------------------

    # Inicializar la Aplicación Principal
    app = App(window)
    window.update()

    # Inicializar Botón de Preferencias (AQUÍ SE CONECTA LA CONSOLA Y CONFIGURACIÓN)
    from warlock_preferences import PreferencesButton
    preferences_btn = PreferencesButton(
        master=window,
        current_version=version,
        repo_owner="Ivan-Ayub97",
        repo_name="Warlock-Studio"
    )
    # Posición en la esquina superior derecha
    preferences_btn.place(relx=0.95, rely=0.05, anchor="center")
    preferences_btn.lift()

    # --- NUEVO: Botón del Manual ---
    # Colocado a la izquierda del engranaje (relx=0.88)
    manual_btn = ManualButton(master=window)
    manual_btn.place(relx=0.88, rely=0.05, anchor="center")
    manual_btn.lift()
    # -------------------------------
    place_chain_button()
    # Iniciar Bucle Principal
    window.mainloop()

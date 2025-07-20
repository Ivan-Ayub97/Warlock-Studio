"""
AI Model Downloader for Warlock-Studio
Descarga automática de modelos AI cuando no están presentes
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from typing import Optional
import tempfile
import shutil
from tkinter import messagebox
import threading
import time

class ModelDownloader:
    def __init__(self):
        self.base_path = self._get_base_path()
        self.ai_models_path = os.path.join(self.base_path, "AI-onnx")
        self.download_url = "https://github.com/Ivan-Ayub97/Warlock-Studio/releases/download/v4.0/AI-onnx-models.zip"
        self.backup_urls = [
            "https://sourceforge.net/projects/warlock-studio/files/AI-Models/AI-onnx-models.zip",
            # Agregar más URLs de respaldo aquí
        ]
    
    def _get_base_path(self) -> str:
        """Obtener la ruta base de la aplicación"""
        if getattr(sys, '_MEIPASS', None):
            return sys._MEIPASS
        return os.path.dirname(os.path.abspath(__file__))
    
    def check_models_exist(self) -> bool:
        """Verificar si los modelos AI existen"""
        if not os.path.exists(self.ai_models_path):
            return False
        
        required_models = [
            "BSRGANx2_fp16.onnx",
            "BSRGANx4_fp16.onnx",
            "GFPGANv1.4.fp16.onnx",
            "IRCNN_Lx1_fp16.onnx",
            "IRCNN_Mx1_fp16.onnx",
            "RIFE_Lite_fp32.onnx",
            "RIFE_fp32.onnx",
            "RealESRGANx4_fp16.onnx",
            "RealESRNetx4_fp16.onnx",
            "RealESR_Animex4_fp16.onnx",
            "RealESR_Gx4_fp16.onnx",
            "RealSRx4_Anime_fp16.onnx",
            "super-resolution-10.onnx"
        ]
        
        for model in required_models:
            model_path = os.path.join(self.ai_models_path, model)
            if not os.path.exists(model_path):
                return False
        
        return True
    
    def download_with_progress(self, url: str, destination: str, progress_callback=None) -> bool:
        """Descargar archivo con barra de progreso"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(progress, downloaded, total_size)
            
            return True
            
        except Exception as e:
            print(f"Error downloading from {url}: {str(e)}")
            return False
    
    def extract_zip(self, zip_path: str, extract_path: str) -> bool:
        """Extraer archivo ZIP"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            return True
        except Exception as e:
            print(f"Error extracting {zip_path}: {str(e)}")
            return False
    
    def download_models(self, progress_callback=None) -> bool:
        """Descargar modelos AI"""
        if self.check_models_exist():
            print("AI models already exist, skipping download")
            return True
        
        print("AI models not found, downloading...")
        
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "AI-onnx-models.zip")
            
            # Intentar descargar desde URL principal
            success = False
            for url in [self.download_url] + self.backup_urls:
                print(f"Attempting download from: {url}")
                if self.download_with_progress(url, zip_path, progress_callback):
                    success = True
                    break
                else:
                    print(f"Failed to download from {url}, trying next URL...")
            
            if not success:
                return False
            
            # Crear directorio de destino
            os.makedirs(self.ai_models_path, exist_ok=True)
            
            # Extraer archivos
            if self.extract_zip(zip_path, self.base_path):
                print("AI models downloaded and extracted successfully")
                return True
            else:
                return False
    
    def download_models_async(self, completion_callback=None, progress_callback=None):
        """Descargar modelos de forma asíncrona"""
        def download_thread():
            success = self.download_models(progress_callback)
            if completion_callback:
                completion_callback(success)
        
        thread = threading.Thread(target=download_thread)
        thread.daemon = True
        thread.start()
        return thread

# Función para integrar con el código principal
def ensure_models_available(show_dialog=True) -> bool:
    """Asegurar que los modelos AI estén disponibles"""
    downloader = ModelDownloader()
    
    if downloader.check_models_exist():
        return True
    
    if show_dialog:
        from tkinter import messagebox
        result = messagebox.askyesno(
            "AI Models Required",
            "AI models are required but not found. Would you like to download them now?\n\n"
            "This will download approximately 327MB of data.\n\n"
            "Click 'Yes' to download or 'No' to continue without AI functionality."
        )
        
        if not result:
            return False
    
    # Mostrar ventana de progreso simple
    try:
        import tkinter as tk
        from tkinter import ttk
        
        progress_window = tk.Toplevel()
        progress_window.title("Downloading AI Models")
        progress_window.geometry("400x120")
        progress_window.resizable(False, False)
        
        progress_label = tk.Label(progress_window, text="Downloading AI models...")
        progress_label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
        progress_bar.pack(pady=10)
        
        status_label = tk.Label(progress_window, text="Starting download...")
        status_label.pack(pady=5)
        
        download_complete = [False]
        
        def update_progress(percentage, downloaded, total):
            progress_bar['value'] = percentage
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            status_label.config(text=f"Downloaded: {mb_downloaded:.1f} MB / {mb_total:.1f} MB")
            progress_window.update()
        
        def on_completion(success):
            download_complete[0] = True
            if success:
                status_label.config(text="Download completed successfully!")
            else:
                status_label.config(text="Download failed!")
            progress_window.after(2000, progress_window.destroy)
        
        # Iniciar descarga
        downloader.download_models_async(on_completion, update_progress)
        
        # Mantener ventana activa hasta completar
        while not download_complete[0]:
            progress_window.update()
            time.sleep(0.1)
        
        return download_complete[0]
        
    except ImportError:
        # Si no hay tkinter, descargar sin interfaz gráfica
        return downloader.download_models()

if __name__ == "__main__":
    downloader = ModelDownloader()
    if not downloader.check_models_exist():
        print("Downloading AI models...")
        success = downloader.download_models()
        if success:
            print("Models downloaded successfully!")
        else:
            print("Failed to download models!")
    else:
        print("AI models already available!")

import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image


# --- CONFIGURACIÓN Y ESTILOS ---
class Theme:
    BG_COLOR = "#101010"           # Fondo general un poco más oscuro
    ITEM_BG = "#3A3A3A"            # Fondo de cada tarjeta
    TEXT_MAIN = "#FFFFFF"
    TEXT_SEC = "#FDEF2F"
    ACCENT = "#F7F7F7"             # Dorado original
    ERROR = "#CF6679"              # Rojo suave para errores
    BTN_HOVER_DANGER = "#B00020"
    BTN_HOVER_NORMAL = "#4A4A4A"
    BORDER_COLOR = "#555555"

    FONT_MAIN = ("Roboto Medium", 13)
    FONT_SUB = ("Roboto", 11)
    FONT_MONO = ("Consolas", 10)


# Configuración de logging para debug
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class QueueItem:
    """Clase de datos para manejar el estado de cada archivo individualmente."""

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.info_text = "Cargando info..."
        self.thumbnail: Optional[ctk.CTkImage] = None
        self.is_loaded = False
        self.has_error = False
        self.id = id(self)  # Identificador único para referenciar widgets


class FileQueueManager(ctk.CTkScrollableFrame):
    def __init__(self, master, clear_icon=None, on_queue_empty_callback=None, **kwargs):
        super().__init__(master, fg_color=Theme.BG_COLOR, **kwargs)

        self.on_queue_empty_callback = on_queue_empty_callback
        self.clear_icon = clear_icon

        # Estado interno
        self.queue_items: List[QueueItem] = []
        # Mapa para acceder a widgets por ID de item
        self._widget_refs: Dict[int, dict] = {}

        # Factores de redimensionado
        self.upscale_factor = 1
        self.input_resize_factor = 0
        self.output_resize_factor = 0

        # Sistema de hilos para no congelar la GUI
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.msg_queue = queue.Queue()

        # Grid layout básico
        self.grid_columnconfigure(0, weight=1)

        # Iniciar loop de verificación de mensajes (para actualizaciones desde hilos)
        self._check_queue()

    def _check_queue(self):
        """Revisa la cola de mensajes para actualizar la UI desde el hilo principal."""
        try:
            while True:
                task_type, item_id, data = self.msg_queue.get_nowait()
                if task_type == "UPDATE_ITEM":
                    self._update_item_ui(item_id, data)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._check_queue)

    def add_files(self, file_paths: List[str]):
        """Agrega archivos y lanza la carga en segundo plano."""
        added_any = False
        existing_paths = {item.path for item in self.queue_items}

        for path in file_paths:
            norm_path = os.path.normpath(path)
            if norm_path not in existing_paths and os.path.exists(path):
                new_item = QueueItem(norm_path)
                self.queue_items.append(new_item)
                added_any = True

                # Renderizar placeholder inmediatamente
                self._render_item(new_item, index=len(self.queue_items)-1)

                # Lanzar tarea pesada en background
                self.executor.submit(self._worker_load_info, new_item)

        if added_any:
            self._update_layout_indices()

    def _worker_load_info(self, item: QueueItem):
        """Método que corre en otro hilo (Thread) para procesar OpenCV."""
        try:
            info_text, pil_img = self._extract_file_data(item.path)

            # Crear CTkImage debe hacerse preferiblemente al volver al main,
            # pero PIL Image es seguro pasarlo.
            item.info_text = info_text
            item.is_loaded = True

            # Enviamos resultados al hilo principal
            self.msg_queue.put(
                ("UPDATE_ITEM", item.id, {"info": info_text, "img": pil_img}))

        except Exception as e:
            logging.error(f"Error procesando {item.path}: {e}")
            item.has_error = True
            item.info_text = "Error al leer archivo"
            self.msg_queue.put(("UPDATE_ITEM", item.id, {"error": True}))

    def _update_item_ui(self, item_id, data):
        """Actualiza un widget específico una vez que el hilo terminó."""
        widgets = self._widget_refs.get(item_id)
        if not widgets:
            return

        if "error" in data:
            widgets['lbl_info'].configure(
                text="⚠️ Archivo corrupto o ilegible", text_color=Theme.ERROR)
            return

        # Actualizar texto
        widgets['lbl_info'].configure(text=data["info"])

        # Actualizar imagen si existe
        if data["img"]:
            ctk_img = ctk.CTkImage(data["img"], size=data["img"].size)
            # Guardamos referencia en el objeto item para que no se pierda
            item = next((x for x in self.queue_items if x.id == item_id), None)
            if item:
                item.thumbnail = ctk_img
                widgets['lbl_icon'].configure(
                    image=ctk_img, text="")  # Quitar texto placeholder

    def _render_item(self, item: QueueItem, index: int):
        """Dibuja una tarjeta para el archivo."""
        row_idx = index + 1  # +1 para dejar espacio al botón Clean arriba si fuera necesario

        # Frame contenedor (Tarjeta)
        card = ctk.CTkFrame(self, fg_color=Theme.ITEM_BG, corner_radius=8,
                            border_width=1, border_color=Theme.BORDER_COLOR)
        card.grid(row=row_idx, column=0, sticky="ew", padx=10, pady=5)
        card.grid_columnconfigure(1, weight=1)

        # 1. Icono / Thumbnail
        lbl_icon = ctk.CTkLabel(card, text="⏳", width=64,
                                height=64, fg_color="#222", corner_radius=6)
        lbl_icon.grid(row=0, column=0, rowspan=2, padx=8, pady=8)

        # Si ya teníamos la imagen (ej: reordenando lista), la ponemos directo
        if item.thumbnail:
            lbl_icon.configure(image=item.thumbnail, text="")

        # 2. Información
        lbl_name = ctk.CTkLabel(
            card, text=item.name, font=Theme.FONT_MAIN, text_color=Theme.ACCENT, anchor="w")
        lbl_name.grid(row=0, column=1, sticky="w", padx=5, pady=(8, 0))

        lbl_info = ctk.CTkLabel(card, text=item.info_text, font=Theme.FONT_MONO,
                                text_color=Theme.TEXT_SEC, justify="left", anchor="w")
        lbl_info.grid(row=1, column=1, sticky="w", padx=5, pady=(0, 8))

        # 3. Controles (Frame derecho)
        ctrl_frame = ctk.CTkFrame(card, fg_color="transparent")
        ctrl_frame.grid(row=0, column=2, rowspan=2, padx=8, sticky="e")

        # Botón Subir
        btn_up = ctk.CTkButton(ctrl_frame, text="▲", width=28, height=28,
                               fg_color="transparent", border_width=1, border_color=Theme.TEXT_SEC,
                               hover_color=Theme.BTN_HOVER_NORMAL, text_color=Theme.TEXT_MAIN,
                               font=("Arial", 12),
                               command=lambda: self.move_up(item))
        btn_up.pack(side="left", padx=2)

        # Botón Eliminar
        btn_del = ctk.CTkButton(ctrl_frame, text="✕", width=28, height=28,
                                fg_color="transparent", border_width=1, border_color=Theme.ERROR,
                                text_color=Theme.ERROR, hover_color=Theme.BTN_HOVER_DANGER,
                                font=("Arial", 12, "bold"),
                                command=lambda: self.remove_item(item))
        btn_del.pack(side="left", padx=2)

        # Guardar referencias para actualizaciones asíncronas
        self._widget_refs[item.id] = {
            "card": card,
            "lbl_info": lbl_info,
            "lbl_icon": lbl_icon,
            "btn_up": btn_up  # Guardamos por si queremos deshabilitar el primero
        }

        # Renderizar encabezado si es el primer elemento
        if index == 0:
            self._render_header()

    def _render_header(self):
        """Dibuja el botón de limpiar todo si hay elementos."""
        # Limpiar header anterior si existe
        for w in self.grid_slaves(row=0):
            w.destroy()

        if self.queue_items:
            header_frame = ctk.CTkFrame(self, fg_color="transparent")
            header_frame.grid(row=0, column=0, sticky="ew",
                              padx=10, pady=(5, 0))

            ctk.CTkLabel(header_frame, text=f"Queue: {len(self.queue_items)} Files",
                         font=Theme.FONT_SUB, text_color=Theme.TEXT_SEC).pack(side="left")

            ctk.CTkButton(header_frame, text="CLEAN ALL", image=self.clear_icon,
                          font=Theme.FONT_SUB, height=24, width=100,
                          fg_color="transparent", border_width=1, border_color=Theme.ACCENT, text_color=Theme.ACCENT,
                          hover_color=Theme.BTN_HOVER_NORMAL,
                          command=self.clean_file_list).pack(side="right")

    def _update_layout_indices(self):
        """Re-dibuja toda la lista. Útil para reordenar o borrar."""
        # Nota: En una app muy grande esto sería ineficiente, pero para <100 archivos está bien
        # destruir y recrear widgets garantiza orden visual correcto.

        # Limpiar UI actual
        for widget in self.winfo_children():
            widget.destroy()

        self._widget_refs.clear()

        # Redibujar
        if self.queue_items:
            self._render_header()
            for i, item in enumerate(self.queue_items):
                self._render_item(item, i)
        else:
            # Lista vacía
            if self.on_queue_empty_callback:
                self.on_queue_empty_callback()

    # --- LÓGICA DE DATOS ---

    def move_up(self, item: QueueItem):
        try:
            idx = self.queue_items.index(item)
            if idx > 0:
                self.queue_items[idx], self.queue_items[idx -
                                                        1] = self.queue_items[idx - 1], self.queue_items[idx]
                self._update_layout_indices()
        except ValueError:
            pass

    def remove_item(self, item: QueueItem):
        if item in self.queue_items:
            self.queue_items.remove(item)
            # Eliminar referencias para liberar memoria
            if item.id in self._widget_refs:
                del self._widget_refs[item.id]
            self._update_layout_indices()

    def clean_file_list(self):
        self.queue_items.clear()
        self._widget_refs.clear()
        self._update_layout_indices()

    def get_selected_file_list(self) -> List[str]:
        return [item.path for item in self.queue_items if not item.has_error]

    def regenerate_all_info(self):
        """Recalcula el texto de redimensionado sin recargar imágenes."""
        for item in self.queue_items:
            if not item.has_error and item.is_loaded:
                # Recalculamos solo el texto basado en lo que ya sabemos (o recargamos dims rápidas)
                # Para optimizar, asumimos que podemos volver a lanzar el worker o
                # simplemente recalcular si guardáramos width/height en el objeto.
                # Por simplicidad y robustez, relanzamos el worker ligero.
                self.executor.submit(self._worker_load_info, item)

    # --- SETTERS ---
    def set_upscale_factor(self, f):
        self.upscale_factor = f

    def set_input_resize_factor(self, f):
        self.input_resize_factor = f

    def set_output_resize_factor(self, f):
        self.output_resize_factor = f

    # --- UTILIDADES ESTÁTICAS (Lógica pura) ---

    def _extract_file_data(self, file_path) -> Tuple[str, Optional[Image.Image]]:
        """Extrae info y genera thumbnail PIL (Seguro para Threads)."""
        info_str = "Info no disponible"
        pil_image = None

        try:
            exts_video = ['.mp4', '.avi', '.mkv',
                          '.mov', '.wmv', '.flv', '.webm']
            is_video = any(file_path.lower().endswith(ext)
                           for ext in exts_video)

            width, height = 0, 0

            if is_video:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    # Leer frame central para thumbnail
                    try:
                        if frames > 100:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = self._process_thumbnail(frame)
                    except:
                        pass  # Fallo silencioso solo en thumbnail

                    dur = frames/fps if fps > 0 else 0
                    m, s = divmod(int(dur), 60)
                    info_str = f"Video: {m}m:{s:02d}s • {frames} frames • {width}x{height}\n"
                cap.release()
            else:
                # Imagen
                # cv2.imdecode es mejor para rutas con caracteres especiales en Windows
                img_array = np.fromfile(file_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    height, width = img.shape[:2]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_image = self._process_thumbnail(img)
                    info_str = f"Image: {width}x{height}\n"

            # Añadir datos de redimensionado calculado
            if width > 0 and height > 0:
                info_str += self._calculate_resize_text(width, height)

        except Exception as e:
            logging.error(f"Error interno extract: {e}")
            raise e

        return info_str, pil_image

    def _process_thumbnail(self, img_array) -> Image.Image:
        """Redimensiona y recorta imagen para thumbnail."""
        h, w = img_array.shape[:2]
        target_size = 64

        # Mantener aspect ratio dentro de 64x64
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img_array, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized)

    def _calculate_resize_text(self, w, h) -> str:
        """Lógica pura de cálculo de dimensiones."""
        if self.upscale_factor <= 0:
            return ""

        iw = int(w * (self.input_resize_factor/100)
                 ) if self.input_resize_factor else w
        ih = int(h * (self.input_resize_factor/100)
                 ) if self.input_resize_factor else h

        uw, uh = int(iw * self.upscale_factor), int(ih * self.upscale_factor)

        ow = int(uw * (self.output_resize_factor/100)
                 ) if self.output_resize_factor else uw
        oh = int(uh * (self.output_resize_factor/100)
                 ) if self.output_resize_factor else uh

        txt = ""
        if self.input_resize_factor:
            txt += f"Input AI ({int(self.input_resize_factor)}%) ➜ {iw}x{ih}\n"

        txt += f"Output AI (x{self.upscale_factor}) ➜ {uw}x{uh}"

        if self.output_resize_factor:
            txt += f"\nFinal Output ({int(self.output_resize_factor)}%) ➜ {ow}x{oh}"

        return txt

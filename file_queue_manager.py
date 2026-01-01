import logging
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image


# --- CONFIGURACIÓN Y ESTILOS ---
class Theme:
    BG_COLOR = "#101010"           # Fondo general
    ITEM_BG = "#3A3A3A"            # Fondo de la tarjeta (sin procesar)

    # Nuevo: Colores para la barra de progreso y estados
    PROGRESS_BG = "#2E7D32"        # Verde oscuro para el fondo de progreso (permite leer texto blanco)
    COMPLETED_BG = "#1B5E20"       # Verde más oscuro para completado
    PROCESSING_BORDER = "#F7F7F7"  # Borde blanco/dorado cuando está procesando

    TEXT_MAIN = "#FFFFFF"
    TEXT_SEC = "#FDEF2F"           # Amarillo (info secundaria)
    TEXT_DONE = "#AAAAAA"          # Gris para archivos completados

    ACCENT = "#F7F7F7"             # Dorado/Blanco principal
    ERROR = "#CF6679"              # Rojo error

    BTN_HOVER_DANGER = "#B00020"
    BTN_HOVER_NORMAL = "#4A4A4A"
    BORDER_COLOR = "#555555"

    FONT_MAIN = ("Roboto Medium", 13)
    FONT_SUB = ("Roboto", 11)
    FONT_MONO = ("Consolas", 10)


# Configuración de logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class QueueItem:
    """Clase de datos para manejar el estado de cada archivo individualmente."""

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.info_text = "Cargando info..."
        self.thumbnail: Optional[ctk.CTkImage] = None

        # Estado
        self.is_loaded = False
        self.has_error = False
        self.is_completed = False
        self.progress = 0.0  # 0.0 a 1.0

        self.id = id(self)  # Identificador único


class FileQueueManager(ctk.CTkScrollableFrame):
    def __init__(self, master, clear_icon=None, on_queue_empty_callback=None, **kwargs):
        super().__init__(master, fg_color=Theme.BG_COLOR, **kwargs)

        self.on_queue_empty_callback = on_queue_empty_callback
        self.clear_icon = clear_icon

        # Estado interno
        self.queue_items: List[QueueItem] = []
        # Mapa para acceder a widgets por ID de item o por PATH
        self._widget_refs: Dict[int, dict] = {}
        self._path_to_id: Dict[str, int] = {}

        # Factores de redimensionado
        self.upscale_factor = 1
        self.input_resize_factor = 0
        self.output_resize_factor = 0

        # Sistema de hilos
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.msg_queue = queue.Queue()

        # Grid layout básico
        self.grid_columnconfigure(0, weight=1)

        # Iniciar loop de verificación
        self._check_queue()

    def _check_queue(self):
        """Revisa la cola de mensajes para actualizar la UI desde hilos secundarios."""
        try:
            while True:
                task_type, item_id, data = self.msg_queue.get_nowait()
                if task_type == "UPDATE_ITEM_DATA":
                    self._update_item_ui_data(item_id, data)
                elif task_type == "UPDATE_PROGRESS":
                    self._render_progress_update(item_id, data)
        except queue.Empty:
            pass
        finally:
            self.after(50, self._check_queue)  # Revisar cada 50ms para fluidez

    def add_files(self, file_paths: List[str]):
        """Agrega archivos y lanza la carga en segundo plano."""
        added_any = False
        existing_paths = {item.path for item in self.queue_items}

        for path in file_paths:
            norm_path = os.path.normpath(path)
            if norm_path not in existing_paths and os.path.exists(path):
                new_item = QueueItem(norm_path)
                self.queue_items.append(new_item)

                # Mapeo rápido Path -> ID
                self._path_to_id[norm_path] = new_item.id
                added_any = True

                # Renderizar placeholder inmediatamente
                self._render_item(new_item, index=len(self.queue_items)-1)

                # Lanzar tarea pesada
                self.executor.submit(self._worker_load_info, new_item)

        if added_any:
            self._update_layout_indices()

    def _worker_load_info(self, item: QueueItem):
        """Worker en segundo plano para leer metadatos e imágenes."""
        try:
            info_text, pil_img = self._extract_file_data(item.path)

            # Pre-calcular CTkImage en el hilo principal enviando el PIL
            self.msg_queue.put(("UPDATE_ITEM_DATA", item.id, {
                "info": info_text,
                "img_pil": pil_img,
                "is_loaded": True
            }))
        except Exception as e:
            logging.error(f"Error procesando {item.path}: {e}")
            self.msg_queue.put(("UPDATE_ITEM_DATA", item.id, {"error": True}))

    def _update_item_ui_data(self, item_id, data):
        """Actualiza metadatos (texto/icono) tras la carga."""
        widgets = self._widget_refs.get(item_id)
        item = next((x for x in self.queue_items if x.id == item_id), None)

        if not widgets or not item:
            return

        if "error" in data:
            item.has_error = True
            widgets['lbl_info'].configure(text="⚠️ Error de lectura", text_color=Theme.ERROR)
            widgets['card'].configure(border_color=Theme.ERROR)
            return

        # Actualizar datos del objeto
        if "info" in data:
            item.info_text = data["info"]
            widgets['lbl_info'].configure(text=data["info"])

        if "img_pil" in data and data["img_pil"]:
            # Crear CTkImage en el hilo principal (obligatorio por Tkinter)
            ctk_img = ctk.CTkImage(data["img_pil"], size=data["img_pil"].size)
            item.thumbnail = ctk_img
            widgets['lbl_icon'].configure(image=ctk_img, text="")

        item.is_loaded = True

    # --- RENDERIZADO VISUAL ---

    def _render_item(self, item: QueueItem, index: int):
        """Dibuja la tarjeta del archivo con soporte para barra de progreso de fondo."""
        row_idx = index + 1

        # 1. Frame contenedor (Tarjeta)
        card = ctk.CTkFrame(self, fg_color=Theme.ITEM_BG, corner_radius=8,
                            border_width=1, border_color=Theme.BORDER_COLOR)
        card.grid(row=row_idx, column=0, sticky="ew", padx=10, pady=5)
        card.grid_columnconfigure(1, weight=1)

        # 2. Barra de Progreso (Fondo dinámico)
        # Se coloca con .place() para estar "detrás" del contenido del grid,
        # pero para que funcione, los widgets encima deben tener fg_color="transparent".
        progress_bar = ctk.CTkFrame(card, fg_color=Theme.PROGRESS_BG, corner_radius=8, height=0)
        # Inicialmente ancho 0. relheight=1 ocupa toda la altura de la tarjeta.
        progress_bar.place(relx=0, rely=0, relheight=1, relwidth=item.progress)

        # 3. Contenido (Grid)

        # Icono / Thumbnail
        lbl_icon = ctk.CTkLabel(card, text="⏳", width=64, height=64,
                                fg_color="#222", corner_radius=6)
        lbl_icon.grid(row=0, column=0, rowspan=2, padx=8, pady=8)

        if item.thumbnail:
            lbl_icon.configure(image=item.thumbnail, text="")
        elif item.is_completed:
            lbl_icon.configure(text="✓")

        # Nombre
        lbl_name = ctk.CTkLabel(card, text=item.name, font=Theme.FONT_MAIN,
                                text_color=Theme.TEXT_MAIN, anchor="w", fg_color="transparent")
        lbl_name.grid(row=0, column=1, sticky="w", padx=5, pady=(8, 0))

        # Info técnica
        lbl_info = ctk.CTkLabel(card, text=item.info_text, font=Theme.FONT_MONO,
                                text_color=Theme.TEXT_SEC, justify="left", anchor="w", fg_color="transparent")
        lbl_info.grid(row=1, column=1, sticky="w", padx=5, pady=(0, 8))

        # 4. Controles (Botones)
        ctrl_frame = ctk.CTkFrame(card, fg_color="transparent")
        ctrl_frame.grid(row=0, column=2, rowspan=2, padx=8, sticky="e")

        # Botón Subir
        btn_up = ctk.CTkButton(ctrl_frame, text="▲", width=28, height=28,
                               fg_color="transparent", border_width=1, border_color=Theme.TEXT_SEC,
                               hover_color=Theme.BTN_HOVER_NORMAL, text_color=Theme.TEXT_MAIN,
                               command=lambda: self.move_up(item))
        btn_up.pack(side="left", padx=2)

        # Botón Eliminar
        btn_del = ctk.CTkButton(ctrl_frame, text="✕", width=28, height=28,
                                fg_color="transparent", border_width=1, border_color=Theme.ERROR,
                                text_color=Theme.ERROR, hover_color=Theme.BTN_HOVER_DANGER,
                                font=("Arial", 12, "bold"),
                                command=lambda: self.remove_item(item))
        btn_del.pack(side="left", padx=2)

        # 5. Guardar referencias
        self._widget_refs[item.id] = {
            "card": card,
            "progress_bar": progress_bar,
            "lbl_info": lbl_info,
            "lbl_name": lbl_name,
            "lbl_icon": lbl_icon,
            "btn_del": btn_del,
            "btn_up": btn_up
        }

        # Aplicar estado visual si ya estaba completado (útil al redibujar)
        if item.is_completed:
            self._apply_completed_style(item.id)

        # Renderizar encabezado si es el primero
        if index == 0:
            self._render_header()

    def _render_header(self):
        """Dibuja el botón de limpiar todo."""
        for w in self.grid_slaves(row=0):
            w.destroy()

        if self.queue_items:
            header_frame = ctk.CTkFrame(self, fg_color="transparent")
            header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 0))

            ctk.CTkLabel(header_frame, text=f"Queue: {len(self.queue_items)} Files",
                         font=Theme.FONT_SUB, text_color=Theme.TEXT_SEC).pack(side="left")

            ctk.CTkButton(header_frame, text="CLEAN ALL", image=self.clear_icon,
                          font=Theme.FONT_SUB, height=24, width=100,
                          fg_color="transparent", border_width=1, border_color=Theme.ACCENT, text_color=Theme.ACCENT,
                          hover_color=Theme.BTN_HOVER_NORMAL,
                          command=self.clean_file_list).pack(side="right")

    def _update_layout_indices(self):
        """Redibuja toda la lista para mantener orden visual."""
        for widget in self.winfo_children():
            widget.destroy()

        self._widget_refs.clear()

        if self.queue_items:
            self._render_header()
            for i, item in enumerate(self.queue_items):
                self._render_item(item, i)
        else:
            if self.on_queue_empty_callback:
                self.on_queue_empty_callback()

    # --- API PÚBLICA PARA ACTUALIZACIÓN DE PROGRESO (Desde Warlock-Studio) ---

    def update_file_progress_by_path(self, file_path: str, progress: float):
        """
        Actualiza la barra de progreso de un archivo específico.
        :param file_path: Ruta del archivo.
        :param progress: Float de 0.0 a 100.0 (o 0.0 a 1.0).
        """
        # Normalizar path
        norm_path = os.path.normpath(file_path)
        item_id = self._path_to_id.get(norm_path)

        if item_id:
            # Normalizar a rango 0.0 - 1.0
            if progress > 1.0:
                progress = progress / 100.0

            # Enviar a la cola del hilo principal
            self.msg_queue.put(("UPDATE_PROGRESS", item_id, {"progress": progress}))

    def mark_file_processing(self, file_path: str):
        """Marca visualmente que el archivo se está procesando actualmente."""
        norm_path = os.path.normpath(file_path)
        item_id = self._path_to_id.get(norm_path)
        if item_id:
            self.msg_queue.put(("UPDATE_PROGRESS", item_id, {"status": "processing"}))

    def mark_file_completed(self, file_path: str):
        """Marca el archivo como completado (barra llena, checkmark)."""
        norm_path = os.path.normpath(file_path)
        item_id = self._path_to_id.get(norm_path)
        if item_id:
            self.msg_queue.put(("UPDATE_PROGRESS", item_id, {"status": "completed"}))

    def mark_file_error(self, file_path: str):
        """Marca el archivo con error."""
        norm_path = os.path.normpath(file_path)
        item_id = self._path_to_id.get(norm_path)
        if item_id:
            self.msg_queue.put(("UPDATE_PROGRESS", item_id, {"status": "error"}))

    # --- LÓGICA INTERNA DE ACTUALIZACIÓN VISUAL ---

    def _render_progress_update(self, item_id, data):
        """Ejecutado por _check_queue en el hilo principal."""
        widgets = self._widget_refs.get(item_id)
        item = next((x for x in self.queue_items if x.id == item_id), None)

        if not widgets or not item:
            return

        # Actualizar barra de progreso
        if "progress" in data:
            prog = max(0.0, min(1.0, data["progress"]))
            item.progress = prog
            # Actualizamos ancho relativo (0 a 1)
            widgets['progress_bar'].place(relwidth=prog)

        # Actualizar Estados
        if "status" in data:
            status = data["status"]

            if status == "processing":
                widgets['card'].configure(border_color=Theme.PROCESSING_BORDER, border_width=2)
                # Asegurar que la barra sea visible
                widgets['progress_bar'].place(relwidth=max(0.05, item.progress))

            elif status == "completed":
                item.progress = 1.0
                item.is_completed = True
                self._apply_completed_style(item_id)

            elif status == "error":
                item.has_error = True
                widgets['card'].configure(border_color=Theme.ERROR)
                widgets['lbl_info'].configure(text_color=Theme.ERROR)

    def _apply_completed_style(self, item_id):
        """Aplica el estilo 'Completado' a los widgets."""
        widgets = self._widget_refs.get(item_id)
        if not widgets:
            return

        # Barra llena en verde oscuro
        widgets['progress_bar'].configure(fg_color=Theme.COMPLETED_BG)
        widgets['progress_bar'].place(relwidth=1.0)

        # Borde normal
        widgets['card'].configure(border_color=Theme.COMPLETED_BG, border_width=1)

        # Icono Check
        widgets['lbl_icon'].configure(image=None, text="✓", font=("Arial", 24, "bold"), text_color="#FFF")

        # Texto atenuado
        widgets['lbl_name'].configure(text_color=Theme.TEXT_DONE)

        # Deshabilitar controles
        widgets['btn_up'].configure(state="disabled")

    # --- GESTIÓN DE LISTA ---

    def move_up(self, item: QueueItem):
        try:
            idx = self.queue_items.index(item)
            if idx > 0:
                self.queue_items[idx], self.queue_items[idx - 1] = \
                    self.queue_items[idx - 1], self.queue_items[idx]
                self._update_layout_indices()
        except ValueError:
            pass

    def remove_item(self, item: QueueItem):
        if item in self.queue_items:
            self.queue_items.remove(item)
            if item.id in self._widget_refs:
                del self._widget_refs[item.id]

            # Limpiar mapa de paths
            if item.path in self._path_to_id:
                del self._path_to_id[item.path]

            self._update_layout_indices()

    def clean_file_list(self):
        self.queue_items.clear()
        self._widget_refs.clear()
        self._path_to_id.clear()
        self._update_layout_indices()

    def get_selected_file_list(self) -> List[str]:
        return [item.path for item in self.queue_items if not item.has_error]

    def regenerate_all_info(self):
        """Recalcula info para actualizaciones masivas de configuración."""
        for item in self.queue_items:
            if not item.has_error and item.is_loaded and not item.is_completed:
                self.executor.submit(self._worker_load_info, item)

    # --- SETTERS ---
    def set_upscale_factor(self, f):
        self.upscale_factor = f

    def set_input_resize_factor(self, f):
        self.input_resize_factor = f

    def set_output_resize_factor(self, f):
        self.output_resize_factor = f

    # --- UTILIDADES DE IMAGEN/VIDEO ---

    def _extract_file_data(self, file_path) -> Tuple[str, Optional[Image.Image]]:
        """Extrae info y thumbnail. Seguro para threads."""
        info_str = "Info no disponible"
        pil_image = None

        try:
            exts_video = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
            is_video = any(file_path.lower().endswith(ext) for ext in exts_video)

            width, height = 0, 0

            if is_video:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    # Thumbnail del frame central
                    try:
                        if frames > 100:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frames//2, 50))
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = self._process_thumbnail(frame)
                    except:
                        pass

                    dur = frames/fps if fps > 0 else 0
                    m, s = divmod(int(dur), 60)
                    info_str = f"Video: {m}m:{s:02d}s • {frames} frames • {width}x{height}\n"
                cap.release()
            else:
                img_array = np.fromfile(file_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    height, width = img.shape[:2]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_image = self._process_thumbnail(img)
                    info_str = f"Image: {width}x{height}\n"

            if width > 0 and height > 0:
                info_str += self._calculate_resize_text(width, height)

        except Exception as e:
            logging.error(f"Error interno extract: {e}")
            raise e

        return info_str, pil_image

    def _process_thumbnail(self, img_array) -> Image.Image:
        h, w = img_array.shape[:2]
        target_size = 64
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized)

    def _calculate_resize_text(self, w, h) -> str:
        if self.upscale_factor <= 0:
            return ""

        iw = int(w * (self.input_resize_factor/100)) if self.input_resize_factor else w
        ih = int(h * (self.input_resize_factor/100)) if self.input_resize_factor else h

        uw, uh = int(iw * self.upscale_factor), int(ih * self.upscale_factor)

        ow = int(uw * (self.output_resize_factor/100)) if self.output_resize_factor else uw
        oh = int(uh * (self.output_resize_factor/100)) if self.output_resize_factor else uh

        txt = ""
        if self.input_resize_factor:
            txt += f"In: {int(self.input_resize_factor)}% ➜ {iw}x{ih}\n"

        txt += f"AI x{self.upscale_factor} ➜ {uw}x{uh}"

        if self.output_resize_factor:
            txt += f"\nOut: {int(self.output_resize_factor)}% ➜ {ow}x{oh}"

        return txt

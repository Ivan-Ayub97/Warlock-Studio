import copy
import json
import logging
import os
import sys
import uuid
import tkinter as tk
from dataclasses import asdict, dataclass, field
from tkinter import filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

# -----------------------------------------------------------------------------
# CONFIGURACIÓN Y LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("ChainManager")

# Constantes Globales
GPU_OPTIONS = ["Auto", "GPU 1", "GPU 2", "GPU 3", "GPU 4"]
FRAME_GEN_OPTIONS = ["OFF", "x2", "x4", "x8",
                     "Slowmotion x2", "Slowmotion x4", "Slowmotion x8"]
# Lista combinada de códecs
VIDEO_CODECS = ["x264", "x265", "h264_nvenc", "hevc_nvenc",
                "h264_amf", "hevc_amf", "h264_qsv", "hevc_qsv"]
IMG_EXTENSIONS = [".png", ".jpg", ".bmp", ".tiff", ".webp"]
VID_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".webm"]

# Colores del Tema (Coincidentes con Warlock Studio)
COLOR_BG = "#000000"
COLOR_WIDGET = "#1A1A1A"
COLOR_ACCENT = "#FFC107"
COLOR_TEXT = "#F5F5F5"
COLOR_TEXT_SEC = "#9E9E9E"
COLOR_BTN_HOVER = "#C62828"
COLOR_BORDER = "#2D2D2D"
COLOR_SUCCESS = "#00C853"
COLOR_ERROR = "#B71C1C"

# -----------------------------------------------------------------------------
# 0. HELPERS
# -----------------------------------------------------------------------------

def _resolve_asset_path(filename: str) -> str:
    """Busca robustamente el archivo en la carpeta Assets (Soporte PyInstaller)."""
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base_path, "Assets", filename),
            os.path.join(base_path, "..", "Assets", filename),
            os.path.join(os.getcwd(), "Assets", filename)
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""
    except Exception as e:
        logger.warning(f"Error resolving asset path: {e}")
        return ""

def _resolve_folder_path(foldername: str) -> str:
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base_path, foldername),
            os.path.join(base_path, "..", foldername),
            os.path.join(os.getcwd(), foldername)
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return ""
    except Exception as e:
        logger.warning(f"Error resolving folder path: {e}")
        return ""

def list_available_models() -> List[str]:
    ai_dir = _resolve_folder_path("AI-onnx")
    if not ai_dir:
        return []
    candidates = []
    try:
        for fname in os.listdir(ai_dir):
            if not fname.lower().endswith(".onnx"):
                continue
            name = fname[:-5]  # remove .onnx
            # remove common precision suffixes
            for suf in ["_fp16", "_fp32", ".fp16", ".fp32"]:
                if name.endswith(suf):
                    name = name[: -len(suf)]
            # map GFPGAN variants to base
            if name.startswith("GFPGAN"):
                name = "GFPGAN"
            candidates.append(name)
        # Make unique while preserving order
        seen = set()
        result = []
        for n in candidates:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result
    except Exception as e:
        logger.warning(f"Error listing models: {e}")
        return []

# -----------------------------------------------------------------------------
# 1. DATA MODEL (MODELO DE DATOS)
# -----------------------------------------------------------------------------

@dataclass
class ProcessingStep:
    """
    Representa un nodo de procesamiento en la cadena.
    """
    model_name: str
    input_resize: float
    output_resize: float
    blending: float
    vram_limit: float
    extension: str
    video_codec: Optional[str] = None
    frame_gen: str = "OFF"
    keep_frames: bool = False
    gpu: str = "Auto"

    # Metadatos internos
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enabled: bool = True
    expanded: bool = False

    # Versionado para migraciones futuras
    version: int = 1

    @property
    def is_video_operation(self) -> bool:
        """Determina si el paso es intrínsecamente de video."""
        is_rife = "RIFE" in self.model_name.upper()
        has_interpolation = self.frame_gen != "OFF"
        is_vid_ext = self.extension.lower() in VID_EXTENSIONS
        return is_rife or has_interpolation or is_vid_ext

    @property
    def estimated_scale_factor(self) -> float:
        """Calcula el factor de escala total del paso (In * AI_Factor * Out)."""
        # Estimación básica del modelo
        model_factor = 1.0
        name = self.model_name.upper()
        if "X2" in name: model_factor = 2.0
        elif "X4" in name: model_factor = 4.0
        elif "X8" in name: model_factor = 8.0

        return self.input_resize * model_factor * self.output_resize

    def validate(self) -> List[str]:
        """Devuelve una lista de advertencias si la configuración es sospechosa."""
        warnings = []
        if self.is_video_operation and self.extension not in VID_EXTENSIONS:
            warnings.append(f"Video operation '{self.model_name}' has image extension '{self.extension}'.")
        if not self.is_video_operation and self.extension in VID_EXTENSIONS:
            # Esto podría ser válido (crear video de imagen), pero es raro como paso intermedio
            warnings.append(f"Image operation exporting directly to video container.")
        if self.input_resize <= 0 or self.output_resize <= 0:
            warnings.append("Resize factors must be > 0.")
        return warnings

    def get_summary(self) -> str:
        icon = "🎬" if self.is_video_operation else "🖼️"
        state = "" if self.enabled else "(BYPASS)"

        details = []
        if self.input_resize != 1.0: details.append(f"In:{int(self.input_resize*100)}%")

        # Detectar escala del modelo
        scale_txt = ""
        if "X2" in self.model_name.upper(): scale_txt = " (x2)"
        elif "X4" in self.model_name.upper(): scale_txt = " (x4)"

        if self.output_resize != 1.0: details.append(f"Out:{int(self.output_resize*100)}%")
        if self.frame_gen != "OFF": details.append(f"Gen:{self.frame_gen}")

        detail_str = f"| {', '.join(details)}" if details else ""
        return f"{icon} {self.model_name}{scale_txt} {detail_str} {state}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ProcessingStep':
        known_keys = ProcessingStep.__annotations__.keys()
        filtered = {k: v for k, v in data.items() if k in known_keys}
        if 'id' not in filtered: filtered['id'] = str(uuid.uuid4())
        return ProcessingStep(**filtered)

# -----------------------------------------------------------------------------
# 2. UI COMPONENTS
# -----------------------------------------------------------------------------

class ToolTip:
    """Tooltip nativo para Tkinter."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text: return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#333333", foreground="#FFFFFF", relief=tk.SOLID, borderwidth=1,
                         font=("Arial", "9", "normal"))
        label.pack(ipadx=5, ipady=2)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class ValidatedEntry(ctk.CTkEntry):
    def __init__(self, master, is_float=True, min_val=0.0, max_val=9999.0, **kwargs):
        super().__init__(master, **kwargs)
        self.is_float = is_float
        self.min_val = min_val
        self.max_val = max_val
        vcmd = (self.register(self._validate), '%P')
        self.configure(validate="key", validatecommand=vcmd)
        self.bind("<FocusOut>", self._on_focus_out)

    def _validate(self, new_value):
        if new_value == "": return True
        try:
            val = float(new_value) if self.is_float else int(new_value)
            return True
        except ValueError:
            return False

    def _on_focus_out(self, event):
        val = self.get()
        if not val: return
        try:
            num = float(val)
            if num < self.min_val: self.delete(0, "end"); self.insert(0, str(self.min_val))
            elif num > self.max_val: self.delete(0, "end"); self.insert(0, str(self.max_val))
        except: pass

class StepEditorDialog(ctk.CTkToplevel):
    """Editor modal avanzado."""
    def __init__(self, parent, step: ProcessingStep, on_save_callback: Callable):
        super().__init__(parent)
        self.title(f"Edit Step")
        self.geometry("450x650")
        self.resizable(False, False)
        self.configure(fg_color=COLOR_BG)

        icon_path = _resolve_asset_path("logo.ico")
        if icon_path: self.after(200, lambda: self.iconbitmap(icon_path))

        self.step = step
        self.on_save = on_save_callback
        self.widgets = {}

        self.transient(parent)
        self.grab_set()

        self.main_frame = ctk.CTkScrollableFrame(self, fg_color=COLOR_WIDGET, label_text="Step Configuration")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_ui()
        self._populate_values()
        self._build_buttons()

    def _build_ui(self):
        row = 0
        # Model & Hardware
        self._header("AI Model & Hardware", row); row += 1

        self._label("Model Name:", row)
        try:
            model_values = list_available_models()
        except Exception:
            model_values = []
        if not model_values:
            model_values = ["RealESR_Gx4", "RealESR_Animex4", "BSRGANx4", "BSRGANx2", "RealESRGANx4", "RealESRNetx4", "IRCNN_Mx1", "IRCNN_Lx1", "GFPGAN", "RIFE", "RIFE_Lite"]
        self.widgets['model_name'] = ctk.CTkComboBox(self.main_frame, values=model_values, width=220)
        self.widgets['model_name'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        self._label("Compute Device:", row)
        self.widgets['gpu'] = ctk.CTkComboBox(self.main_frame, values=GPU_OPTIONS, width=220)
        self.widgets['gpu'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        # Scaling
        self._header("Resolution & Scaling", row); row += 1

        self._label("Input Resize (0.1 - 1.0):", row)
        self.widgets['input_resize'] = ValidatedEntry(self.main_frame, min_val=0.1, max_val=1.0, width=220)
        self.widgets['input_resize'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        self._label("Output Resize (0.1 - 8.0):", row)
        self.widgets['output_resize'] = ValidatedEntry(self.main_frame, min_val=0.1, max_val=8.0, width=220)
        self.widgets['output_resize'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        # Advanced
        self._header("Advanced Processing", row); row += 1

        self._label("Blending (0.0 - 1.0):", row)
        self.widgets['blending'] = ValidatedEntry(self.main_frame, min_val=0.0, max_val=1.0, width=220)
        self.widgets['blending'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        self._label("VRAM Limit (GB):", row)
        self.widgets['vram_limit'] = ValidatedEntry(self.main_frame, min_val=0.0, max_val=24.0, width=220)
        self.widgets['vram_limit'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        # Output
        self._header("Output Format", row); row += 1

        self._label("Extension:", row)
        exts = list(set(IMG_EXTENSIONS + VID_EXTENSIONS)); exts.sort()
        self.widgets['extension'] = ctk.CTkComboBox(self.main_frame, values=exts, width=220)
        self.widgets['extension'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        self._label("Video Codec:", row)
        self.widgets['video_codec'] = ctk.CTkComboBox(self.main_frame, values=[""] + VIDEO_CODECS, width=220)
        self.widgets['video_codec'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        self._label("Frame Generation:", row)
        self.widgets['frame_gen'] = ctk.CTkComboBox(self.main_frame, values=FRAME_GEN_OPTIONS, width=220)
        self.widgets['frame_gen'].grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

        self.widgets['keep_frames'] = ctk.CTkCheckBox(self.main_frame, text="Keep Intermediate Frames", text_color=COLOR_TEXT)
        self.widgets['keep_frames'].grid(row=row, column=0, columnspan=2, pady=15)

    def _header(self, text, r):
        lbl = ctk.CTkLabel(self.main_frame, text=text, font=("Roboto", 13, "bold"), text_color=COLOR_ACCENT, anchor="w")
        lbl.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 5), padx=5)
        # Separator
        sep = ctk.CTkFrame(self.main_frame, height=2, fg_color=COLOR_BORDER)
        sep.grid(row=r, column=0, columnspan=2, sticky="ews", pady=(0, 0))

    def _label(self, text, r):
        ctk.CTkLabel(self.main_frame, text=text, text_color=COLOR_TEXT_SEC, anchor="w").grid(row=r, column=0, sticky="w", padx=10)

    def _populate_values(self):
        s = self.step
        # Para ComboBox, usar set()
        try:
            self.widgets['model_name'].set(s.model_name)
        except Exception:
            # Fallback: si no está en lista, agregarlo dinámicamente
            current_vals = list(self.widgets['model_name'].cget("values"))
            if s.model_name and s.model_name not in current_vals:
                current_vals = [s.model_name] + current_vals
                self.widgets['model_name'].configure(values=current_vals)
                self.widgets['model_name'].set(s.model_name)
        self.widgets['gpu'].set(s.gpu)
        self.widgets['input_resize'].insert(0, str(s.input_resize))
        self.widgets['output_resize'].insert(0, str(s.output_resize))
        self.widgets['blending'].insert(0, str(s.blending))
        self.widgets['vram_limit'].insert(0, str(s.vram_limit))

        if s.extension not in self.widgets['extension'].cget("values"):
             self.widgets['extension'].set(s.extension)
        else:
             self.widgets['extension'].set(s.extension)

        self.widgets['video_codec'].set(s.video_codec or "")
        self.widgets['frame_gen'].set(s.frame_gen)
        if s.keep_frames: self.widgets['keep_frames'].select()

    def _build_buttons(self):
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkButton(btn_frame, text="Cancel", fg_color="#424242", hover_color="#616161",
                      command=self.destroy).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(btn_frame, text="Save Changes", fg_color=COLOR_SUCCESS, hover_color="#00E676", text_color="white",
                      command=self.save).pack(side="right", expand=True, padx=5)

    def save(self):
        try:
            self.step.model_name = self.widgets['model_name'].get()
            self.step.gpu = self.widgets['gpu'].get()
            self.step.input_resize = float(self.widgets['input_resize'].get())
            self.step.output_resize = float(self.widgets['output_resize'].get())
            self.step.blending = float(self.widgets['blending'].get())
            self.step.vram_limit = float(self.widgets['vram_limit'].get())
            self.step.extension = self.widgets['extension'].get()

            codec = self.widgets['video_codec'].get()
            self.step.video_codec = codec if codec.strip() else None
            self.step.frame_gen = self.widgets['frame_gen'].get()
            self.step.keep_frames = bool(self.widgets['keep_frames'].get())

            self.on_save(self.step)
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Check numeric fields: {e}")

class ModernStepCard(ctk.CTkFrame):
    """Tarjeta visual que representa un paso."""
    def __init__(self, master, step: ProcessingStep, index: int, total_steps: int, callbacks: Dict):

        color_bg = COLOR_WIDGET if step.enabled else "#111111"
        color_border = COLOR_BORDER if step.enabled else "#222222"

        super().__init__(master, corner_radius=8, border_width=1, fg_color=color_bg, border_color=color_border)

        self.step = step
        self.callbacks = callbacks
        self.index = index

        self.grid_columnconfigure(1, weight=1)

        # --- HEADER ---
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=8)

        # Índice y Icono
        icon = "🎬" if step.is_video_operation else "🖼️"
        ctk.CTkLabel(header, text=f"{index+1}", font=("Arial", 16, "bold"), text_color=COLOR_ACCENT, width=25).pack(side="left")
        ctk.CTkLabel(header, text=icon, font=("Arial", 16)).pack(side="left", padx=(0, 5))

        # Título
        title_txt = step.model_name if step.expanded else step.get_summary()
        title_col = COLOR_TEXT if step.enabled else "gray"
        self.lbl_title = ctk.CTkLabel(header, text=title_txt, font=("Roboto", 13, "bold"), text_color=title_col, anchor="w")
        self.lbl_title.pack(side="left", fill="x", expand=True, padx=5)

        # Validación Warning
        warnings = step.validate()
        if warnings and step.enabled:
            warn_lbl = ctk.CTkLabel(header, text="⚠️", text_color=COLOR_ACCENT)
            warn_lbl.pack(side="right", padx=5)
            ToolTip(warn_lbl, "\n".join(warnings))

        # Switch y Expand
        self.switch = ctk.CTkSwitch(header, text="", width=35, height=20, command=self._on_toggle,
                                    onvalue=True, offvalue=False, progress_color=COLOR_SUCCESS)
        if step.enabled: self.switch.select()
        else: self.switch.deselect()
        self.switch.pack(side="right", padx=5)

        btn_exp = ctk.CTkButton(header, text="▼" if not step.expanded else "▲", width=25, height=25,
                                fg_color="transparent", text_color="gray", hover_color="#333333",
                                command=self._toggle_expand)
        btn_exp.pack(side="right")

        # --- BODY EXPANDIDO ---
        if step.expanded and step.enabled:
            body = ctk.CTkFrame(self, fg_color="transparent")
            body.pack(fill="x", padx=10, pady=(0, 10))

            # Info Grid
            self._info_row(body, 0, "Input Scale:", f"{int(step.input_resize*100)}%", "Output Scale:", f"{int(step.output_resize*100)}%")
            self._info_row(body, 1, "VRAM Limit:", f"{step.vram_limit} GB", "Blending:", str(step.blending))
            self._info_row(body, 2, "GPU:", step.gpu, "Extension:", step.extension)
            if step.frame_gen != "OFF":
                self._info_row(body, 3, "Frame Gen:", step.frame_gen, "Codec:", step.video_codec or "Auto")

            # Action Buttons
            actions = ctk.CTkFrame(body, fg_color="transparent", height=30)
            actions.grid(row=99, column=0, columnspan=4, sticky="ew", pady=(15, 0))

            # Move
            if index > 0:
                self._btn(actions, "⬆", lambda: callbacks['move'](step.id, -1), "left")
            if index < total_steps - 1:
                self._btn(actions, "⬇", lambda: callbacks['move'](step.id, 1), "left")

            # CRUD
            self._btn(actions, "🗑 Delete", lambda: callbacks['delete'](step.id), "right", COLOR_ERROR, "#D32F2F")
            self._btn(actions, "⧉ Clone", lambda: callbacks['clone'](step.id), "right", "#5E35B1", "#7E57C2")
            self._btn(actions, "✎ Edit", lambda: callbacks['edit'](step.id), "right", "#1976D2", "#42A5F5")

    def _info_row(self, master, r, t1, v1, t2, v2):
        self._cell(master, r, 0, t1, v1)
        self._cell(master, r, 1, t2, v2)

    def _cell(self, master, r, c, title, value):
        f = ctk.CTkFrame(master, fg_color="transparent")
        f.grid(row=r, column=c, sticky="w", padx=5, pady=2)
        ctk.CTkLabel(f, text=title, font=("Arial", 11), text_color=COLOR_TEXT_SEC).pack(side="left")
        ctk.CTkLabel(f, text=str(value), font=("Arial", 11, "bold"), text_color=COLOR_TEXT).pack(side="left", padx=5)

    def _btn(self, master, txt, cmd, side, col="#424242", hov="#616161"):
        ctk.CTkButton(master, text=txt, width=50, height=24, fg_color=col, hover_color=hov,
                      font=("Arial", 11), command=cmd).pack(side=side, padx=2)

    def _on_toggle(self):
        self.step.enabled = bool(self.switch.get())
        self.callbacks['refresh']()

    def _toggle_expand(self):
        self.step.expanded = not self.step.expanded
        self.callbacks['refresh']()

# -----------------------------------------------------------------------------
# 3. MANAGER CONTROLLER
# -----------------------------------------------------------------------------

class ChainManager(ctk.CTkToplevel):
    def __init__(self, parent, get_current_settings_callback: Callable):
        super().__init__(parent)
        self.title("Workflow Chain Manager")
        self.geometry("500x750")
        self.minsize(450, 500)
        self.configure(fg_color=COLOR_BG)
        self.transient(parent)

        icon_path = _resolve_asset_path("logo.ico")
        if icon_path: self.after(200, lambda: self.iconbitmap(icon_path))

        self.get_current_settings = get_current_settings_callback
        self.steps: List[ProcessingStep] = []

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_toolbar()

        # Scroll Area
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="#121212", label_text="Processing Pipeline")
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self._build_bottom_panel()
        self.refresh_ui()

    def _build_toolbar(self):
        tb = ctk.CTkFrame(self, height=50, fg_color="transparent")
        tb.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(tb, text="Active Chain", font=("Roboto", 18, "bold"), text_color=COLOR_ACCENT).pack(side="left", padx=5)

        self.lbl_stats = ctk.CTkLabel(tb, text="", font=("Arial", 11), text_color="gray")
        self.lbl_stats.pack(side="left", padx=15)

        self._tb_btn(tb, "🗑 Clear", self.clear_chain, COLOR_ERROR, "#D32F2F")
        self._tb_btn(tb, "💾 Save", self.save_preset, "#455A64", "#607D8B")
        self._tb_btn(tb, "📂 Load", self.load_preset, "#455A64", "#607D8B")

    def _tb_btn(self, master, txt, cmd, col, hov):
        ctk.CTkButton(master, text=txt, width=60, height=28, fg_color=col, hover_color=hov,
                      font=("Arial", 11, "bold"), command=cmd).pack(side="right", padx=2)

    def _build_bottom_panel(self):
        panel = ctk.CTkFrame(self, height=80, fg_color=COLOR_WIDGET)
        panel.grid(row=2, column=0, sticky="ew")

        self.btn_add = ctk.CTkButton(panel, text="➕ APPEND CURRENT SETTINGS AS STEP",
                                     command=self.add_step_from_gui,
                                     fg_color="#00695C", hover_color="#00897B",
                                     height=45, font=("Roboto", 13, "bold"))
        self.btn_add.pack(padx=20, pady=15, fill="x")

    def add_step_from_gui(self):
        try:
            s = self.get_current_settings()
            if not s['model'] or "•••" in s['model']:
                messagebox.showerror("Error", "Please select a valid AI Model in the main window first.")
                return

            # Inferencia automática de tipo de extensión
            is_rife = "RIFE" in s['model'].upper()
            has_gen = s.get('frame_gen', "OFF") != "OFF"

            # Si es RIFE o tiene FrameGen, forzamos modo Video si la extensión no es explícita
            # Prioridad: Extensión de video si está seleccionada, sino .mp4
            if is_rife or has_gen:
                ext = s.get('ext_vid') if s.get('ext_vid') in VID_EXTENSIONS else ".mp4"
            else:
                # Upscalers normales usan extensión de imagen a menos que el usuario haya seleccionado explicitamente video en main
                # Pero en la cadena, paso intermedio suele ser imagen a menos que sea el final.
                # Por defecto tomamos la configuración visual actual.
                ext = s.get('ext_img') if s.get('ext_img') in IMG_EXTENSIONS else ".png"

            new_step = ProcessingStep(
                model_name=s['model'],
                input_resize=s['input_resize'],
                output_resize=s['output_resize'],
                blending=s['blending'],
                vram_limit=s['vram'],
                extension=ext,
                video_codec=s.get('codec'),
                frame_gen=s.get('frame_gen', "OFF"),
                keep_frames=s.get('keep_frames', False),
                gpu=s.get('gpu', "Auto")
            )

            self.steps.append(new_step)
            self.refresh_ui()
            self.after(100, lambda: self.scroll_frame._parent_canvas.yview_moveto(1.0))

        except Exception as e:
            messagebox.showerror("Error", f"Could not capture settings: {e}")

    # --- Lógica CRUD ---

    def _move(self, uid, direction):
        idx = next((i for i, s in enumerate(self.steps) if s.id == uid), -1)
        if idx == -1: return
        n_idx = idx + direction
        if 0 <= n_idx < len(self.steps):
            self.steps[idx], self.steps[n_idx] = self.steps[n_idx], self.steps[idx]
            self.refresh_ui()

    def _delete(self, uid):
        self.steps = [s for s in self.steps if s.id != uid]
        self.refresh_ui()

    def _clone(self, uid):
        idx = next((i for i, s in enumerate(self.steps) if s.id == uid), -1)
        if idx != -1:
            cloned = copy.deepcopy(self.steps[idx])
            cloned.id = str(uuid.uuid4())
            self.steps.insert(idx + 1, cloned)
            self.refresh_ui()

    def _edit(self, uid):
        step = next((s for s in self.steps if s.id == uid), None)
        if step:
            StepEditorDialog(self, step, lambda x: self.refresh_ui())

    def clear_chain(self):
        if self.steps and messagebox.askyesno("Confirm", "Clear entire chain?"):
            self.steps = []
            self.refresh_ui()

    def get_chain(self) -> List[ProcessingStep]:
        return [s for s in self.steps if s.enabled]

    # --- Rendering ---

    def refresh_ui(self):
        for w in self.scroll_frame.winfo_children(): w.destroy()

        if not self.steps:
            f = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
            f.pack(pady=60)
            ctk.CTkLabel(f, text="Workflow is Empty", font=("Arial", 16, "bold"), text_color="gray").pack()
            ctk.CTkLabel(f, text="Configure main window settings\nand click 'Append' below.", text_color="#555").pack(pady=5)
            self.lbl_stats.configure(text="")
        else:
            cbs = {'move': self._move, 'delete': self._delete, 'clone': self._clone, 'edit': self._edit, 'refresh': self.refresh_ui}

            total_scale = 1.0
            for i, step in enumerate(self.steps):
                if step.enabled: total_scale *= step.estimated_scale_factor
                ModernStepCard(self.scroll_frame, step, i, len(self.steps), cbs).pack(fill="x", pady=6, padx=5)

            self.lbl_stats.configure(text=f"Est. Scale: x{total_scale:.2f}")

    # --- JSON Persistence ---

    def save_preset(self):
        if not self.steps: return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Warlock Preset", "*.json")])
        if path:
            try:
                data = {"version": 1, "steps": [s.to_dict() for s in self.steps]}
                with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4)
                messagebox.showinfo("Saved", "Workflow saved.")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {e}")

    def load_preset(self):
        path = filedialog.askopenfilename(filetypes=[("Warlock Preset", "*.json")])
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f: data = json.load(f)

                # Soporte legacy (si el json es una lista directa)
                items = data if isinstance(data, list) else data.get("steps", [])

                self.steps = [ProcessingStep.from_dict(i) for i in items]
                self.refresh_ui()
            except Exception as e:
                messagebox.showerror("Error", f"Load failed: {e}")

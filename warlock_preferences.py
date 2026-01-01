import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
import psutil
import requests
from packaging import version as pkg_version

# Integración con Hardware (Opcional)
try:
    import wmi
except ImportError:
    wmi = None
try:
    import GPUtil
except ImportError:
    GPUtil = None

# -----------------------------------------------------------------------------
# IMPORTAR TEMA Y PUENTE DE COMPATIBILIDAD (FIX FINAL)
# -----------------------------------------------------------------------------
from warlock_theme import *

# --- THEME BRIDGE ---
# Mapea los nombres que usa Preferences a los nombres que existen en WarlockTheme
_aliases = {
    "bg_sidebar": "widget_bg",
    "bg_main": "bg",
    "card_bg": "widget_bg",
    "card_hover": "btn_hover",
    "text_dim": "text_sec",
    "input_bg": "entry_bg",
    "gold": "accent",
    "border": "widget_border"  # <--- AGREGADO: Esto soluciona tu error actual
}

# Aplicar los alias de forma segura
for _new_key, _existing_key in _aliases.items():
    if _new_key not in THEME:
        # Si la clave original tampoco existe, usamos un gris oscuro por seguridad
        THEME[_new_key] = THEME.get(_existing_key, "#333333")

# Aseguramos compatibilidad inversa extra si el tema es muy antiguo
if "info" not in THEME:
    THEME["info"] = "#0277BD"
if "success" not in THEME:
    THEME["success"] = "#00C853"
if "error" not in THEME:
    THEME["error"] = "#B71C1C"

# =============================================================================
# UTILIDADES DE UI (TOOLTIPS)
# =============================================================================


class ToolTip:
    """Muestra un mensaje flotante al pasar el mouse sobre un widget."""

    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.id = None
        self.tw = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)

    def enter(self, event=None):
        self.id = self.widget.after(self.delay, self.show)

    def leave(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        self.hide()

    def show(self):
        if self.tw:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tw = ctk.CTkToplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")

        # Estilo del Tooltip
        label = ctk.CTkLabel(
            self.tw,
            text=self.text,
            justify='left',
            bg_color="#2B2B2B",
            fg_color="#2B2B2B",
            text_color="#FFFFFF",
            corner_radius=4,
            font=("Segoe UI", 10),
            padx=8, pady=4
        )
        label.pack()

        # Borde sutil
        frame = ctk.CTkFrame(self.tw, width=0, height=0,
                             border_width=1, border_color="#555")
        frame.pack()

    def hide(self):
        if self.tw:
            self.tw.destroy()
            self.tw = None

# =============================================================================
# GESTOR DE CONFIGURACIÓN (CENTRALIZADO & EXPANDIDO)
# =============================================================================


class ConfigManager:
    # Configuración Maestra con todas las opciones nuevas
    DEFAULT_CONFIG = {
        # --- GENERAL ---
        "app_theme": "Dark",            # Dark, Light, System
        "ui_scaling": 1.0,              # 0.8 - 1.25
        "window_opacity": 1.0,          # 0.5 - 1.0
        "keep_window_on_top": False,
        "check_updates_on_startup": True,
        "start_minimized": False,       # Nuevo
        "notifications_enabled": True,  # Nuevo
        "play_sounds": True,            # Nuevo

        # --- APARIENCIA AVANZADA ---
        "corner_radius": 10,            # Nuevo: Redondez de ventanas
        "font_size_offset": 0,          # Nuevo: Ajuste global de tamaño letra

        # --- RENDIMIENTO ---
        "process_priority": "Normal",
        "onnx_provider_preference": "Auto",
        "auto_close_on_finish": False,
        "max_cpu_threads": 0,           # 0 = Auto
        "gpu_memory_limit": 0,          # 0 = No limit (Nuevo)

        # --- SALIDA Y ARCHIVOS (NUEVO) ---
        "default_output_dir": "",       # Vacío = Carpeta del proyecto
        "filename_pattern": "Warlock_Output_{date}_{time}",  # Patrón de nombre
        "auto_save_logs": True,
        "backup_count": 5,

        # --- RUTAS MANUALES ---
        "custom_ffmpeg_path": "",
        "custom_exiftool_path": "",

        # --- DEBUG ---
        "extended_logging": False
    }

    @staticmethod
    def get_base_path():
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def get_assets_path():
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(
            os.path.abspath(__file__)))
        return os.path.join(base_dir, "Assets")

    @staticmethod
    def get_config_path():
        return os.path.join(ConfigManager.get_base_path(), "warlock_config.json")

    @staticmethod
    def load_config() -> Dict[str, Any]:
        path = ConfigManager.get_config_path()
        config = ConfigManager.DEFAULT_CONFIG.copy()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        config[k] = v
            except Exception as e:
                print(f"[Config] Warn: {e}")
        return config

    @staticmethod
    def save_config(config_data: Dict[str, Any]):
        try:
            path = ConfigManager.get_config_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"[Config] Error saving: {e}")

    @staticmethod
    def reset_config():
        ConfigManager.save_config(ConfigManager.DEFAULT_CONFIG)
        return ConfigManager.DEFAULT_CONFIG.copy()

# =============================================================================
# HARDWARE SCANNER
# =============================================================================


class HardwareScanner:
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        info = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "ram_total": 0, "ram_used": 0,
            "gpu": "Integrated / Unknown", "vram": 0
        }
        try:
            mem = psutil.virtual_memory()
            info["ram_total"] = round(mem.total / (1024**3), 2)
            info["ram_used"] = round(mem.used / (1024**3), 2)

            if wmi and platform.system() == "Windows":
                try:
                    c = wmi.WMI()
                    info["cpu"] = c.Win32_Processor()[0].Name.strip()
                except:
                    pass

            # Detección GPU mejorada
            gpu_found = False
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        info["gpu"] = gpus[0].name
                        info["vram"] = round(gpus[0].memoryTotal / 1024, 2)
                        gpu_found = True
                except:
                    pass

            if not gpu_found and wmi and platform.system() == "Windows":
                try:
                    c = wmi.WMI()
                    best_vram = -1
                    for g in c.Win32_VideoController():
                        try:
                            v_bytes = int(g.AdapterRAM) if g.AdapterRAM else 0
                            if v_bytes < 0:
                                v_bytes += 2**32
                            v_gb = round(v_bytes / (1024**3), 2)
                            if v_gb > best_vram:
                                best_vram = v_gb
                                info["gpu"] = g.Name
                                info["vram"] = v_gb
                        except:
                            continue
                except:
                    pass
        except Exception as e:
            info["os"] = f"Error: {e}"
        return info

# =============================================================================
# UPDATE MANAGER
# =============================================================================


class UpdateManager:
    def __init__(self, owner, repo, version):
        self.url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        self.current = version

    def check(self):
        try:
            r = requests.get(self.url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                tag = data.get("tag_name", "v0.0").lstrip("v")
                if pkg_version.parse(tag) > pkg_version.parse(self.current):
                    return True, tag, data.get("html_url"), data.get("body")
            return False, self.current, None, None
        except Exception as e:
            return None, str(e), None, None

# =============================================================================
# COMPONENTES UI
# =============================================================================


class SidebarButton(ctk.CTkButton):
    def __init__(self, master, text, icon_char, command, is_selected=False):
        color = THEME["card_bg"] if is_selected else "transparent"
        text_col = THEME["accent"] if is_selected else THEME["text_dim"]

        super().__init__(
            master, text=f"  {icon_char}   {text}", anchor="w",
            fg_color=color, hover_color=THEME["card_hover"],
            text_color=text_col, height=45, corner_radius=6,
            font=("Segoe UI", 12, "bold"), command=command,
            border_width=0
        )


class SettingRow(ctk.CTkFrame):
    def __init__(self, master, title, desc=None, tooltip_text=None):
        super().__init__(master, fg_color="transparent")
        self.pack(fill="x", pady=8)

        lbl_frame = ctk.CTkFrame(self, fg_color="transparent")
        lbl_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

        t_lbl = ctk.CTkLabel(lbl_frame, text=title, font=("Segoe UI", 12, "bold"),
                             text_color=THEME["text_main"], anchor="w")
        t_lbl.pack(fill="x")

        if tooltip_text:
            ToolTip(t_lbl, tooltip_text)

        if desc:
            d_lbl = ctk.CTkLabel(lbl_frame, text=desc, font=("Segoe UI", 10),
                                 text_color=THEME["text_dim"], anchor="w")
            d_lbl.pack(fill="x")
            if tooltip_text and not getattr(t_lbl, "tw", None):
                ToolTip(d_lbl, tooltip_text)

        self.control_area = ctk.CTkFrame(self, fg_color="transparent")
        self.control_area.pack(side="right")

    def add_widget(self, widget):
        widget.pack(in_=self.control_area)

# =============================================================================
# VENTANA PRINCIPAL DE PREFERENCIAS
# =============================================================================


class PreferencesWindow(ctk.CTkToplevel):
    def __init__(self, master, version, owner, repo):
        super().__init__(master)

        # Config Inicial
        self.title("Warlock Preferences")
        self.geometry("850x600")
        self.minsize(850, 600)
        self.configure(fg_color=THEME["bg_main"])

        # Cargar datos
        self.config = ConfigManager.load_config()
        self.version = version
        self.owner = owner
        self.repo = repo
        self.cached_hw_info = None

        # Grid Principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_content_area()
        self.show_page("general")

        # Intentar poner al frente
        self.after(200, lambda: self.attributes('-topmost', True))
        self.lift()
        self.focus_force()

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self, width=250, corner_radius=0, fg_color=THEME["bg_sidebar"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Header Sidebar
        head = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        head.pack(pady=30, padx=20, fill="x")
        ctk.CTkLabel(head, text="PREFERENCES", font=("Impact", 24),
                     text_color=THEME["accent"]).pack(anchor="w")
        ctk.CTkLabel(head, text="Warlock Studio Control Center", font=(
            "Segoe UI", 10), text_color=THEME["text_dim"]).pack(anchor="w")

        # Botones Navegación
        self.nav_btns = {}
        pages = [
            ("general", "General & UI", "⚙"),
            ("output", "Output & Files", "📂"),
            ("performance", "Performance / AI", "🚀"),
            ("integrations", "Tools & Paths", "🔗"),
            ("logs", "Maintenance", "🛠"),
            ("about", "About & Update", "ℹ")
        ]

        for pid, text, icon in pages:
            btn = SidebarButton(self.sidebar, text, icon,
                                lambda p=pid: self.show_page(p))
            btn.pack(fill="x", padx=15, pady=4)
            self.nav_btns[pid] = btn

    def setup_content_area(self):
        self.content = ctk.CTkScrollableFrame(
            self, fg_color="transparent", corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)

        # Scrollbar styling hack
        try:
            self.content._scrollbar.configure(
                width=12, fg_color=THEME["bg_main"])
        except:
            pass

    def show_page(self, page_id):
        # Actualizar estilo botones
        for pid, btn in self.nav_btns.items():
            btn.configure(fg_color=THEME["card_bg"] if pid == page_id else "transparent",
                          text_color=THEME["accent"] if pid == page_id else THEME["text_dim"])

        # Limpiar frame
        for widget in self.content.winfo_children():
            widget.destroy()

        # Renderizar
        method = getattr(self, f"build_{page_id}", None)
        if method:
            # Animación simple de entrada (fade in simulado por orden de pack)
            title = page_id.replace("_", " ").title()
            if page_id == "output":
                title = "Output Management"
            self.header(title)
            method()

    def header(self, text):
        ctk.CTkLabel(self.content, text=text, font=("Segoe UI", 26, "bold"),
                     text_color=THEME["text_main"]).pack(anchor="w", pady=(0, 25))
        ctk.CTkFrame(self.content, height=2, fg_color=THEME["border"]).pack(
            fill="x", pady=(0, 20))

    # =========================================================================
    # BUILDERS (PÁGINAS)
    # =========================================================================

    def build_general(self):
        # --- SECCIÓN VISUAL ---
        self.sub_header("Visual Appearance")
        grp_vis = self.create_group()

        # Theme
        r_theme = SettingRow(grp_vis, "Application Theme",
                             "Light or Dark mode preference")
        om_theme = ctk.CTkOptionMenu(grp_vis, values=["Dark", "Light", "System"],
                                     command=lambda v: self.save(
                                         "app_theme", v, lambda: ctk.set_appearance_mode(v)),
                                     fg_color=THEME["input_bg"], button_color=THEME["accent"], width=140)
        om_theme.set(self.config.get("app_theme", "Dark"))
        r_theme.add_widget(om_theme)

        # Corner Radius
        r_rad = SettingRow(grp_vis, "Window Corner Radius",
                           "Roundness of interface elements (Restart required)")
        sl_rad = ctk.CTkSlider(grp_vis, from_=0, to=20, number_of_steps=20,
                               command=lambda v: self.save_debounced(
                                   "corner_radius", int(v)),
                               progress_color=THEME["accent"], button_color=THEME["gold"])
        sl_rad.set(self.config.get("corner_radius", 10))
        r_rad.add_widget(sl_rad)

        # Opacidad
        r_op = SettingRow(grp_vis, "Window Opacity",
                          "Transparency level of the main window")
        sl_op = ctk.CTkSlider(grp_vis, from_=0.5, to=1.0, number_of_steps=50,
                              command=lambda v: self.update_opacity(v),
                              progress_color=THEME["accent"], button_color=THEME["gold"])
        sl_op.set(self.config.get("window_opacity", 1.0))
        sl_op.bind("<ButtonRelease-1>",
                   lambda e: ConfigManager.save_config(self.config))
        r_op.add_widget(sl_op)

        # Scaling
        r_sca = SettingRow(grp_vis, "UI Scaling",
                           "Increase size for 4K monitors")
        om_sca = ctk.CTkOptionMenu(grp_vis, values=["80%", "90%", "100%", "110%", "125%", "150%"],
                                   command=self.set_ui_scaling,
                                   fg_color=THEME["input_bg"], button_color=THEME["accent"], width=140)
        om_sca.set(f"{int(self.config.get('ui_scaling', 1.0)*100)}%")
        r_sca.add_widget(om_sca)

        # --- SECCIÓN COMPORTAMIENTO ---
        self.sub_header("Behavior & System", pady=20)
        grp_beh = self.create_group()

        self.add_switch(grp_beh, "Always on Top", "keep_window_on_top",
                        cmd=lambda v: self.master.winfo_toplevel().attributes("-topmost", v))
        self.add_switch(grp_beh, "Notifications Enabled",
                        "notifications_enabled", "Show toast messages on finish")
        self.add_switch(grp_beh, "Play Sounds", "play_sounds",
                        "Audio feedback for actions")
        self.add_switch(grp_beh, "Check Updates on Startup",
                        "check_updates_on_startup")

    def build_output(self):
        # --- GESTIÓN DE ARCHIVOS ---
        self.sub_header("File Management")
        grp = self.create_group()

        # Default Directory
        ctk.CTkLabel(grp, text="Default Output Directory", font=(
            "Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 0))

        dir_row = ctk.CTkFrame(grp, fg_color="transparent")
        dir_row.pack(fill="x", padx=10, pady=5)

        self.ent_dir = ctk.CTkEntry(
            dir_row, fg_color=THEME["input_bg"], border_color=THEME["border"])
        self.ent_dir.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.ent_dir.insert(0, self.config.get("default_output_dir", ""))
        self.ent_dir.configure(state="readonly")

        ctk.CTkButton(dir_row, text="Browse...", width=100, fg_color=THEME["accent"], text_color="#000",
                      command=self.browse_output_dir).pack(side="right")

        ctk.CTkButton(dir_row, text="Reset", width=60, fg_color=THEME["bg_main"],
                      command=lambda: self.save_dir("")).pack(side="right", padx=5)

        # Filename Pattern
        r_pat = SettingRow(grp, "Filename Pattern",
                           "Use {date}, {time}, {original} as placeholders")
        ent_pat = ctk.CTkEntry(
            grp, fg_color=THEME["input_bg"], border_color=THEME["border"], width=250)
        ent_pat.insert(0, self.config.get(
            "filename_pattern", "Warlock_{date}_{time}"))
        ent_pat.bind("<FocusOut>", lambda e: self.save(
            "filename_pattern", ent_pat.get()))
        r_pat.add_widget(ent_pat)

        # Options
        self.add_switch(grp, "Auto-Save Processing Logs",
                        "auto_save_logs", "Save a .txt log with each export")

    def build_performance(self):
        # --- HARDWARE INFO ---
        self.hw_frame = ctk.CTkFrame(
            self.content, fg_color=THEME["card_bg"], border_color=THEME["accent"], border_width=1)
        self.hw_frame.pack(fill="x", pady=(0, 20))

        if self.cached_hw_info:
            self._display_hw_info(self.cached_hw_info)
        else:
            ctk.CTkLabel(self.hw_frame, text="Scanning System Hardware...",
                         text_color=THEME["accent"]).pack(pady=20)
            threading.Thread(target=self._run_hw_scan, daemon=True).start()

        # --- AI SETTINGS ---
        self.sub_header("Artificial Intelligence Engine")
        grp = self.create_group()

        # Provider
        r_prov = SettingRow(grp, "Inference Backend", "Hardware acceleration provider (Requires Restart)",
                            tooltip_text="CUDA: NVIDIA GPUs (Fastest)\nDirectML: AMD/Intel/NVIDIA (Compatible)\nCPU: Slowest, universal compatibility.")
        om_prov = ctk.CTkOptionMenu(grp, values=["Auto", "CUDA", "DirectML", "CPU", "OpenVINO"],
                                    command=lambda v: self.save(
                                        "onnx_provider_preference", v),
                                    fg_color=THEME["input_bg"], button_color=THEME["accent"])
        om_prov.set(self.config.get("onnx_provider_preference", "Auto"))
        r_prov.add_widget(om_prov)

        # Priority
        r_prio = SettingRow(grp, "Process Priority",
                            "OS CPU scheduling priority")
        om_prio = ctk.CTkOptionMenu(grp, values=["Normal", "Above Normal", "High", "Realtime"],
                                    command=lambda v: self.save(
                                        "process_priority", v),
                                    fg_color=THEME["input_bg"], button_color=THEME["accent"])
        om_prio.set(self.config.get("process_priority", "Normal"))
        r_prio.add_widget(om_prio)

        self.add_switch(grp, "Auto-Close After Task", "auto_close_on_finish")

    def build_integrations(self):
        ctk.CTkLabel(self.content, text="External Tools Configuration", font=(
            "Segoe UI", 12), text_color=THEME["text_dim"]).pack(anchor="w")

        grp = self.create_group()
        self.create_path_selector(
            grp, "FFmpeg Binary", "custom_ffmpeg_path", "ffmpeg.exe")
        self.create_path_selector(
            grp, "ExifTool Binary", "custom_exiftool_path", "exiftool.exe")

    def build_logs(self):
        # --- MAINTENANCE ---
        grp = self.create_group()

        # Tools
        btn_row = ctk.CTkFrame(grp, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=10)

        self.create_tool_btn(btn_row, "Open Logs Folder",
                             self.open_logs_dir, THEME["border"])
        self.create_tool_btn(btn_row, "Clean Temp Files",
                             self.clean_temp_files, THEME["info"])
        self.create_tool_btn(btn_row, "Export Debug Zip",
                             self.export_debug_info, THEME["accent"], text_col="#000")

        # Reset Danger Zone
        ctk.CTkFrame(grp, height=1, fg_color="#444").pack(
            fill="x", pady=15, padx=10)

        dz_row = ctk.CTkFrame(grp, fg_color="transparent")
        dz_row.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(dz_row, text="Danger Zone:", text_color=THEME["error"], font=(
            "Segoe UI", 12, "bold")).pack(side="left")
        ctk.CTkButton(dz_row, text="Factory Reset", fg_color="transparent", border_color=THEME["error"],
                      border_width=1, text_color=THEME["error"], hover_color=THEME["error"],
                      command=self.reset_all).pack(side="right")

        # Log Viewer
        ctk.CTkLabel(self.content, text="Latest Log Preview", font=(
            "Segoe UI", 14, "bold"), pady=10).pack(anchor="w")
        log_box = ctk.CTkTextbox(
            self.content, height=250, fg_color="#111", text_color="#DDD", font=("Consolas", 10))
        log_box.pack(fill="x")
        self.load_log_preview(log_box)

    def build_about(self):
        card = ctk.CTkFrame(self.content, fg_color=THEME["card_bg"])
        card.pack(fill="x", pady=10)

        # Hero
        ctk.CTkLabel(card, text="WARLOCK STUDIO", font=(
            "Impact", 42), text_color=THEME["accent"]).pack(pady=(30, 0))
        ctk.CTkLabel(card, text="Professional Media Suite", font=(
            "Segoe UI", 12, "bold"), text_color=THEME["text_main"]).pack()
        ctk.CTkLabel(card, text=f"v{self.version} | Build 2024", font=(
            "Consolas", 10), text_color=THEME["text_dim"]).pack(pady=5)

        ctk.CTkButton(card, text="Visit GitHub", fg_color="#24292e", hover_color="#000", width=120,
                      command=lambda: webbrowser.open(f"https://github.com/{self.owner}/{self.repo}")).pack(pady=20)

        # Update
        self.btn_upd = ctk.CTkButton(self.content, text="Check for Updates", height=45, fg_color=THEME["border"],
                                     hover_color=THEME["success"], command=self.check_updates)
        self.btn_upd.pack(fill="x", pady=10)

    # =========================================================================
    # HELPERS & LOGIC
    # =========================================================================

    def sub_header(self, text, pady=5):
        ctk.CTkLabel(self.content, text=text, font=("Segoe UI", 14, "bold"),
                     text_color=THEME["accent"]).pack(anchor="w", pady=(pady, 5))

    def create_group(self):
        f = ctk.CTkFrame(self.content, fg_color=THEME["card_bg"])
        f.pack(fill="x", pady=(0, 20))
        return f

    def add_switch(self, parent, text, key, desc=None, cmd=None):
        r = SettingRow(parent, text, desc)
        sw = ctk.CTkSwitch(
            parent, text="", progress_color=THEME["accent"], onvalue=True, offvalue=False)

        def callback():
            val = sw.get()
            self.config[key] = bool(val)
            ConfigManager.save_config(self.config)
            if cmd:
                cmd(val)

        sw.configure(command=callback)
        if self.config.get(key, False):
            sw.select()
        r.add_widget(sw)

    def create_tool_btn(self, parent, text, cmd, color, text_col="#FFF"):
        ctk.CTkButton(parent, text=text, command=cmd, fg_color=color,
                      text_color=text_col, width=120).pack(side="left", padx=5)

    def update_opacity(self, val):
        v = float(val)
        self.master.winfo_toplevel().attributes("-alpha", v)
        self.attributes("-alpha", v)
        self.config["window_opacity"] = v

    def set_ui_scaling(self, value):
        scale_map = {"80%": 0.8, "90%": 0.9, "100%": 1.0,
                     "110%": 1.1, "125%": 1.2, "150%": 1.5}
        new = scale_map.get(value, 1.0)
        ctk.set_widget_scaling(new)
        self.save("ui_scaling", new)

    def save(self, key, value, callback=None):
        self.config[key] = value
        ConfigManager.save_config(self.config)
        if callback:
            callback()

    def save_debounced(self, key, value):
        self.config[key] = value

    def save_dir(self, path):
        self.config["default_output_dir"] = path
        ConfigManager.save_config(self.config)
        self.ent_dir.configure(state="normal")
        self.ent_dir.delete(0, "end")
        self.ent_dir.insert(0, path)
        self.ent_dir.configure(state="readonly")

    def browse_output_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.save_dir(p)

    def create_path_selector(self, parent, title, config_key, file_filter):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(f, text=title, font=(
            "Segoe UI", 12, "bold")).pack(anchor="w")

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", pady=(5, 0))

        ent = ctk.CTkEntry(
            row, fg_color=THEME["input_bg"], border_color=THEME["border"])
        ent.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ent.insert(0, self.config.get(config_key, ""))

        def pick():
            p = filedialog.askopenfilename(
                filetypes=[("Executables", "*.exe"), ("All", "*.*")])
            if p:
                ent.delete(0, "end")
                ent.insert(0, p)
                self.save(config_key, p)

        ctk.CTkButton(row, text="Browse", width=70,
                      fg_color=THEME["border"], command=pick).pack(side="left")

    # --- SYSTEM INFO & LOGS ---
    def _run_hw_scan(self):
        self.cached_hw_info = HardwareScanner.get_system_info()
        self.after(0, lambda: self._display_hw_info(self.cached_hw_info))

    def _display_hw_info(self, hw):
        if not hasattr(self, "hw_frame") or not self.hw_frame.winfo_exists():
            return
        for w in self.hw_frame.winfo_children():
            w.destroy()

        grid = ctk.CTkFrame(self.hw_frame, fg_color="transparent")
        grid.pack(padx=15, pady=10, fill="x")

        data = [("OS", hw['os']), ("CPU", hw['cpu']), ("RAM",
                                                       f"{hw['ram_used']} / {hw['ram_total']} GB"), ("GPU", hw['gpu']), ("VRAM", f"{hw['vram']} GB")]

        for i, (k, v) in enumerate(data):
            ctk.CTkLabel(grid, text=k+":", font=("Segoe UI", 11, "bold"),
                         text_color=THEME["text_dim"]).grid(row=i//2, column=(i % 2)*2, sticky="w", padx=(0, 5))
            ctk.CTkLabel(grid, text=v, font=("Segoe UI", 11), text_color=THEME["text_main"]).grid(
                row=i//2, column=(i % 2)*2+1, sticky="w", padx=(0, 25))

    def load_log_preview(self, textbox):
        base = ConfigManager.get_base_path()
        logs = glob.glob(os.path.join(base, "*_Logs", "*.log"))
        if logs:
            latest = max(logs, key=os.path.getmtime)
            try:
                with open(latest, "r", encoding="utf-8", errors="ignore") as f:
                    textbox.insert("0.0", f.read())
            except:
                textbox.insert("0.0", "Error reading log.")
        else:
            textbox.insert("0.0", "No logs found.")

    # --- ACTIONS ---
    def open_logs_dir(self):
        base = ConfigManager.get_base_path()
        d = glob.glob(os.path.join(base, "*_Logs"))
        if d:
            os.startfile(d[0]) if platform.system(
            ) == "Windows" else webbrowser.open(d[0])
        else:
            messagebox.showinfo("Info", "No log folder yet.")

    def clean_temp_files(self):
        if messagebox.askyesno("Clean", "Delete temporary files?"):
            c = 0
            for p in ["*.tmp", "*.part", "temp_*"]:
                for f in glob.glob(os.path.join(ConfigManager.get_base_path(), p)):
                    try:
                        os.remove(f)
                        c += 1
                    except:
                        pass
            messagebox.showinfo("Done", f"Removed {c} files.")

    def export_debug_info(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".zip", initialfile=f"Debug_{datetime.now().strftime('%Y%m%d')}.zip")
        if path:
            with zipfile.ZipFile(path, 'w') as z:
                cfg = ConfigManager.get_config_path()
                if os.path.exists(cfg):
                    z.write(cfg, "config.json")
            messagebox.showinfo("Export", "Debug info saved.")

    def reset_all(self):
        if messagebox.askyesno("RESET", "Reset ALL settings to default? App will close."):
            ConfigManager.reset_config()
            self.master.destroy()
            sys.exit()

    def check_updates(self):
        self.btn_upd.configure(text="Checking...", state="disabled")
        threading.Thread(target=self._update_thread, daemon=True).start()

    def _update_thread(self):
        um = UpdateManager(self.owner, self.repo, self.version)
        is_new, tag, url, _ = um.check()

        def cb():
            self.btn_upd.configure(state="normal")
            if is_new:
                self.btn_upd.configure(
                    text=f"Update Available: {tag}", fg_color=THEME["success"])
                if messagebox.askyesno("Update", "Open download page?"):
                    webbrowser.open(url)
            else:
                self.btn_upd.configure(
                    text="Up to date", fg_color=THEME["border"])
        self.after(0, cb)

# =============================================================================
# BOTÓN DE INTEGRACIÓN (MAIN UI)
# =============================================================================


class PreferencesButton(ctk.CTkButton):
    def __init__(self, master, current_version, repo_owner="Ivan-Ayub97", repo_name="Warlock-Studio", **kwargs):
        super().__init__(master, text="⚙", width=40, height=30,
                         fg_color=THEME["bg_sidebar"], border_color=THEME["border"], border_width=1,
                         hover_color=THEME["accent"], text_color=THEME["text_main"],
                         command=self.open_window, **kwargs)
        self.ver = current_version
        self.repo = (repo_owner, repo_name)
        self.after(500, self._apply_startup)

    def open_window(self):
        for w in self.winfo_toplevel().winfo_children():
            if isinstance(w, PreferencesWindow):
                w.lift()
                w.focus_force()
                return
        PreferencesWindow(self.winfo_toplevel(), self.ver, *self.repo)

    def _apply_startup(self):
        c = ConfigManager.load_config()
        root = self.winfo_toplevel()

        # Tema y Escala
        try:
            ctk.set_appearance_mode(c.get("app_theme", "Dark"))
        except:
            pass
        try:
            ctk.set_widget_scaling(c.get("ui_scaling", 1.0))
        except:
            pass

        # Atributos de Ventana
        # CORRECCIÓN: Aplicamos el valor directamente (sea True o False)
        # Esto fuerza a la ventana a bajar si la opción está desactivada,
        # limpiando el estado heredado del Splash Screen.
        should_be_topmost = c.get("keep_window_on_top", False)
        root.attributes("-topmost", should_be_topmost)

        # Opacidad
        if 0.5 <= c.get("window_opacity", 1.0) <= 1.0:
            root.attributes("-alpha", c["window_opacity"])

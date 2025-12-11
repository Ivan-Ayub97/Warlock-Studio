import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
# GUI Imports
import tkinter as tk
import webbrowser
import zipfile
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional

import customtkinter as ctk
# Hardware Info Imports
import psutil
import requests
from packaging import version as pkg_version

# Optional Imports Handling
try:
    import wmi
except ImportError:
    wmi = None
try:
    import GPUtil
except ImportError:
    GPUtil = None

# =============================================================================
# CONSTANTES Y TEMA VISUAL (WARLOCK IDENTITY)
# =============================================================================

CONFIG_FILE = "warlock_config.json"

THEME = {
    "bg_main": "#121212",        # Fondo principal muy oscuro
    "bg_sidebar": "#1E1E1E",     # Fondo barra lateral
    "card_bg": "#252525",        # Fondo de tarjetas
    "card_hover": "#2F2F2F",     # Hover tarjetas
    "text_main": "#FFFFFF",      # Texto blanco puro
    "text_dim": "#A0A0A0",       # Texto secundario
    "accent": "#E63946",         # Rojo Warlock Vibrante
    "accent_hover": "#C92A35",   # Rojo oscuro hover
    "gold": "#FDEF2F",           # Dorado (Highlights)
    "success": "#00E676",        # Verde éxito
    "error": "#CF6679",          # Rojo error
    "border": "#333333",         # Bordes sutiles
    "input_bg": "#151515"        # Fondo de inputs
}

FONTS = {
    "header": ("Segoe UI", 24, "bold"),
    "subheader": ("Segoe UI", 16, "bold"),
    "label": ("Segoe UI", 12),
    "label_bold": ("Segoe UI", 12, "bold"),
    "small": ("Segoe UI", 10),
    "mono": ("Consolas", 11)
}

# =============================================================================
# GESTOR DE CONFIGURACIÓN (CENTRALIZADO)
# =============================================================================


class ConfigManager:
    # Configuración por defecto expandida
    DEFAULT_CONFIG = {
        # General
        "app_theme": "Dark",
        "ui_scaling": 1.0,
        "window_opacity": 1.0,
        "keep_window_on_top": False,
        "check_updates_on_startup": True,

        # Rendimiento
        "process_priority": "Normal",
        "onnx_provider_preference": "Auto",  # Auto, CUDA, DirectML, CPU
        "auto_close_on_finish": False,
        "max_cpu_threads": 0,  # 0 = Auto

        # Integraciones (Rutas manuales)
        "custom_ffmpeg_path": "",
        "custom_exiftool_path": "",

        # Logs
        "extended_logging": False
    }

    @staticmethod
    def get_base_path():
        """Obtiene la ruta base de instalación (funciona en .py y .exe)."""
        if getattr(sys, 'frozen', False):
            # Si es un ejecutable (PyInstaller), usa la carpeta del .exe
            return os.path.dirname(sys.executable)
        else:
            # Si es script, usa la carpeta del script
            return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def get_assets_path():
        """Obtiene la ruta de assets compatible con PyInstaller --onefile"""
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(
            os.path.abspath(__file__)))
        return os.path.join(base_dir, "Assets")

    @staticmethod
    def get_config_path():
        return os.path.join(ConfigManager.get_base_path(), CONFIG_FILE)

    @staticmethod
    def load_config() -> Dict[str, Any]:
        path = ConfigManager.get_config_path()
        config = ConfigManager.DEFAULT_CONFIG.copy()

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge seguro
                    for k, v in data.items():
                        # Permitimos nuevas claves si el usuario edita manualmente,
                        # pero priorizamos la estructura existente
                        config[k] = v
            except Exception as e:
                print(f"[Config] Error loading config: {e}")

        return config

    @staticmethod
    def save_config(config_data: Dict[str, Any]):
        try:
            path = ConfigManager.get_config_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"[Config] Error saving config: {e}")

    @staticmethod
    def reset_config():
        ConfigManager.save_config(ConfigManager.DEFAULT_CONFIG)
        return ConfigManager.DEFAULT_CONFIG.copy()

# =============================================================================
# HARDWARE SCANNER (NEO ENGINE OPTIMIZED)
# =============================================================================


class HardwareScanner:
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        info = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "ram_total": 0,
            "ram_used": 0,
            "gpu": "Unknown",
            "vram": 0
        }

        try:
            # RAM
            mem = psutil.virtual_memory()
            info["ram_total"] = round(mem.total / (1024**3), 2)
            info["ram_used"] = round(mem.used / (1024**3), 2)

            # CPU Name Cleanup
            if wmi and platform.system() == "Windows":
                try:
                    c = wmi.WMI()
                    cpu = c.Win32_Processor()[0]
                    info["cpu"] = cpu.Name.strip()
                except:
                    pass
            elif platform.system() == "Darwin":
                info["cpu"] = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode('utf-8')

            # GPU (Prioridad: NVIDIA > AMD/Intel Dedicada > Integrada)
            gpu_found = False

            # 1. GPUtil (NVIDIA)
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        info["gpu"] = gpus[0].name
                        info["vram"] = round(gpus[0].memoryTotal / 1024, 2)
                        gpu_found = True
                except:
                    pass

            # 2. WMI (Fallback Windows)
            if not gpu_found and wmi and platform.system() == "Windows":
                try:
                    c = wmi.WMI()
                    best_vram = -1
                    for g in c.Win32_VideoController():
                        v_bytes = 0
                        try:
                            # Algunas implementaciones WMI devuelven enteros negativos para valores altos
                            v_bytes = int(g.AdapterRAM)
                            if v_bytes < 0:
                                v_bytes += 2**32
                        except:
                            continue

                        v_gb = round(v_bytes / (1024**3), 2)
                        # Buscamos la GPU con más VRAM (generalmente la dedicada)
                        if v_gb > best_vram:
                            best_vram = v_gb
                            info["gpu"] = g.Name
                            info["vram"] = v_gb
                except:
                    pass

        except Exception as e:
            print(f"Hardware scan error: {e}")
            info["os"] = f"Error scanning: {str(e)}"

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

                # Comparación semántica de versiones
                if pkg_version.parse(tag) > pkg_version.parse(self.current):
                    return True, tag, data.get("html_url"), data.get("body")
            return False, self.current, None, None
        except Exception as e:
            return None, str(e), None, None

# =============================================================================
# COMPONENTES UI (WIDGETS)
# =============================================================================


class SidebarButton(ctk.CTkButton):
    def __init__(self, master, text, icon_char, command, is_selected=False):
        color = THEME["card_bg"] if is_selected else "transparent"
        text_col = THEME["accent"] if is_selected else THEME["text_dim"]

        super().__init__(master, text=f"  {icon_char}   {text}", anchor="w",
                         fg_color=color, hover_color=THEME["card_hover"],
                         text_color=text_col, height=45, corner_radius=8,
                         font=FONTS["label_bold"], command=command)


class SettingRow(ctk.CTkFrame):
    def __init__(self, master, title, desc=None):
        super().__init__(master, fg_color="transparent")
        self.pack(fill="x", pady=10)

        lbl_frame = ctk.CTkFrame(self, fg_color="transparent")
        lbl_frame.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(lbl_frame, text=title, font=FONTS["label_bold"],
                     text_color=THEME["text_main"], anchor="w").pack(fill="x")

        if desc:
            ctk.CTkLabel(lbl_frame, text=desc, font=FONTS["small"],
                         text_color=THEME["text_dim"], anchor="w").pack(fill="x")

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

        # Configuración Ventana
        self.title("Preferences")
        self.geometry("750x500")
        self.minsize(850, 600)
        self.configure(fg_color=THEME["bg_main"])

        # Intentar centrar la ventana respecto al padre
        try:
            self.after(10, self.lift)
            self.after(10, self.focus_force)
        except:
            pass

        # Datos
        self.config = ConfigManager.load_config()
        self.version = version
        self.owner = owner
        self.repo = repo
        self.cached_hw_info = None  # Cache para no escanear constantemente

        # Layout Principal (Grid 2 columnas: Sidebar | Content)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_content_area()

        # Cargar pestaña inicial
        self.show_page("general")

        # Icono (Intentar cargar)
        try:
            assets = ConfigManager.get_assets_path()
            icon_path = os.path.join(assets, "logo.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        except:
            pass

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self, width=240, corner_radius=0, fg_color=THEME["bg_sidebar"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Logo / Título
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.pack(pady=30, padx=20, fill="x")

        ctk.CTkLabel(title_frame, text="Warlock-Studio", font=("Impact", 28),
                     text_color=THEME["accent"]).pack(anchor="w")
        ctk.CTkLabel(title_frame, text="CONFIG", font=("Arial", 10, "bold"),
                     text_color=THEME["text_dim"]).pack(anchor="w", pady=(0, 10))

        # Separador
        ctk.CTkFrame(self.sidebar, height=2, fg_color=THEME["border"]).pack(
            fill="x", padx=20, pady=(0, 20))

        # Botones de Navegación
        self.nav_btns = {}
        pages = [
            ("general", "General", "⚙"),
            ("performance", "Performance & AI", "🚀"),
            ("integrations", "Paths & Tools", "🔗"),
            ("logs", "Logs & Maintenance", "🛠"),
            ("about", "About & Update", "ℹ")
        ]

        for pid, text, icon in pages:
            btn = SidebarButton(self.sidebar, text, icon,
                                lambda p=pid: self.show_page(p))
            btn.pack(fill="x", padx=15, pady=5)
            self.nav_btns[pid] = btn

        # Footer Sidebar
        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.pack(side="bottom", pady=20, fill="x")
        ctk.CTkLabel(footer, text=f"v{self.version}", font=FONTS["small"],
                     text_color=THEME["text_dim"]).pack()

    def setup_content_area(self):
        self.content = ctk.CTkScrollableFrame(
            self, fg_color="transparent", corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    def show_page(self, page_id):
        # Actualizar estado botones sidebar
        for pid, btn in self.nav_btns.items():
            is_active = (pid == page_id)
            btn.configure(fg_color=THEME["card_bg"] if is_active else "transparent",
                          text_color=THEME["accent"] if is_active else THEME["text_dim"])

        # Limpiar contenido
        for widget in self.content.winfo_children():
            widget.destroy()

        # Construir página
        if page_id == "general":
            self.build_general()
        elif page_id == "performance":
            self.build_performance()
        elif page_id == "integrations":
            self.build_integrations()
        elif page_id == "logs":
            self.build_logs()
        elif page_id == "about":
            self.build_about()

    # =========================================================================
    # CONSTRUCTORES DE PÁGINAS
    # =========================================================================

    def header(self, text):
        ctk.CTkLabel(self.content, text=text, font=FONTS["header"],
                     text_color=THEME["text_main"]).pack(anchor="w", pady=(0, 20))

    def build_general(self):
        self.header("General Settings")

        # UI Appearance
        grp = ctk.CTkFrame(self.content, fg_color=THEME["card_bg"])
        grp.pack(fill="x", pady=10, padx=5)

        # Theme
        r1 = SettingRow(grp, "App Theme",
                        "Select the visual style of the application")
        om = ctk.CTkOptionMenu(grp, values=["Dark", "Light", "System"],
                               command=lambda v: self.save(
                                   "app_theme", v, lambda: ctk.set_appearance_mode(v)),
                               fg_color=THEME["input_bg"], button_color=THEME["accent"], width=150)
        om.set(self.config.get("app_theme", "Dark"))
        r1.add_widget(om)

        # UI Scaling
        r_scale = SettingRow(
            grp, "UI Scaling", "Adjust the size of the interface elements")
        om_scale = ctk.CTkOptionMenu(grp, values=["80%", "90%", "100%", "110%", "125%"],
                                     command=self.set_ui_scaling,
                                     fg_color=THEME["input_bg"], button_color=THEME["accent"], width=150)
        current_scale = f"{int(self.config.get('ui_scaling', 1.0) * 100)}%"
        om_scale.set(current_scale)
        r_scale.add_widget(om_scale)

        # Opacity
        r2 = SettingRow(grp, "Window Opacity",
                        "Adjust transparency (0.5 - 1.0)")

        def update_opacity(val):
            # Convertir a float porque el slider devuelve float pero a veces muchos decimales
            v = float(val)
            self.master.attributes("-alpha", v)  # Aplica al padre
            self.attributes("-alpha", v)  # Aplica a esta ventana también
            self.config["window_opacity"] = v

        sl = ctk.CTkSlider(grp, from_=0.5, to=1.0, number_of_steps=50,
                           command=update_opacity,
                           progress_color=THEME["accent"], button_color=THEME["gold"])
        sl.set(self.config.get("window_opacity", 1.0))
        # Bind release para guardar configuración y no saturar disco
        sl.bind("<ButtonRelease-1>",
                lambda event: ConfigManager.save_config(self.config))
        r2.add_widget(sl)

        # Toggles
        # Usamos winfo_toplevel() para asegurar que obtenemos la ventana raíz
        root_win = self.master.winfo_toplevel()

        r3 = SettingRow(grp, "Always on Top",
                        "Keep the main window above others")
        sw1 = ctk.CTkSwitch(grp, text="", command=lambda: self.save_bool("keep_window_on_top", sw1, lambda: root_win.attributes("-topmost", sw1.get())),
                            progress_color=THEME["accent"], button_color="white")
        if self.config.get("keep_window_on_top", False):
            sw1.select()
        r3.add_widget(sw1)

        r4 = SettingRow(grp, "Check Updates on Startup",
                        "Automatically check Github for new versions")
        sw2 = ctk.CTkSwitch(grp, text="", command=lambda: self.save_bool("check_updates_on_startup", sw2),
                            progress_color=THEME["accent"])
        if self.config.get("check_updates_on_startup", True):
            sw2.select()
        r4.add_widget(sw2)

        r5 = SettingRow(grp, "Auto-Close on Finish",
                        "Close app automatically when processing ends")
        sw3 = ctk.CTkSwitch(grp, text="", command=lambda: self.save_bool("auto_close_on_finish", sw3),
                            progress_color=THEME["accent"])
        if self.config.get("auto_close_on_finish", False):
            sw3.select()
        r5.add_widget(sw3)

    def set_ui_scaling(self, value):
        scale_map = {"80%": 0.8, "90%": 0.9,
                     "100%": 1.0, "110%": 1.1, "125%": 1.2}
        new_scale = scale_map.get(value, 1.0)
        ctk.set_widget_scaling(new_scale)
        self.save("ui_scaling", new_scale)

    def build_performance(self):
        self.header("Performance & AI")

        # Hardware Info Card
        self.hw_frame = ctk.CTkFrame(
            self.content, fg_color=THEME["card_bg"], border_color=THEME["accent"], border_width=1)
        self.hw_frame.pack(fill="x", pady=(0, 20))

        # Loading Label
        self.lbl_hw_loading = ctk.CTkLabel(
            self.hw_frame, text="Scanning Hardware...", text_color=THEME["text_dim"])
        self.lbl_hw_loading.pack(pady=20)

        # Iniciar scan en hilo para no congelar
        if self.cached_hw_info:
            self._display_hw_info(self.cached_hw_info)
        else:
            threading.Thread(target=self._run_hw_scan, daemon=True).start()

        # Settings
        grp = ctk.CTkFrame(self.content, fg_color=THEME["card_bg"])
        grp.pack(fill="x")

        # ONNX Provider
        r1 = SettingRow(grp, "ONNX Provider",
                        "Force specific AI execution backend (Restart Required)")
        om1 = ctk.CTkOptionMenu(grp, values=["Auto", "CUDA", "DirectML", "CPU"],
                                command=lambda v: self.save(
                                    "onnx_provider_preference", v),
                                fg_color=THEME["input_bg"], button_color=THEME["accent"])
        om1.set(self.config.get("onnx_provider_preference", "Auto"))
        r1.add_widget(om1)

        # Process Priority
        r2 = SettingRow(grp, "Process Priority",
                        "System priority for FFmpeg and AI threads")
        om2 = ctk.CTkOptionMenu(grp, values=["Normal", "Above Normal", "High"],
                                command=lambda v: self.save(
                                    "process_priority", v),
                                fg_color=THEME["input_bg"], button_color=THEME["accent"])
        om2.set(self.config.get("process_priority", "Normal"))
        r2.add_widget(om2)

    def _run_hw_scan(self):
        self.cached_hw_info = HardwareScanner.get_system_info()
        # Actualizar UI en main thread
        self.after(0, lambda: self._display_hw_info(self.cached_hw_info))

    def _display_hw_info(self, hw):

        # Seguridad: evitar error cuando el frame ya no existe
        if not hasattr(self, "hw_frame"):
            return

        if not str(self.hw_frame) or not self.hw_frame.winfo_exists():
            return

        for w in self.hw_frame.winfo_children():
            w.destroy()

        lbl_grid = ctk.CTkFrame(self.hw_frame, fg_color="transparent")
        lbl_grid.pack(padx=20, pady=15, fill="x")

        items = [
            ("OS:", hw['os']),
            ("CPU Model:", hw['cpu']),
            ("RAM Usage:", f"{hw['ram_used']} GB / {hw['ram_total']} GB"),
            ("GPU Model:", hw['gpu']),
            ("VRAM:", f"{hw['vram']} GB")
        ]

        for i, (k, v) in enumerate(items):
            ctk.CTkLabel(lbl_grid, text=k, font=FONTS["label_bold"], text_color=THEME["text_dim"]).grid(
                row=i//2, column=(i % 2)*2, sticky="w", padx=(0, 10), pady=2)
            ctk.CTkLabel(lbl_grid, text=v, font=FONTS["label"], text_color=THEME["text_main"]).grid(
                row=i//2, column=(i % 2)*2+1, sticky="w", padx=(0, 40), pady=2)

    def build_integrations(self):
        self.header("Paths & Tools")

        desc = ctk.CTkLabel(self.content, text="Manually configure paths for external tools if auto-detection fails. Leave empty to use built-in/default paths.",
                            text_color=THEME["text_dim"], font=FONTS["small"], wraplength=500, justify="left")
        desc.pack(anchor="w", pady=(0, 20))

        grp = ctk.CTkFrame(self.content, fg_color=THEME["card_bg"])
        grp.pack(fill="x")

        # FFmpeg
        self.create_path_selector(
            grp, "FFmpeg Executable", "custom_ffmpeg_path", "ffmpeg.exe")

        # ExifTool
        self.create_path_selector(
            grp, "ExifTool Executable", "custom_exiftool_path", "exiftool.exe")

    def create_path_selector(self, parent, title, config_key, file_filter):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(f, text=title, font=FONTS["label_bold"]).pack(anchor="w")

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", pady=(5, 0))

        entry = ctk.CTkEntry(
            row, fg_color=THEME["input_bg"], border_color=THEME["border"])
        entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        current_val = self.config.get(config_key, "")
        entry.insert(0, current_val)

        # Validación visual simple
        if current_val and not os.path.exists(current_val):
            entry.configure(text_color=THEME["error"])

        def pick():
            path = filedialog.askopenfilename(
                filetypes=[("Executables", "*.exe"), ("All Files", "*.*")])
            if path:
                entry.delete(0, "end")
                entry.insert(0, path)
                entry.configure(text_color=THEME["text_main"])
                self.save(config_key, path)

        def clear():
            entry.delete(0, "end")
            self.save(config_key, "")
            entry.configure(text_color=THEME["text_main"])

        btn_pick = ctk.CTkButton(
            row, text="Browse", width=80, fg_color=THEME["border"], command=pick)
        btn_pick.pack(side="left", padx=5)

        btn_clear = ctk.CTkButton(
            row, text="✖", width=30, fg_color=THEME["bg_main"], hover_color=THEME["accent"], command=clear)
        btn_clear.pack(side="left", padx=(5, 0))

    def build_logs(self):
        self.header("Logs & Maintenance")

        grp = ctk.CTkFrame(self.content, fg_color=THEME["card_bg"])
        grp.pack(fill="x", pady=10)

        # Actions Row
        act_row = ctk.CTkFrame(grp, fg_color="transparent")
        act_row.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(act_row, text="Open Logs Folder", fg_color=THEME["border"],
                      command=self.open_logs_dir).pack(side="left", padx=5)

        ctk.CTkButton(act_row, text="Clean Temp Files", fg_color=THEME["accent"],
                      command=self.clean_temp_files).pack(side="left", padx=5)

        ctk.CTkButton(act_row, text="Export Debug Zip", fg_color="#444",
                      command=self.export_debug_info).pack(side="left", padx=5)

        ctk.CTkButton(act_row, text="Reset Config", fg_color="transparent", border_color="#FF0000", border_width=1, text_color="#FF0000",
                      command=self.reset_all).pack(side="right", padx=5)

        # Log Viewer
        ctk.CTkLabel(grp, text="Latest Log Preview:",
                     font=FONTS["label_bold"]).pack(anchor="w", padx=10, pady=(10, 0))

        log_box = ctk.CTkTextbox(
            grp, height=300, fg_color=THEME["input_bg"], font=FONTS["mono"])
        log_box.pack(fill="x", padx=10, pady=10)

        # Load Log Content
        base = ConfigManager.get_base_path()
        log_folder_pattern = os.path.join(base, "*_Logs")
        log_folders = glob.glob(log_folder_pattern)

        if log_folders:
            log_folder = log_folders[0]
            log_path_pattern = os.path.join(log_folder, "*.log")
            files = glob.glob(log_path_pattern)

            if files:
                latest = max(files, key=os.path.getmtime)
                try:
                    with open(latest, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        log_box.insert("0.0", content)
                        log_box.see("end")  # Auto scroll al final
                except Exception as e:
                    log_box.insert("0.0", f"Error reading log file: {e}")
            else:
                log_box.insert(
                    "0.0", "Log folder found, but no .log files present.")
        else:
            log_box.insert("0.0", "No logs folder found.")

    def build_about(self):
        self.header("About")

        card = ctk.CTkFrame(self.content, fg_color=THEME["card_bg"])
        card.pack(fill="x", pady=10)

        ctk.CTkLabel(card, text="Warlock-Studio", font=("Impact", 48),
                     text_color=THEME["accent"]).pack(pady=(30, 5))
        ctk.CTkLabel(
            card, text=f"Version {self.version}", font=FONTS["label_bold"]).pack()
        ctk.CTkLabel(card, text="Developed by Ivan-Ayub97",
                     font=FONTS["small"], text_color=THEME["text_dim"]).pack(pady=(0, 20))

        # Buttons
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(pady=20)

        ctk.CTkButton(btn_row, text="GitHub Repository", fg_color="#333", width=140,
                      command=lambda: webbrowser.open(f"https://github.com/{self.owner}/{self.repo}")).pack(side="left", padx=10)

        # License View
        lbl_lic = ctk.CTkLabel(card, text="LICENSE AGREEMENT",
                               font=FONTS["label_bold"], text_color=THEME["text_dim"])
        lbl_lic.pack(pady=(10, 5))

        # Intento robusto de cargar licencia
        l_txt = "License file not found."
        try:
            # Opción 1: Assets/license.txt
            assets_path = ConfigManager.get_assets_path()
            lic_path = os.path.join(assets_path, "license.txt")
            if not os.path.exists(lic_path):
                # Opción 2: Raíz/license.txt
                lic_path = os.path.join(
                    ConfigManager.get_base_path(), "license.txt")

            if os.path.exists(lic_path):
                with open(lic_path, "r", encoding="utf-8") as f:
                    l_txt = f.read()
        except:
            pass

        ld = ctk.CTkTextbox(card, height=150, fg_color="#111111",
                            text_color=THEME["text_dim"], font=("Consolas", 10))
        ld.pack(fill="x", pady=10, padx=20)
        ld.insert("0.0", l_txt)
        ld.configure(state="disabled")

        # Update Checker
        upd_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        upd_frame.pack(fill="x", pady=20)

        self.btn_upd = ctk.CTkButton(upd_frame, text="Check for Updates", height=50, font=FONTS["subheader"],
                                     fg_color=THEME["border"], hover_color=THEME["success"], command=self.check_updates)
        self.btn_upd.pack(fill="x")

    # =========================================================================
    # LOGICA DE ACCIONES
    # =========================================================================

    def save(self, key, value, callback=None):
        self.config[key] = value
        ConfigManager.save_config(self.config)
        if callback:
            callback()

    def save_bool(self, key, switch_widget, callback=None):
        val = bool(switch_widget.get())
        self.save(key, val, callback)

    def open_logs_dir(self):
        base = ConfigManager.get_base_path()
        log_dirs = glob.glob(os.path.join(base, "*_Logs"))
        target = log_dirs[0] if log_dirs else base

        try:
            if platform.system() == "Windows":
                os.startfile(target)
            else:
                webbrowser.open(target)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")

    def clean_temp_files(self):
        if not messagebox.askyesno("Confirm Cleanup", "Are you sure you want to delete all temporary and intermediate files? This action cannot be undone."):
            return

        base = ConfigManager.get_base_path()
        patterns = ["*.tmp", "*.checkpoint", "*.part", "temp_*",
                    "*_frames.txt", "Pipeline_temp.json", "warlock_config.json.bak"]
        count = 0

        for p in patterns:
            for f in glob.glob(os.path.join(base, p)):
                try:
                    os.remove(f)
                    count += 1
                except:
                    pass

        messagebox.showinfo("Cleanup", f"Deleted {count} temporary files.")

    def export_debug_info(self):
        """Genera un ZIP con logs y config para facilitar reporte de bugs."""
        save_path = filedialog.asksaveasfilename(defaultextension=".zip",
                                                 initialfile=f"Warlock_Debug_{datetime.now().strftime('%Y%m%d')}.zip")
        if not save_path:
            return

        try:
            base = ConfigManager.get_base_path()
            with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Agregar Config
                cfg_path = os.path.join(base, CONFIG_FILE)
                if os.path.exists(cfg_path):
                    zf.write(cfg_path, CONFIG_FILE)

                # Agregar Logs recientes
                log_dirs = glob.glob(os.path.join(base, "*_Logs"))
                if log_dirs:
                    for f in glob.glob(os.path.join(log_dirs[0], "*.log")):
                        zf.write(f, os.path.join("Logs", os.path.basename(f)))

            messagebox.showinfo("Export Successful",
                                f"Debug info saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def reset_all(self):
        if messagebox.askyesno("Reset", "Are you sure you want to reset ALL preferences to default? The application will restart."):
            ConfigManager.reset_config()
            messagebox.showinfo(
                "Reset", "Configuration reset. The application will now close.")
            self.master.destroy()
            sys.exit(0)

    def check_updates(self):
        self.btn_upd.configure(text="Checking...", state="disabled")
        threading.Thread(target=self._update_thread, daemon=True).start()

    def _update_thread(self):
        um = UpdateManager(self.owner, self.repo, self.version)
        is_new, tag, url, body = um.check()

        def ui_cb():
            self.btn_upd.configure(state="normal")
            if is_new is True:
                self.btn_upd.configure(
                    text=f"Update Available: {tag}", fg_color=THEME["success"], hover_color="#00C853")
                if messagebox.askyesno("Update Found", f"Version {tag} is available.\n\nOpen download page?"):
                    webbrowser.open(url)
            elif is_new is False:
                self.btn_upd.configure(
                    text="You are up to date", fg_color=THEME["border"], hover_color=THEME["card_hover"])
            else:
                self.btn_upd.configure(
                    text=f"Error checking updates: {tag}", fg_color=THEME["accent"], hover_color=THEME["accent_hover"])

        self.after(0, ui_cb)

# =============================================================================
# BOTÓN DE ACCESO (INTEGRACIÓN)
# =============================================================================


class PreferencesButton(ctk.CTkButton):
    def __init__(self, master, current_version, repo_owner="Ivan-Ayub97", repo_name="Warlock-Studio", **kwargs):
        super().__init__(master, text="⚙", width=40, height=28,
                         fg_color=THEME["bg_sidebar"], border_color=THEME["border"], border_width=1,
                         hover_color=THEME["accent"], text_color=THEME["text_main"],
                         command=self.open_window, **kwargs)
        self.ver = current_version
        self.repo_data = (repo_owner, repo_name)

        # Aplicar configuraciones iniciales
        # Usamos after para asegurar que el widget esté cargado en la jerarquía
        self.after(100, self._apply_startup)

    def open_window(self):
        # Evitar múltiples ventanas
        for w in self.winfo_toplevel().winfo_children():
            if isinstance(w, PreferencesWindow):
                w.lift()
                w.focus_force()
                return
        # Pasamos el toplevel actual como master
        PreferencesWindow(self.winfo_toplevel(), self.ver, *self.repo_data)

    def _apply_startup(self):
        """Aplica la configuración al iniciar la app de forma segura."""
        c = ConfigManager.load_config()

        # Obtener la ventana raíz real (no el frame donde está este botón)
        root = self.winfo_toplevel()

        # Tema
        try:
            ctk.set_appearance_mode(c.get("app_theme", "Dark"))
        except:
            pass

        # Escala
        try:
            ctk.set_widget_scaling(c.get("ui_scaling", 1.0))
        except:
            pass

        # Opacidad
        alpha = c.get("window_opacity", 1.0)
        if 0.1 <= alpha <= 1.0:
            root.attributes("-alpha", alpha)

        # Siempre Visible
        top = c.get("keep_window_on_top", False)
        root.attributes("-topmost", top)

import ctypes
import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
import zipfile
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional

import customtkinter as ctk
# --- IMPORTACIONES DEL MOTOR NEO (Hardware Info) ---
import psutil
import requests
from packaging import version as pkg_version

try:
    import wmi
except ImportError:
    wmi = None
try:
    import GPUtil
except ImportError:
    GPUtil = None

# -----------------------------------------------------------------------------
# CONSTANTES DE ESTILO & TEMAS
# -----------------------------------------------------------------------------
THEME = {
    "bg": "#1B1818",
    "widget_bg": "#2A2727",
    "card_bg": "#212121",
    "text": "#FFFFFF",
    "text_sec": "#CAC9C9",
    "accent": "#FDEF2F",      # Amarillo Dorado (Warlock)
    "accent_hover": "#D4C428",
    "title": "#FF3232",       # Rojo Warlock
    "hover": "#D41C1C",       # Rojo oscuro
    "border": "#E2340D",
    "success": "#00E676",
    "error": "#B00020",
    "warning": "#FFA000",
    "info_bg": "#1E3A8A",
    "scroll_bg": "#181818"
}

CONFIG_FILE = "warlock_config.json"

# -----------------------------------------------------------------------------
# UTILIDADES DE UI
# -----------------------------------------------------------------------------


def get_font(size: int, weight: str = "normal", family: str = "Segoe UI") -> tuple:
    """Genera una fuente base."""
    return (family, size, weight)

# -----------------------------------------------------------------------------
# GESTOR DE CONFIGURACIÓN
# -----------------------------------------------------------------------------


class ConfigManager:
    DEFAULT_CONFIG = {
        "check_updates_on_startup": True,
        "keep_window_on_top": False,
        "ui_scaling": 1.0,
        "font_scale": 1.0,
        "window_opacity": 1.0,
        "app_theme": "Dark",
        "process_priority": "Normal",
        "auto_close_on_finish": False,
        "auto_clean_temp": False,
        "notifications_enabled": True,
        "last_gpu_index": "Auto"
    }

    @staticmethod
    def get_config_path():
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, CONFIG_FILE)

    @staticmethod
    def load_config() -> Dict[str, Any]:
        path = ConfigManager.get_config_path()
        config = ConfigManager.DEFAULT_CONFIG.copy()

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k in config:
                            if isinstance(v, type(config[k])) or (isinstance(config[k], float) and isinstance(v, int)):
                                config[k] = v
            except Exception as e:
                print(f"[Config] Error loading/validating config: {e}")

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

# -----------------------------------------------------------------------------
# ESCÁNER DE HARDWARE (INTEGRACIÓN MOTOR NEO)
# -----------------------------------------------------------------------------


class HardwareScanner:
    @staticmethod
    def get_specs_detailed() -> Dict[str, Any]:
        """
        Utiliza el motor de detección de NEO (WMI/PSUTIL/GPUTIL)
        adaptado para devolver el diccionario que espera Warlock.
        """
        specs = {
            "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
            "cpu_name": platform.processor(),
            "cpu_cores": os.cpu_count() or 4,
            "ram_total": 0.0,
            "ram_used": 0.0,
            "gpu_name": "Integrated / Unknown",
            "gpu_vram": 0,
            "disk_free": 0.0,
        }

        try:
            # 1. CPU y SO (Motor NEO: WMI Processor)
            if wmi:
                try:
                    c = wmi.WMI()
                    # SO Info detallada
                    specs["os"] = f"{platform.system()} {platform.release()} {platform.machine()}"

                    # CPU Info detallada
                    proc_obj = c.Win32_Processor()[0]
                    specs["cpu_name"] = proc_obj.Name.strip()
                    specs["cpu_cores"] = proc_obj.NumberOfLogicalProcessors
                except Exception:
                    pass  # Fallback a los valores por defecto de platform

            # 2. RAM (Motor NEO: Psutil + WMI logic adaptation)
            mem = psutil.virtual_memory()
            specs["ram_total"] = round(mem.total / (1024**3), 2)
            specs["ram_used"] = round(mem.used / (1024**3), 2)

            # 3. Almacenamiento (Motor NEO logic)
            total_free_space = 0
            for p in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(p.mountpoint)
                    total_free_space += usage.free
                except PermissionError:
                    continue
            specs["disk_free"] = round(total_free_space / (1024**3), 2)

            # 4. GPU (Motor NEO: GPUtil Priority -> WMI Fallback)
            gpu_found = False

            # Intento A: GPUtil (NVIDIA)
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        best_gpu = gpus[0]
                        specs["gpu_name"] = f"[NVIDIA] {best_gpu.name}"
                        specs["gpu_vram"] = round(
                            best_gpu.memoryTotal / 1024, 2)  # GPUtil returns MB
                        gpu_found = True
                except Exception:
                    pass

            # Intento B: WMI (AMD / Intel / Fallback)
            if not gpu_found and wmi:
                try:
                    c = wmi.WMI()
                    for gpu in c.Win32_VideoController():
                        # Lógica simple para elegir la GPU dedicada si hay varias
                        name = gpu.Name
                        vram_bytes = 0
                        try:
                            # AdapterRAM suele devolver bytes, pero a veces es negativo en int32 overflow
                            vram_bytes = int(gpu.AdapterRAM)
                            if vram_bytes < 0:
                                vram_bytes += 2**32
                        except:
                            vram_bytes = 0

                        vram_gb = round(vram_bytes / (1024**3), 2)

                        # Preferir GPU con más VRAM o que no sea "Intel" si ya tenemos una integrada
                        current_best = specs["gpu_vram"]
                        # Si encontramos una con más VRAM o es la primera dedicada que vemos
                        if vram_gb > current_best or (vram_gb > 0 and "Intel" not in name and specs["gpu_name"] == "Integrated / Unknown"):
                            specs["gpu_name"] = name
                            specs["gpu_vram"] = vram_gb
                except Exception:
                    pass

        except Exception as e:
            print(f"[HardwareScanner] Error using NEO Engine: {e}")

        return specs

    @staticmethod
    def get_smart_recommendations(specs: Dict) -> Dict:
        rec = {}
        vram = specs.get("gpu_vram", 0)
        ram = specs.get("ram_total", 0)

        # Cálculo de seguridad Warlock
        safe_vram = max(0.5, vram - 1.5)
        rec_tiles = int(safe_vram * 3.5)
        if rec_tiles < 2:
            rec_tiles = 2

        rec["Recommended Tiles"] = f"{rec_tiles}"
        rec["Safe VRAM Limit"] = f"{safe_vram:.1f} GB"

        if vram >= 8 and ram >= 16:
            rec["AI Model Class"] = "High End (RealESRGANx4 / BSRGANx4)"
            rec["Multithreading"] = "2 - 4 Threads"
        elif vram >= 4:
            rec["AI Model Class"] = "Mid Range (RealESR_Animex4)"
            rec["Multithreading"] = "2 Threads"
        else:
            rec["AI Model Class"] = "Low End (RealESR_Gx4 / RIFE Lite)"
            rec["Multithreading"] = "OFF (1 Thread)"

        return rec

# -----------------------------------------------------------------------------
# UPDATE MANAGER
# -----------------------------------------------------------------------------


class UpdateManager:
    def __init__(self, owner, repo, version):
        self.api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        self.current_ver = version

    def check_update(self):
        try:
            r = requests.get(self.api_url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                tag = data.get("tag_name", "v0.0").lstrip("v")

                if pkg_version.parse(tag) > pkg_version.parse(self.current_ver):
                    return True, tag, data.get("assets", []), data.get("body", "No changelog provided.")
            return False, self.current_ver, None, None
        except Exception as e:
            return None, str(e), None, None

    def download(self, url, path, progress_callback):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                downloaded = 0
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total > 0:
                            progress_callback(downloaded / total)
            return path
        except Exception as e:
            raise e

# -----------------------------------------------------------------------------
# COMPONENTES UI PERSONALIZADOS (Blender Style)
# -----------------------------------------------------------------------------


class SettingCard(ctk.CTkFrame):
    def __init__(self, master, title, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.pack(fill="x", pady=(5, 0), padx=5)

        h = ctk.CTkFrame(self, fg_color="transparent")
        h.pack(fill="x", pady=(5, 5))

        ctk.CTkLabel(h, text="—", text_color=THEME["border"], font=(
            "Arial", 12, "bold")).pack(side="left", padx=(0, 5))
        ctk.CTkLabel(h, text=title.upper(), text_color=THEME["text_sec"], font=get_font(
            11, "bold")).pack(side="left")
        ctk.CTkFrame(self, height=1, fg_color=THEME["widget_bg"]).pack(
            fill="x", padx=0, pady=(0, 5))


class CollapsibleMenu(ctk.CTkFrame):
    def __init__(self, master, title, **kwargs):
        super().__init__(master,
                         fg_color=THEME["card_bg"], border_color=THEME["border"], border_width=1, **kwargs)
        self.pack(fill="x", pady=8, padx=15)
        self.grid_columnconfigure(0, weight=1)

        self.header_frame = ctk.CTkFrame(
            self, fg_color=THEME["widget_bg"], height=40, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.header_frame, text=title.upper(), font=get_font(
            13, "bold"), text_color=THEME["accent"], padx=15, anchor="w").grid(row=0, column=0, sticky="ew", pady=5)

        self.is_expanded = True
        self.toggle_button = ctk.CTkButton(self.header_frame, text="▼", width=30, height=30,
                                           fg_color=THEME["widget_bg"], hover_color=THEME["hover"],
                                           text_color=THEME["text"], command=self.toggle)
        self.toggle_button.grid(row=0, column=1, padx=10, pady=5)

        self.content_frame = ctk.CTkFrame(
            self, fg_color=THEME["card_bg"], corner_radius=0)
        self.content_frame.grid(
            row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.content_frame.grid_columnconfigure(0, weight=1)

    def toggle(self):
        if self.is_expanded:
            self.content_frame.grid_remove()
            self.toggle_button.configure(text="▶")
        else:
            self.content_frame.grid(
                row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
            self.toggle_button.configure(text="▼")
        self.is_expanded = not self.is_expanded


class DownloadWindow(ctk.CTkToplevel):
    def __init__(self, master, filename):
        super().__init__(master)
        self.title("Update Downloader")
        self.geometry("400x150")
        self.attributes("-topmost", True)
        self.configure(fg_color=THEME["bg"])

        try:
            x = master.winfo_x() + (master.winfo_width()//2) - 200
            y = master.winfo_y() + (master.winfo_height()//2) - 75
            self.geometry(f"+{x}+{y}")
        except:
            pass

        ctk.CTkLabel(self, text=f"Downloading {filename}...", text_color=THEME["accent"], font=get_font(
            12)).pack(pady=(30, 10))

        self.prog = ctk.CTkProgressBar(
            self, width=300, progress_color=THEME["success"])
        self.prog.pack(pady=10)
        self.prog.set(0)

        self.lbl = ctk.CTkLabel(self, text="0%", text_color=THEME["text_sec"])
        self.lbl.pack()

    def update_progress(self, val):
        self.prog.set(val)
        self.lbl.configure(text=f"{int(val*100)}%")
        self.update_idletasks()

# -----------------------------------------------------------------------------
# VENTANA PRINCIPAL DE PREFERENCIAS
# -----------------------------------------------------------------------------


class PreferencesWindow(ctk.CTkToplevel):
    def __init__(self, master, version, owner, repo):
        super().__init__(master)
        self.title("Warlock-Studio - Preferences")
        self.geometry("650x750")
        self.configure(fg_color=THEME["bg"])
        self.attributes("-topmost", True)

        try:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_dir, "Assets", "logo.ico")
            self.iconbitmap(icon_path)
        except:
            pass

        self.curr_ver = version
        self.upd_mgr = UpdateManager(owner, repo, version)
        self.config = ConfigManager.load_config()

        self.create_ui()
        self.after(100, lambda: self.focus_force())

    def create_ui(self):
        # HEADER
        head = ctk.CTkFrame(self, fg_color="transparent", height=80)
        head.pack(fill="x", padx=25, pady=15)

        t_frame = ctk.CTkFrame(head, fg_color="transparent")
        t_frame.pack(side="left")

        ctk.CTkLabel(t_frame, text="WARLOCK", font=("Impact", 36),
                     text_color=THEME["title"]).pack(side="left")
        ctk.CTkLabel(t_frame, text="- STUDIO", font=("Impact", 36),
                     text_color=THEME["text"]).pack(side="left")

        ver_badge = ctk.CTkLabel(head, text=f"v{self.curr_ver}", fg_color=THEME["widget_bg"],
                                 text_color=THEME["accent"], corner_radius=6, padx=12, font=get_font(12, "bold"))
        ver_badge.pack(side="right", padx=10)

        self.status_lbl = ctk.CTkLabel(
            head, text="", font=get_font(11, "bold"))
        self.status_lbl.pack(side="right", padx=10)

        # SCROLLABLE CONTENT
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color=THEME["bg"], scrollbar_button_color=THEME["border"],
                                                   label_text="SETTINGS MENU", label_text_color=THEME["text_sec"],
                                                   label_font=get_font(10, "bold"))
        self.scroll_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        self._build_general_menu()
        self._build_system_menu()
        self._build_updates_menu()
        self._build_logs_menu()
        self._build_about_menu()

        self.after(500, self._run_scan)

    # --- MENUS ---

    def _build_general_menu(self):
        menu = CollapsibleMenu(self.scroll_frame, "General Settings")
        frame = menu.content_frame

        c1 = SettingCard(frame, "USER INTERFACE")
        self._add_dropdown(frame, "Theme Mode", "app_theme", ["Dark", "Light", "System"],
                           lambda v: self._update("app_theme", v, lambda: ctk.set_appearance_mode(v)))
        self._add_slider(frame, "Window Opacity", "window_opacity", 0.5, 1.0, 50,
                         lambda v: self._update("window_opacity", v, lambda: self.master.attributes("-alpha", v)))
        self._add_slider(frame, "UI Scaling", "ui_scaling", 0.8, 1.5, 7,
                         lambda v: self._update("ui_scaling", v, lambda: ctk.set_widget_scaling(v)))

        c2 = SettingCard(frame, "BEHAVIOR")
        self._add_switch(frame, "Keep Window on Top", "keep_window_on_top",
                         lambda: self.master.attributes("-topmost", self.config["keep_window_on_top"]))
        self._add_switch(frame, "Check Updates on Startup", "check_updates_on_startup",
                         lambda: self._update("check_updates_on_startup", self.config["check_updates_on_startup"]))
        self._add_switch(frame, "Auto-Close App when Task Finishes", "auto_close_on_finish",
                         lambda: self._update("auto_close_on_finish", self.config["auto_close_on_finish"]))

        c3 = SettingCard(frame, "MAINTENANCE")
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=5, pady=10)
        ctk.CTkButton(row, text="Clean Temporary Files", fg_color="#4A0000", hover_color=THEME["error"], border_color=THEME["error"], border_width=1,
                      command=self._clean_temp).pack(side="left", expand=True, fill="x", padx=(0, 5))
        ctk.CTkButton(row, text="Reset All Settings", fg_color="transparent", border_color=THEME["warning"], border_width=1, text_color=THEME["warning"],
                      hover_color="#332200", command=self._reset_settings).pack(side="right", expand=True, fill="x", padx=(5, 0))

    def _build_system_menu(self):
        self.hw_menu = CollapsibleMenu(
            self.scroll_frame, "System & AI Configuration")
        frame = self.hw_menu.content_frame

        SettingCard(frame, "HARDWARE DIAGNOSTICS (NEO ENGINE)")
        self.hw_content = ctk.CTkFrame(frame, fg_color="transparent")
        self.hw_content.pack(fill="both", padx=10, pady=5)

        self.btn_scan = ctk.CTkButton(frame, text="Run Hardware Scan", fg_color=THEME["border"], hover_color=THEME["hover"],
                                      command=self._run_scan, height=35)
        self.btn_scan.pack(pady=15, padx=10)

        SettingCard(frame, "PROCESSING PRIORITY")
        self._add_dropdown(frame, "FFmpeg/AI Priority", "process_priority", ["Normal", "Above Normal", "High"],
                           lambda v: self._update("process_priority", v))

    def _run_scan(self):
        self.btn_scan.configure(state="disabled", text="Scanning System...")
        for widget in self.hw_content.winfo_children():
            widget.destroy()
        ctk.CTkLabel(self.hw_content, text="Scanning with NEO Engine... Please wait.",
                     text_color=THEME["text_sec"]).pack(pady=20)
        threading.Thread(target=self._thread_scan, daemon=True).start()

    def _thread_scan(self):
        specs = HardwareScanner.get_specs_detailed()
        recs = HardwareScanner.get_smart_recommendations(specs)
        if self.winfo_exists():
            self.after(0, lambda: self._render_specs(specs, recs))

    def _render_specs(self, specs, recs):
        if not self.winfo_exists():
            return
        self.btn_scan.configure(state="normal", text="Refresh Hardware Info")
        for widget in self.hw_content.winfo_children():
            widget.destroy()

        grid = ctk.CTkFrame(self.hw_content, fg_color="transparent")
        grid.pack(fill="x")
        grid.grid_columnconfigure(1, weight=1)

        items = [("CPU", specs['cpu_name']), ("Cores", f"{specs['cpu_cores']} Logical Cores"),
                 ("RAM",
                  f"{specs['ram_total']} GB (Used: {specs['ram_used']} GB)"),
                 ("GPU", specs['gpu_name']), ("VRAM", f"{specs['gpu_vram']} GB"), ("OS", specs['os'])]

        for i, (label, val) in enumerate(items):
            ctk.CTkLabel(grid, text=label, font=get_font(11, "bold"), text_color=THEME["text_sec"], anchor="w").grid(
                row=i, column=0, sticky="w", pady=4, padx=5)
            ctk.CTkLabel(grid, text=str(val), font=get_font(11), text_color=THEME["accent"], anchor="w").grid(
                row=i, column=1, sticky="w", padx=20, pady=4)

        rec_frame = ctk.CTkFrame(
            self.hw_content, fg_color="#1A1A1A", corner_radius=6)
        rec_frame.pack(fill="x", padx=10, pady=20)
        ctk.CTkLabel(rec_frame, text="RECOMMENDED AI SETTINGS",
                     text_color=THEME["success"], font=get_font(11, "bold")).pack(pady=(10, 10))

        rec_grid = ctk.CTkFrame(rec_frame, fg_color="transparent")
        rec_grid.pack(pady=(0, 10), padx=10)
        r_items = [("Tiles Resolution", recs.get("Recommended Tiles", "N/A")),
                   ("VRAM Limit", recs.get("Safe VRAM Limit", "N/A")),
                   ("AI Threads", recs.get("Multithreading", "OFF"))]
        for i, (k, v) in enumerate(r_items):
            ctk.CTkLabel(rec_grid, text=k + ": ", text_color=THEME["text_sec"], anchor="e").grid(
                row=i, column=0, padx=(10, 5), pady=2, sticky="e")
            ctk.CTkLabel(rec_grid, text=v, text_color="#FFFFFF", font=get_font(
                12, "bold"), anchor="w").grid(row=i, column=1, padx=(0, 10), pady=2, sticky="w")

    def _build_updates_menu(self):
        menu = CollapsibleMenu(self.scroll_frame, "Updates & Changelog")
        frame = menu.content_frame
        v_frame = ctk.CTkFrame(frame, fg_color="transparent")
        v_frame.pack(pady=15)
        ctk.CTkLabel(v_frame, text="CURRENT VERSION", font=get_font(
            10), text_color=THEME["text_sec"]).pack()
        ctk.CTkLabel(v_frame, text=f"v{self.curr_ver}", font=(
            "Impact", 42), text_color=THEME["accent"]).pack()
        self.btn_upd = ctk.CTkButton(frame, text="Check for Updates", height=45,
                                     fg_color=THEME["border"], hover_color=THEME["hover"], font=get_font(14, "bold"), command=self._check_update)
        self.btn_upd.pack(fill="x", padx=60, pady=(0, 20))
        ctk.CTkLabel(frame, text="CHANGELOG / STATUS:", font=get_font(11, "bold"),
                     text_color=THEME["text_sec"], anchor="w").pack(fill="x", pady=(10, 5), padx=5)
        self.upd_log = ctk.CTkTextbox(frame, height=200, fg_color="#111111", text_color=THEME["text"], font=(
            "Consolas", 11), border_width=1, border_color=THEME["widget_bg"])
        self.upd_log.pack(fill="x", expand=True, pady=5, padx=5)
        self.upd_log.insert(
            "0.0", "Click 'Check for Updates' to connect to GitHub repository...")

    def _check_update(self):
        self.btn_upd.configure(state="disabled", text="Checking GitHub...")
        self.upd_log.delete("0.0", "end")
        self.upd_log.insert("end", "Connecting to repository...\n")
        threading.Thread(target=self._thread_upd, daemon=True).start()

    def _thread_upd(self):
        res = self.upd_mgr.check_update()
        if self.winfo_exists():
            self.after(0, lambda: self._res_upd(res))

    def _res_upd(self, res):
        is_new, tag, assets, body = res
        self.btn_upd.configure(state="normal", text="Check Again")
        self.upd_log.delete("0.0", "end")
        if is_new is None:
            self.upd_log.insert("end", f"Connection Error: {tag}\n")
        elif is_new:
            self.upd_log.insert(
                "end", f"NEW UPDATE AVAILABLE: v{tag}\n\n--- CHANGELOG ---\n\n{body}\n")
            if messagebox.askyesno("Update Found", f"Version v{tag} available. Download?"):
                target = next(
                    (a for a in assets if a['name'].endswith('.exe')), None)
                if target:
                    self._start_download(
                        target['browser_download_url'], target['name'])
                else:
                    webbrowser.open(
                        "https://github.com/Ivan-Ayub97/Warlock-Studio/releases")
        else:
            self.upd_log.insert(
                "end", f"You are using the latest version (v{tag}).")

    def _start_download(self, url, name):
        dw = DownloadWindow(self, name)

        def t():
            try:
                tmp = os.path.join(os.getenv('TEMP'), name)
                self.upd_mgr.download(url, tmp, dw.update_progress)
                dw.destroy()
                if messagebox.askyesno("Install", "Download complete. Install now?"):
                    os.startfile(tmp)
                    sys.exit(0)
            except Exception as e:
                dw.destroy()
                messagebox.showerror("Download Error", str(e))
        threading.Thread(target=t, daemon=True).start()

    def _build_logs_menu(self):
        menu = CollapsibleMenu(self.scroll_frame, "Log Management")
        frame = menu.content_frame
        ctrl = ctk.CTkFrame(frame, fg_color="transparent")
        ctrl.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(ctrl, text="Refresh Logs", width=120,
                      fg_color=THEME["widget_bg"], command=self._load_logs).pack(side="left", padx=5)
        ctk.CTkButton(ctrl, text="Export All Logs (.zip)", width=160,
                      fg_color=THEME["info_bg"], command=self._export_logs).pack(side="right", padx=5)
        self.log_text = ctk.CTkTextbox(frame, font=("Consolas", 11), fg_color="#111111", text_color="#DDDDDD",
                                       wrap="none", border_width=1, border_color=THEME["widget_bg"], height=200)
        self.log_text.pack(fill="x", expand=True, padx=5, pady=5)
        self._load_logs()

    def _load_logs(self):
        self.log_text.delete("0.0", "end")
        log_dir = os.path.join(os.path.expanduser('~'), 'Documents')
        patterns = [os.path.join(log_dir, "Warlock*Logs", "*.log"),
                    os.path.join(log_dir, "Warlock*Logs", "*.txt")]
        found_files = []
        for p in patterns:
            found_files.extend(glob.glob(p))
        found_files.sort(key=os.path.getmtime, reverse=True)
        if not found_files:
            self.log_text.insert("end", "No log files found.")
            return
        latest = found_files[0]
        self.log_text.insert(
            "end", f"--- LOADING LOG: {os.path.basename(latest)} ---\n\n")
        try:
            with open(latest, "r", encoding="utf-8", errors='replace') as f:
                self.log_text.insert("end", f.read())
        except Exception as e:
            self.log_text.insert("end", f"Error reading log: {e}")

    def _export_logs(self):
        try:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".zip", filetypes=[("ZIP", "*.zip")])
            if not save_path:
                return
            log_dir = os.path.join(os.path.expanduser('~'), 'Documents')
            folders = glob.glob(os.path.join(log_dir, "Warlock*Logs"))
            if not folders:
                return
            with zipfile.ZipFile(save_path, 'w') as zf:
                for folder in folders:
                    for root, _, files in os.walk(folder):
                        for file in files:
                            zf.write(os.path.join(root, file), arcname=file)
            messagebox.showinfo("Export", "Logs exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _build_about_menu(self):
        menu = CollapsibleMenu(self.scroll_frame, "About Warlock-Studio")
        frame = menu.content_frame
        ctk.CTkLabel(frame, text="WARLOCK-STUDIO", font=("Impact",
                     40), text_color=THEME["title"]).pack(pady=(20, 5))
        ctk.CTkLabel(frame, text=f"Version {self.curr_ver}", font=get_font(
            12), text_color=THEME["accent"]).pack(pady=(0, 20))
        SettingCard(frame, "DEVELOPER & PROJECT")
        ctk.CTkLabel(frame, text="Developed by Ivan-Ayub97",
                     font=get_font(12, "bold")).pack(pady=5)
        link_frame = ctk.CTkFrame(frame, fg_color="transparent")
        link_frame.pack(pady=10)
        ctk.CTkButton(link_frame, text="GitHub Repository", fg_color="#24292e", width=140, command=lambda: webbrowser.open(
            "https://github.com/Ivan-Ayub97/Warlock-Studio")).pack(side="left", padx=10)
        ctk.CTkButton(link_frame, text="SourceForge", fg_color="#EE7600", width=140, command=lambda: webbrowser.open(
            "https://sourceforge.net/projects/warlock-studio/")).pack(side="left", padx=10)

        SettingCard(frame, "LICENSES")
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(
            os.path.abspath(__file__)))
        license_path = os.path.join(base_dir, "Assets", "license.txt")
        try:
            with open(license_path, "r", encoding="utf-8") as f:
                l_txt = f.read()
        except:
            l_txt = "License file not found."
        ld = ctk.CTkTextbox(frame, height=200, fg_color="#111111",
                            text_color=THEME["text"], font=("Consolas", 10))
        ld.pack(fill="x", expand=True, pady=10, padx=20)
        ld.insert("0.0", l_txt)
        ld.configure(state="disabled")

    # --- HELPERS ---

    def _update(self, key, val, callback=None):
        if isinstance(val, float):
            val = round(val, 2)
        self.config[key] = val
        ConfigManager.save_config(self.config)
        if callback:
            callback()
        self.status_lbl.configure(text="Saved ✓", text_color=THEME["success"])
        self.after(2000, lambda: self.status_lbl.configure(text=""))

    def _add_slider(self, parent, text, key, min_v, max_v, steps, cmd):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f, text=text, width=140, anchor="w",
                     font=get_font(11)).pack(side="left")
        val_lbl = ctk.CTkLabel(
            f, text=f"{self.config.get(key, min_v):.2f}", width=40, text_color=THEME["accent"])
        val_lbl.pack(side="right")

        def internal_cmd(v):
            val_lbl.configure(text=f"{v:.2f}")
            cmd(v)
        s = ctk.CTkSlider(f, from_=min_v, to=max_v, number_of_steps=steps, command=internal_cmd,
                          progress_color=THEME["accent"], button_color=THEME["accent"], button_hover_color=THEME["accent_hover"])
        s.set(self.config.get(key, min_v))
        s.pack(side="right", fill="x", expand=True, padx=10)

    def _add_switch(self, parent, text, key, cmd):
        def internal_cmd():
            self.config[key] = bool(sw.get())
            ConfigManager.save_config(self.config)
            if cmd:
                cmd()
            self.status_lbl.configure(
                text="Saved ✓", text_color=THEME["success"])
            self.after(1500, lambda: self.status_lbl.configure(text=""))
        sw = ctk.CTkSwitch(parent, text=text, command=internal_cmd, font=get_font(11),
                           progress_color=THEME["accent"], button_color="#FFFFFF", button_hover_color="#EEEEEE")
        if self.config.get(key, False):
            sw.select()
        sw.pack(anchor="w", padx=10, pady=8)

    def _add_dropdown(self, parent, text, key, values, cmd):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(f, text=text, width=140, anchor="w",
                     font=get_font(11)).pack(side="left")
        om = ctk.CTkOptionMenu(f, values=values, command=cmd, font=get_font(11),
                               fg_color=THEME["widget_bg"], button_color=THEME["accent"],
                               button_hover_color=THEME["accent_hover"], text_color=THEME["bg"])
        om.set(self.config.get(key, values[0]))
        om.pack(side="right", fill="x", expand=True, padx=10)

    def _clean_temp(self):
        n = 0
        cwd = os.getcwd()
        patterns = ["*.tmp", "*.checkpoint",
                    "*.part", "temp_*", "*_frames.txt"]
        for p in patterns:
            for f in glob.glob(os.path.join(cwd, p)):
                try:
                    os.remove(f)
                    n += 1
                except:
                    pass
        messagebox.showinfo(
            "Clean", f"Cleanup complete.\nRemoved {n} temporary files.")

    def _reset_settings(self):
        if messagebox.askyesno("Reset", "Restore default settings?"):
            ConfigManager.reset_config()
            self.destroy()

# -----------------------------------------------------------------------------
# BOTÓN PRINCIPAL (INTEGRACIÓN)
# -----------------------------------------------------------------------------


class PreferencesButton(ctk.CTkButton):
    def __init__(self, master, current_version, repo_owner="Ivan-Ayub97", repo_name="Warlock-Studio", **kwargs):
        super().__init__(master, text="⚙", width=100, height=28,
                         fg_color=THEME["widget_bg"], border_color=THEME["border"], border_width=1,
                         hover_color=THEME["hover"], text_color=THEME["text"],
                         command=self.open_window, **kwargs)
        self.ver = current_version
        self.repo = (repo_owner, repo_name)
        self._apply_startup_config()

    def open_window(self):
        for w in self.master.winfo_children():
            if isinstance(w, PreferencesWindow):
                w.lift()
                w.focus_force()
                return
        PreferencesWindow(self.master, self.ver, *self.repo)

    def _apply_startup_config(self):
        c = ConfigManager.load_config()
        try:
            ctk.set_appearance_mode(c.get("app_theme", "Dark"))
            ctk.set_widget_scaling(c.get("ui_scaling", 1.0))
        except:
            pass
        if c.get("window_opacity", 1.0) < 1.0:
            self.master.attributes("-alpha", c["window_opacity"])
        if c.get("keep_window_on_top", False):
            self.master.attributes("-topmost", True)

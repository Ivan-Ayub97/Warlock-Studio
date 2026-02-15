import math
import os
import random
from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image


class SplashScreen(ctk.CTkToplevel):
    def __init__(
        self,
        root_window,
        app_title: str,
        version: str,
        asset_loader: Callable[[str], str],
        theme_colors: dict,
        # CAMBIO 1: Duración ultrarrápida (1.2 segundos)
        duration_ms: int = 1200,
        # CAMBIO 2: Callback para ejecutar al finalizar
        on_complete: Optional[Callable] = None
    ):
        super().__init__(root_window)

        # --- CONFIGURACIÓN DE DATOS ---
        self.asset_loader = asset_loader
        self.theme = theme_colors
        self.duration_ms = duration_ms
        self.on_complete = on_complete  # Guardamos la función a ejecutar al final

        # Reducimos los pasos para que coincidan con la duración corta
        self.steps_total = 60
        self.current_step = 0
        self.animation_running = True

        # --- CONFIGURACIÓN DE PARTÍCULAS ---
        self.particles = []
        self.num_particles = 35  # Ligero y rápido

        # --- CONFIGURACIÓN DE VENTANA (AL FRENTE) ---
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        self.lift()
        self.focus_force()

        # --- GEOMETRÍA Y CENTRADO ---
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        width = 500
        height = 320

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

        # --- CONSTRUCCIÓN DE UI ---
        self._setup_ui(app_title, version, width, height)

        # --- INICIAR ANIMACIONES ---
        self._init_particles(width, height)
        self._animate_particles()

        # Iniciar carga casi inmediatamente
        self.after(20, self._animate_loading)

    def _setup_ui(self, app_title, version, width, height):
        """Construye las capas de la interfaz."""

        # --- 1. CAPA DE FONDO (Canvas) ---
        self.canvas = ctk.CTkCanvas(
            self,
            width=width,
            height=height,
            bg=self.theme['bg'],
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Borde decorativo
        self.canvas.create_rectangle(
            2, 2, width-2, height-2,
            outline=self.theme['accent'],
            width=2
        )

        # --- 2. CAPA DE CONTENIDO ---

        # A. Carga del Banner
        banner_path = self.asset_loader(f"Assets{os.sep}banner.png")
        has_banner = False

        if os.path.exists(banner_path):
            try:
                pil_img = Image.open(banner_path)
                banner_ctk = ctk.CTkImage(
                    light_image=pil_img, dark_image=pil_img, size=(460, 180))

                self.banner_label = ctk.CTkLabel(
                    self,
                    image=banner_ctk,
                    text=""
                )
                self.banner_label.place(relx=0.5, rely=0.1, anchor="n")
                has_banner = True
            except Exception as e:
                print(f"[SPLASH] Error cargando banner: {e}")

        # B. Título de respaldo
        if not has_banner:
            self.title_label = ctk.CTkLabel(
                self,
                text=app_title.upper(),
                font=ctk.CTkFont(family="Segoe UI", size=32, weight="bold"),
                text_color=self.theme['app_name']
            )
            self.title_label.place(relx=0.5, rely=0.15, anchor="n")

        # C. Barra de Progreso
        self.progress_bar = ctk.CTkProgressBar(
            self,
            width=300,
            height=8,
            corner_radius=8,
            progress_color=self.theme['accent'],
            fg_color=self.theme['widget_bg'],
            border_width=0
        )
        self.progress_bar.set(0)
        self.progress_bar.place(relx=0.5, rely=0.7, anchor="center")

        # D. Texto de estado
        self.status_label = ctk.CTkLabel(
            self,
            text="Starting...",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=self.theme['text_sec']
        )
        self.status_label.place(relx=0.5, rely=0.8, anchor="center")

        # E. Etiqueta de Versión
        self.version_label = ctk.CTkLabel(
            self,
            text=f"v{version} | Developed by Ivan-Ayub97",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=self.theme['accent']
        )
        self.version_label.place(relx=0.5, rely=0.93, anchor="center")

    def _init_particles(self, width, height):
        particle_color = self.theme.get('text_sec', '#888888')
        for _ in range(self.num_particles):
            size = random.randint(2, 4)
            x = random.randint(20, width - 20)
            y = random.randint(20, height - 20)
            # Un poco más rápido para dar sensación de fluidez
            dx = random.uniform(-1.0, 1.0)
            dy = random.uniform(-1.0, 1.0)

            particle = {
                'id': self.canvas.create_oval(
                    x, y, x+size, y+size,
                    fill=particle_color,
                    outline=""
                ),
                'x': x, 'y': y, 'dx': dx, 'dy': dy,
                'width': width, 'height': height
            }
            self.particles.append(particle)

    def _animate_particles(self):
        if not self.winfo_exists() or not self.animation_running:
            return

        for p in self.particles:
            p['x'] += p['dx']
            p['y'] += p['dy']

            if p['x'] <= 5 or p['x'] >= p['width'] - 5:
                p['dx'] *= -1
            if p['y'] <= 5 or p['y'] >= p['height'] - 5:
                p['dy'] *= -1

            self.canvas.coords(p['id'], p['x'], p['y'], p['x']+3, p['y']+3)

        self.after(33, self._animate_particles)

    def _animate_loading(self):
        messages = ["Loading...", "Modules...", "UI...", "Starting..."]

        if not self.winfo_exists():
            return

        self.lift()
        self.attributes('-topmost', True)

        # Cálculo seguro de tiempo
        step_interval = max(10, self.duration_ms // self.steps_total)

        if self.current_step <= self.steps_total:
            progress = self.current_step / self.steps_total
            self.progress_bar.set(progress)

            if self.current_step % 10 == 0:
                msg_index = int((progress * 100) / (100 / len(messages)))
                if msg_index < len(messages):
                    self.status_label.configure(text=messages[msg_index])

            self.current_step += 1
            self.after(step_interval, self._animate_loading)
        else:
            self.start_fade_out()

    def start_fade_out(self):
        self.animation_running = False
        self._fade_step = 1.0
        self._fade_out()

    def _fade_out(self):
        if not self.winfo_exists():
            return

        if self._fade_step > 0:
            # Desvanecimiento muy rápido
            self.attributes('-alpha', self._fade_step)
            self._fade_step -= 0.1  # Salta de 10% en 10% para velocidad
            self.after(20, self._fade_out)
        else:
            # 1. Destruimos la splash
            self.destroy()
            # 2. CAMBIO CRÍTICO: Ejecutamos el callback para abrir el programa principal
            if self.on_complete:
                self.on_complete()

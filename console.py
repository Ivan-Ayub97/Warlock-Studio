import json
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk


class ConsoleColors:
    """Definición de paleta de colores para los logs."""
    TEXT_DEFAULT = "#E0E0E0"
    TIMESTAMP = "#808080"
    DEBUG = "#A0A0A0"
    INFO = "#61AFEF"  # Azul claro
    SUCCESS = "#98C379"  # Verde pastel
    WARNING = "#E5C07B"  # Amarillo/Naranja
    ERROR = "#E06C75"  # Rojo suave
    CRITICAL = "#FF0000"  # Rojo intenso
    HIGHLIGHT = "#D19A66"  # Color para resaltar búsquedas


class IntegratedConsole(ctk.CTkFrame):
    """
    Un widget de consola completo con barra de herramientas,
    búsqueda y visualización de logs coloreados.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Configuración del Grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # El textbox se expande

        # --- 1. BARRA DE HERRAMIENTAS (Toolbar) ---
        self.toolbar = ctk.CTkFrame(self, height=30, fg_color="transparent")
        self.toolbar.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 5))

        # Botón Limpiar
        self.btn_clear = ctk.CTkButton(
            self.toolbar, text="🗑 Clear", width=60, height=24,
            font=("Segoe UI", 11), fg_color="#3E3E3E", hover_color="#505050",
            command=self.clear_console
        )
        self.btn_clear.pack(side="left", padx=2)

        # Botón Copiar
        self.btn_copy = ctk.CTkButton(
            self.toolbar, text="📋 Copy", width=60, height=24,
            font=("Segoe UI", 11), fg_color="#3E3E3E", hover_color="#505050",
            command=self.copy_all
        )
        self.btn_copy.pack(side="left", padx=2)

        # Botón Guardar
        self.btn_save = ctk.CTkButton(
            self.toolbar, text="💾 Save", width=60, height=24,
            font=("Segoe UI", 11), fg_color="#3E3E3E", hover_color="#505050",
            command=self.save_to_file
        )
        self.btn_save.pack(side="left", padx=2)

        # Separador visual (Spacer)
        ctk.CTkLabel(self.toolbar, text="|", text_color="gray").pack(
            side="left", padx=5)

        # Checkbox Auto-Scroll
        self.auto_scroll_var = ctk.BooleanVar(value=True)
        self.chk_autoscroll = ctk.CTkCheckBox(
            self.toolbar, text="Auto-scroll", variable=self.auto_scroll_var,
            font=("Segoe UI", 11), width=80, height=20, checkbox_width=18, checkbox_height=18
        )
        self.chk_autoscroll.pack(side="left", padx=5)

        # Barra de Búsqueda
        self.entry_search = ctk.CTkEntry(
            self.toolbar, placeholder_text="🔍 Search...", width=120, height=24,
            font=("Segoe UI", 11)
        )
        self.entry_search.pack(side="right", padx=2)
        self.entry_search.bind("<KeyRelease>", self.on_search)

        # --- 2. ÁREA DE TEXTO (Logs) ---
        self.textbox = ctk.CTkTextbox(
            self,
            font=("Consolas", 13),  # Fuente monoespaciada para mejor lectura
            fg_color="#1E1E1E",     # Fondo oscuro tipo editor
            text_color=ConsoleColors.TEXT_DEFAULT,
            wrap="word",
            state="disabled"
        )
        self.textbox.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # Configuración de Tags (Colores)
        self.textbox.tag_config(
            "TIMESTAMP", foreground=ConsoleColors.TIMESTAMP)
        self.textbox.tag_config("DEBUG", foreground=ConsoleColors.DEBUG)
        self.textbox.tag_config("INFO", foreground=ConsoleColors.INFO)
        self.textbox.tag_config("SUCCESS", foreground=ConsoleColors.SUCCESS)
        self.textbox.tag_config("WARNING", foreground=ConsoleColors.WARNING)
        self.textbox.tag_config("ERROR", foreground=ConsoleColors.ERROR)
        self.textbox.tag_config("CRITICAL", foreground=ConsoleColors.CRITICAL)
        self.textbox.tag_config(
            "SEARCH_HIGHLIGHT", background="#444400", foreground="#FFFFFF")

        # Menú contextual (Click derecho)
        self.context_menu = tk.Menu(self, tearoff=0, bg="#2b2b2b", fg="white")
        self.context_menu.add_command(label="Copy All", command=self.copy_all)
        self.context_menu.add_command(
            label="Clear Console", command=self.clear_console)
        self.textbox.bind("<Button-3>", self.show_context_menu)

    def write_log(self, text, level="INFO"):
        """Método principal para escribir en la consola."""
        try:
            self.textbox.configure(state="normal")

            # Timestamp
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            self.textbox.insert("end", f"{timestamp} ", "TIMESTAMP")

            # Etiqueta de Nivel (opcional, visualmente ayuda)
            self.textbox.insert("end", f"[{level.upper()}] ", level.upper())

            # Mensaje
            # Aseguramos que termine en nueva línea
            if not text.endswith("\n"):
                text += "\n"

            self.textbox.insert("end", text, level.upper())

            # Auto-scroll
            if self.auto_scroll_var.get():
                self.textbox.see("end")

            self.textbox.configure(state="disabled")
        except Exception as e:
            print(f"Error writing to console widget: {e}")

    # --- Funcionalidades de la Barra de Herramientas ---

    def clear_console(self):
        self.textbox.configure(state="normal")
        self.textbox.delete("0.0", "end")
        self.textbox.configure(state="disabled")

    def copy_all(self):
        try:
            all_text = self.textbox.get("0.0", "end")
            self.clipboard_clear()
            self.clipboard_append(all_text)
            self.update()  # Necesario para finalizar la operación de portapapeles
        except Exception:
            pass

    def save_to_file(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"),
                           ("Log Files", "*.log"), ("All Files", "*.*")],
                title="Save Log File"
            )
            if filename:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self.textbox.get("0.0", "end"))
        except Exception as e:
            self.write_log(f"Error saving file: {e}", "ERROR")

    def on_search(self, event=None):
        """Resalta texto en tiempo real."""
        search_str = self.entry_search.get()

        # Limpiar tags anteriores de búsqueda
        self.textbox.tag_remove("SEARCH_HIGHLIGHT", "1.0", "end")

        if not search_str:
            return

        # Buscar y resaltar
        start_pos = "1.0"
        while True:
            start_pos = self.textbox.search(
                search_str, start_pos, stopindex="end", nocase=True)
            if not start_pos:
                break
            end_pos = f"{start_pos}+{len(search_str)}c"
            self.textbox.tag_add("SEARCH_HIGHLIGHT", start_pos, end_pos)
            start_pos = end_pos

    def show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()


class StreamRedirector:
    """Redirige stdout/stderr hacia el ConsoleManager."""

    def __init__(self, console_manager, stream_type):
        self.console_manager = console_manager
        self.stream_type = stream_type
        self.original_stream = getattr(sys, stream_type)

    def write(self, message):
        if message.strip():  # Ignorar líneas vacías
            self.console_manager.enqueue_message(message, self.stream_type)

        # Mantener salida en terminal real por seguridad
        try:
            self.original_stream.write(message)
            self.original_stream.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass


class ConsoleManager:
    """
    Singleton que gestiona la lógica de logs y la conexión con la GUI.
    Thread-safe.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConsoleManager, cls).__new__(cls)
            cls._instance.buffer = []
            cls._instance.widget_frame = None  # Referencia al Frame contenedor
            cls._instance.lock = threading.Lock()
            cls._instance.is_setup = False
        return cls._instance

    def setup_redirection(self):
        """Configura la redirección de sys.stdout y sys.stderr."""
        if not self.is_setup:
            sys.stdout = StreamRedirector(self, "stdout")
            sys.stderr = StreamRedirector(self, "stderr")
            self.is_setup = True
            print("[SYSTEM] Output redirected to Warlock Integrated Console.")

    def set_widget(self, widget_frame: IntegratedConsole):
        """Vincula el widget gráfico (IntegratedConsole) y vuelca el buffer."""
        with self.lock:
            self.widget_frame = widget_frame

            # Volcar buffer acumulado
            for msg, stream_type in self.buffer:
                self._process_and_dispatch(msg, stream_type)
            self.buffer = []

    def enqueue_message(self, message, stream_type):
        """Punto de entrada desde stdout/stderr."""
        with self.lock:
            if self.widget_frame:
                self._process_and_dispatch(message, stream_type)
            else:
                self.buffer.append((message, stream_type))

    def _process_and_dispatch(self, message, stream_type="info"):
        """Determina el nivel del log y lo envía a la GUI."""
        try:
            tag = "INFO"
            msg_lower = message.lower()

            # Lógica de detección de nivel automática
            if stream_type == "stderr":
                tag = "ERROR"
            elif "error" in msg_lower or "exception" in msg_lower or "failed" in msg_lower:
                tag = "ERROR"
            elif "warning" in msg_lower:
                tag = "WARNING"
            elif "success" in msg_lower or "completed" in msg_lower:
                tag = "SUCCESS"
            elif "debug" in msg_lower:
                tag = "DEBUG"
            elif "critical" in msg_lower:
                tag = "CRITICAL"

            # Enviar al hilo principal
            self.widget_frame.after(
                0, lambda m=message, t=tag: self.widget_frame.write_log(m, t))
        except Exception:
            pass

    def write_log(self, message, tag="INFO"):
        """
        Método público para logs manuales.
        Uso: console.write_log("Proceso terminado", "SUCCESS")
        """
        with self.lock:
            if self.widget_frame:
                try:
                    self.widget_frame.after(
                        0, lambda m=message, t=tag: self.widget_frame.write_log(m, t))
                except Exception:
                    print(f"[{tag}] {message}")
            else:
                print(f"[{tag}] {message}")


# Instancia global
console = ConsoleManager()

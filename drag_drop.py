# drag_drop.py
from customtkinter import CTk
from tkinterdnd2 import DND_ALL, TkinterDnD

# 1. Clase envoltorio que combina CustomTkinter con TkinterDnD


class DnDCTk(CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

# 2. Función para registrar widgets y conectar tu lógica


def enable_drag_and_drop(window, target_widgets, callback_function):
    """
    Activa Drag & Drop en los widgets especificados.

    :param window: La ventana principal (debe ser instancia de DnDCTk)
    :param target_widgets: Lista de widgets (botones, labels) donde se pueden soltar archivos.
    :param callback_function: La función de tu app principal que recibe la lista de archivos.
    """

    def _internal_drop_event(event):
        # TkinterDnD a veces devuelve las rutas con llaves {} si tienen espacios
        # window.tk.splitlist se encarga de limpiarlas correctamente
        if event.data:
            files = window.tk.splitlist(event.data)
            # Llamamos a tu función principal pasando la lista limpia
            callback_function(files)

    for widget in target_widgets:
        # Registramos el widget para aceptar cualquier cosa (archivos)
        widget.drop_target_register(DND_ALL)
        # Conectamos el evento 'Drop' con nuestra función interna
        widget.dnd_bind('<<Drop>>', _internal_drop_event)

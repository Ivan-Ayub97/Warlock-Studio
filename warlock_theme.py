import math

# -----------------------------------------------------------------------------
# WARLOCK STUDIO: THEME & COLORS (PROFESSIONAL EDITION)
# -----------------------------------------------------------------------------
# Este módulo define la paleta de colores centralizada para la aplicación.
# Diseño: Dark / Gold / Professional
# -----------------------------------------------------------------------------


class WarlockColors:
    """
    Definición estática de la paleta de colores.
    Centraliza los códigos HEX para facilitar cambios globales.
    """

    # --- BASE & FONDO ---
    # Negro profundo mate. Evitamos el negro absoluto (#000) en grandes áreas,
    # pero lo usamos aquí como base para superponer capas.
    BACKGROUND_MAIN = "#000000"

    # Gris carbón muy oscuro. Fondo para paneles laterales y tarjetas.
    BACKGROUND_WIDGET = "#1A1A1A"

    # Un tono ligeramente más claro para campos de entrada (Entries) y Dropdowns.
    BACKGROUND_ENTRY = "#232323"

    # --- IDENTIDAD & ACENTOS ---
    # "Oro Metálico". Elegante, legible y define la marca Warlock.
    APP_TITLE = "#FFD700"

    # "Ámbar Intenso". Color principal de interacción (Checkboxes, Switches, Sliders).
    ACCENT = "#FFC107"

    # Versión más oscura del acento para estados "pressed" o bordes activos.
    ACCENT_DARK = "#FFB300"

    # "Crema Pálido". Para resaltar texto seleccionado o búsquedas.
    HIGHLIGHT = "#FFF59D"

    # --- TEXTO ---
    # Blanco humo ("White Smoke"). Evita la fatiga visual del blanco puro (#FFFFFF).
    TEXT_PRIMARY = "#F5F5F5"

    # Gris Pizarra. Para etiquetas, subtítulos y descripciones.
    TEXT_SECONDARY = "#9E9E9E"

    # Gris oscuro. Para texto deshabilitado o placeholders.
    TEXT_DISABLED = "#616161"

    # --- INTERACCIÓN (BOTONES) ---
    # "Rojo Sangre/Rubí". Para Hover en botones principales o acciones destructivas.
    BUTTON_HOVER = "#D32F2F"

    # "Vino Tinto". Para botones secundarios, info o estados inactivos pero visibles.
    BUTTON_SECONDARY = "#7F1500"

    # Color cuando se presiona click.
    BUTTON_PRESSED = "#8E0000"

    # --- ESTADOS DEL SISTEMA ---
    # "Naranja Quemado". Advertencias legibles sobre fondo oscuro.
    STATUS_WARNING = "#FF6F00"

    # Verde Esmeralda. Éxito, confirmaciones, logs positivos.
    STATUS_SUCCESS = "#00C853"

    # "Rojo Ladrillo". Errores críticos.
    STATUS_ERROR = "#B71C1C"

    # Azul informativo (para logs neutrales o información).
    STATUS_INFO = "#0277BD"

    # --- BORDES & SEPARADORES ---
    # Gris cálido muy oscuro. Sutil para dividir secciones.
    BORDER = "#2D2D2D"

    # Borde activo (cuando un input tiene el foco).
    BORDER_FOCUS = "#424242"

    # --- SCROLLBARS ---
    # Rojo Caoba translúcido/oscuro. Discreto.
    SCROLLBAR_FG = "#3E2723"
    SCROLLBAR_BG = "#0D0D0D"  # Casi negro para el riel del scroll


class ColorUtils:
    """
    Utilidades para manipular colores hexadecimales en tiempo de ejecución.
    Útil para generar variaciones de colores para animaciones o estados.
    """

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convierte '#RRGGBB' a una tupla (r, g, b)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb: tuple) -> str:
        """Convierte una tupla (r, g, b) a '#RRGGBB'."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    @staticmethod
    def adjust_brightness(hex_color: str, factor: float) -> str:
        """
        Ajusta el brillo de un color.
        :param factor: > 1 para aclarar, < 1 para oscurecer.
        Ej: 1.2 es 20% más claro. 0.8 es 20% más oscuro.
        """
        r, g, b = ColorUtils.hex_to_rgb(hex_color)

        # Ajustar y asegurar que esté entre 0-255
        r = min(255, max(0, int(r * factor)))
        g = min(255, max(0, int(g * factor)))
        b = min(255, max(0, int(b * factor)))

        return ColorUtils.rgb_to_hex((r, g, b))

# -----------------------------------------------------------------------------
# DICCIONARIO DE TEMA (EXPORTACIÓN PRINCIPAL)
# -----------------------------------------------------------------------------
# Mapea las constantes a claves semánticas que la UI puede consumir directamente.
# Se incluyen claves compatibles con CustomTkinter (fg_color, hover_color, etc).
# -----------------------------------------------------------------------------


THEME = {
    # --- General ---
    "bg": WarlockColors.BACKGROUND_MAIN,
    "app_name": WarlockColors.APP_TITLE,
    "accent": WarlockColors.ACCENT,

    # --- Widgets (Frames, Cards) ---
    "widget_bg": WarlockColors.BACKGROUND_WIDGET,
    "widget_border": WarlockColors.BORDER,

    # --- Texto ---
    "text_main": WarlockColors.TEXT_PRIMARY,
    "text_sec": WarlockColors.TEXT_SECONDARY,
    "text_disabled": WarlockColors.TEXT_DISABLED,

    # --- Botones (Estados) ---
    "btn_fg": WarlockColors.BACKGROUND_WIDGET,    # Color base del botón
    "btn_hover": WarlockColors.BUTTON_HOVER,      # Color al pasar mouse
    "btn_secondary": WarlockColors.BUTTON_SECONDARY,
    "btn_pressed": WarlockColors.BUTTON_PRESSED,

    # --- Entradas de Texto (Inputs) ---
    "entry_bg": WarlockColors.BACKGROUND_ENTRY,
    "entry_border": WarlockColors.BORDER,
    "entry_focus": WarlockColors.BORDER_FOCUS,

    # --- Feedback / Estados ---
    "success": WarlockColors.STATUS_SUCCESS,
    "warning": WarlockColors.STATUS_WARNING,
    "error": WarlockColors.STATUS_ERROR,
    "info": WarlockColors.STATUS_INFO,

    # --- Varios ---
    "highlight": WarlockColors.HIGHLIGHT,
    "scrollbar": WarlockColors.SCROLLBAR_FG,

    # --- Compatibilidad Legacy (Mantiene compatibilidad con tu código anterior) ---
    "background_color": WarlockColors.BACKGROUND_MAIN,
    "app_name_color": WarlockColors.APP_TITLE,
    "widget_background_color": WarlockColors.BACKGROUND_WIDGET,
    "text_color": WarlockColors.TEXT_PRIMARY,
    "secondary_text_color": WarlockColors.TEXT_SECONDARY,
    "accent_color": WarlockColors.ACCENT,
    "button_hover_color": WarlockColors.BUTTON_HOVER,
    "border_color": WarlockColors.BORDER,
    "info_button_color": WarlockColors.BUTTON_SECONDARY,
    "warning_color": WarlockColors.STATUS_WARNING,
    "success_color": WarlockColors.STATUS_SUCCESS,
    "error_color": WarlockColors.STATUS_ERROR,
    "highlight_color": WarlockColors.HIGHLIGHT,
    "scrollbar_color": WarlockColors.SCROLLBAR_FG
}

# -----------------------------------------------------------------------------
# VARIABLES GLOBALES DIRECTAS (OPCIONALES PARA ACCESO RÁPIDO)
# -----------------------------------------------------------------------------
BACKGROUND_COLOR = WarlockColors.BACKGROUND_MAIN
APP_NAME_COLOR = WarlockColors.APP_TITLE
WIDGET_BACKGROUND_COLOR = WarlockColors.BACKGROUND_WIDGET
TEXT_COLOR = WarlockColors.TEXT_PRIMARY
SECONDARY_TEXT_COLOR = WarlockColors.TEXT_SECONDARY
ACCENT_COLOR = WarlockColors.ACCENT
BUTTON_HOVER_COLOR = WarlockColors.BUTTON_HOVER
BORDER_COLOR = WarlockColors.BORDER
INFO_BUTTON_COLOR = WarlockColors.BUTTON_SECONDARY
WARNING_COLOR = WarlockColors.STATUS_WARNING
SUCCESS_COLOR = WarlockColors.STATUS_SUCCESS
ERROR_COLOR = WarlockColors.STATUS_ERROR
HIGHLIGHT_COLOR = WarlockColors.HIGHLIGHT
SCROLLBAR_COLOR = WarlockColors.SCROLLBAR_FG

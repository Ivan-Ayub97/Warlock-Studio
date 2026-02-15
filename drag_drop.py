from customtkinter import CTk
from tkinterdnd2 import DND_FILES, TkinterDnD


class DnDCTk(CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        TkinterDnD.DnDWrapper.__init__(self)
        self.TkdndVersion = TkinterDnD._require(self)


def enable_drag_and_drop(window, target_widgets, callback_function):
    def _internal_drop_event(event):
        if event.data:
            files = window.tk.splitlist(event.data)
            callback_function(files)

    for widget in target_widgets:
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind('<<Drop>>', _internal_drop_event)

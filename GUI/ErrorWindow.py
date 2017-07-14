try:  # Will work with python 3
    from tkinter.messagebox import showerror
except ImportError:  # Will work with python 2
    from tkMessageBox import showerror


class ErrorWindow:
    def __init__(self, master, message):
        showerror(master=master, title="Error!", message=message)
        return

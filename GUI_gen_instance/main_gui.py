import customtkinter as ctk
from support_gui import SupportFrame
from front_gui import MainFrame

class GUI(ctk.CTk):
    def __init__(self, logic_instance):
        ctk.CTk.__init__(self)
        self.title("Instances Generator")
        self.logic_instance = logic_instance
        self.geometry("800x400")
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        self.minsize(350, 395)
        self.createWidgets()

    def createWidgets(self):
        self.supportFrame = SupportFrame(self)
        self.supportFrame.pack(side="top", fill="x", expand=False)
        self.mainFrame = MainFrame(self)
        self.mainFrame.pack(side="top", fill="both", expand=True)

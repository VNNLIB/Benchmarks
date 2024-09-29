from main_gui import GUI
from logic import logic

# Thanks to Akascape: https://github.com/Akascape/CTkXYFrame, https://github.com/Akascape/CTkScrollableDropdown, https://github.com/Akascape/CtkTable for the custom tkinter widgets

def main():
    logic_instance = logic()
    app = GUI(logic_instance)
    app.mainloop()
    
if __name__ == "__main__":
    main()
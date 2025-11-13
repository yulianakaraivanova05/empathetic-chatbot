from chat.response_generator import *
import tkinter as tk
from chat_gui import ChatbotGUI


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
    #conversation()
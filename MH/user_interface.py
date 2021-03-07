import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from screeninfo import get_monitors

for m in get_monitors():
    print(m)


class UserInterface:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.width, self.height = int(self.cap.get(3)*1.4), int(self.cap.get(4)*1.4)
        print(self.width, self.height)
        self.root = tk.Tk()
        # self.root.attributes("-fullscreen", True)     # Full Screen
        self.set_widow_size(self.width, self.height)
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.lmain = tk.Label(self.root)
        self.lmain.pack()
        self.start_time = time.time()

    def set_widow_size(self, w, h):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.root.title("Active Aging Advisory System")
        # self.root.geometry(str(w)+"x"+str(h)+"+0+0")
        self.root.resizable(False, False)

    def run(self):
        self.show_frame()
        self.info_view()
        self.show_top_button()
        # self.show_main_button()
        self.show_setting_button()
        self.root.mainloop()

    def show_top_button(self):
        btn_plus = tk.Button(self.root, width=5, height=3, text="+", command=self.helloCallBack,
                             activebackground="#123345",
                             borderwidth=0)
        btn_minus = tk.Button(self.root, width=5, height=3, text="-", command=self.helloCallBack,
                              activebackground="#123345",
                              borderwidth=0)
        btn_capture = tk.Button(self.root, width=5, height=3, text="C", command=self.helloCallBack,
                                activebackground="#123345",
                                borderwidth=0)
        btn_plus.place(rely=0.03, relx=0.03)
        btn_minus.place(rely=0.03, relx=0.10)
        btn_capture.place(rely=0.03, relx=0.17)

    def show_main_button(self):
        btn_anal = tk.Button(self.root, width=12, height=3, text="Analytics", command=self.helloCallBack, activebackground="#123345",
                      borderwidth=0)
        btn_recm = tk.Button(self.root, width=12, height=3, text="Recommend", command=self.helloCallBack, activebackground="#123345",
                       borderwidth=0)
        btn_evol = tk.Button(self.root, width=12, height=3, text="Evolution", command=self.helloCallBack, activebackground="#123345",
                       borderwidth=0)
        btn_pht = tk.Button(self.root, width=12, height=3, text="Photo", command=self.helloCallBack, activebackground="#123345",
                       borderwidth=0)
        btn_set = tk.Button(self.root, width=12, height=3, text="setting", command=self.helloCallBack, activebackground="#123345",
                       borderwidth=0)
        btn_anal.place(anchor="nw", relx=0.85, rely=0.15)
        btn_recm.place(anchor="nw", relx=0.85, rely=0.25)
        btn_evol.place(anchor="nw", relx=0.85, rely=0.35)
        btn_pht.place(anchor="nw", relx=0.85, rely=0.45)
        btn_set.place(anchor="nw", relx=0.85, rely=0.55)

    def show_setting_button(self):
        btn_theme = tk.Button(self.root, width=12, height=3, text="Theme", command=self.helloCallBack, activebackground="#123345",
                      borderwidth=0)
        btn_loc = tk.Button(self.root, width=12, height=3, text="Location", command=self.helloCallBack, activebackground="#123345",
                       borderwidth=0)
        btn_display = tk.Button(self.root, width=12, height=3, text="Display Option", command=self.helloCallBack, activebackground="#123345",
                       borderwidth=0)
        btn_register = tk.Button(self.root, width=12, height=3, text="Register", command=self.helloCallBack, activebackground="#123345",
                       borderwidth=0)
        btn_back = tk.Button(self.root, width=12, height=3, text="<-", command=self.helloCallBack, activebackground="#123345",
                       borderwidth=0)
        btn_back.place(anchor="nw", relx=0.85, rely=0.15)
        btn_theme.place(anchor="nw", relx=0.85, rely=0.25)
        btn_loc.place(anchor="nw", relx=0.85, rely=0.35)
        btn_display.place(anchor="nw", relx=0.85, rely=0.45)
        btn_register.place(anchor="nw", relx=0.85, rely=0.55)

    def info_view(self):
        frame1 = tk.Frame(self.root, bd=3, width=10, heigh=10)
        frame1.place(relx=0.03, rely=0.15)
        self.txt = tk.Text(frame1, width=30, heigh=20, wrap="word",   borderwidth=0, highlightthickness=0, relief="flat")
        self.txt.insert("current", "Greeting\n")
        self.txt.insert("current", "Welcome to Active Aging Advisory.")
        self.txt.pack()

    def helloCallBack(self):
        print("hihihih")

    def show_frame(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        if time.time() - self.start_time > 10:
            self.start_time = time.time()
            self.txt.delete(1.0, tk.END)
            self.txt.insert("current", "asdfasdf\n")
        self.lmain.after(10, self.show_frame)


if __name__ == '__main__':
    ui = UserInterface()
    ui.run()

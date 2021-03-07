"""
Date: 2019. 07. 15
Programmer: MH
Description: User interface code for AAA System server tier
"""
import tkinter as tk
import tkinter.ttk
from tkinter import Button, Entry, Frame, Label, Toplevel, StringVar


class ServerUserInterface:

    def __init__(self):
        """
        To init variables
        """
        self.page = ["Main", "DB", "Model", "Report","Backup"]
        self.curr_page = self.page[1]
        self.root = tk.Tk()
        self.root.title("Active Aging Advisory System")
        self.root.geometry("640x400+100+100")
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.is_session = True
        self.session_info = {"id":"1234", "password":"1234"}

    def set_header(self):
        """
        To set header section
        :return: none
        """
        self.header = Frame(self.root, height=70, relief="solid", bg="#FFFFFF")
        self.header.place(y=0, width=640, height=70)
        title = Label(self.header, text="Active Aging Advisory", bg="#FFFFFF")
        title.place(relx=0.4, rely=0.2)
        title.bind("<Button>", self.cbk_main)
        if self.is_session:
            print(self.session_info)
            user_info = Label(self.header, text="Hi! "+self.session_info["id"], bg="#FFFFFF")
            user_info.place(relx=0.75, rely=0.7)
            user_info = Label(self.header, text="Logout", bg="#FFFFFF")
            user_info.place(relx=0.9, rely=0.7)
            user_info.bind("<Button>", self.cbk_logout)

    def set_body(self):
        """
        To set body section
        :return:
        """
        self.frame_body = Frame(self.root, height=330)
        self.frame_body.place(y=70, width=640, height=330)
        if self.is_session:
            if self.curr_page == self.page[0]:
                db_data = {"Basic Diagnosis": {"UD": "123", "df": "123"},
                           "Basic Factors": {"UD": "123", "df": "123"},
                           "Recommendation": {"UD": "123", "df": "123"}}
                md_data = {"Basic Diagnosis": {"UD1": "123", "df1": "123", "UD2": "123", "df2": "123",
                                               "UD3": "123", "df3": "123"}}
                rp_data = {"Last Report": {"Date": "123"}}
                bu_data = {"Last Backup": {"Date": "123"}}
                self._set_main_body(db_data, md_data, rp_data, bu_data)
            elif self.curr_page == self.page[1]:
                self._set_db_body()
            elif self.curr_page == self.page[2]:
                self._set_model_body()
            elif self.curr_page == self.page[3]:
                self._set_report_body()
            elif self.curr_page == self.page[4]:
                self._set_backup_body()
        else:
            self.frame_body.config(bg="#eeeeee")

    def set_menu(self):
        """
        To set menu section
        :return:
        """
        if self.is_session:
            if self.curr_page != self.page[0]:
                self.frame_menu = Frame(self.frame_body, bg="#aaaaaa")
                self.frame_menu.place(x=0, width=160, height=330)
                if self.curr_page == self.page[1]:
                    title = Label(self.frame_menu, text="Data Base", width=10, height=1, bg="#aaaaaa")
                    title.place(x=0, y=3)
                    self.btn_table1 = Button(self.frame_menu, text="Table 1", command=lambda: self.cbk_db_table("Table1"),
                                           activebackground="#123345",
                                           borderwidth=0, bg="#CCCCCC", bd=1)
                    self.btn_table2 = Button(self.frame_menu, text="Table 2", command=lambda: self.cbk_db_table("Table2"),
                                           activebackground="#123345",
                                           borderwidth=0, bg="#CCCCCC", bd=1)
                    self.btn_table3 = Button(self.frame_menu, text="Table 3", command=lambda: self.cbk_db_table("Table3"),
                                           activebackground="#123345",
                                           borderwidth=0, bg="#CCCCCC", bd=1)
                    self.btn_table1.place(anchor="nw", relx=0.1, rely=0.1, width=100, height=25)
                    self.btn_table2.place(anchor="nw", relx=0.1, rely=0.19, width=100, height=25)
                    self.btn_table3.place(anchor="nw", relx=0.1, rely=0.28, width=100, height=25)
                elif self.curr_page == self.page[2]:
                    title = Label(self.frame_menu, text="Model", width=10, height=1, bg="#aaaaaa")
                    title.place(x=0, y=3)
                    self.btn_table1 = Button(self.frame_menu, text="Table 1", command=lambda: self.cbk_model("Table1"),
                                             activebackground="#123345",
                                             borderwidth=0, bg="#CCCCCC", bd=1)
                    self.btn_table2 = Button(self.frame_menu, text="Table 2", command=lambda: self.cbk_model("Table2"),
                                             activebackground="#123345",
                                             borderwidth=0, bg="#CCCCCC", bd=1)
                    self.btn_table3 = Button(self.frame_menu, text="Table 3", command=lambda: self.cbk_model("Table3"),
                                             activebackground="#123345",
                                             borderwidth=0, bg="#CCCCCC", bd=1)
                    self.btn_table1.place(anchor="nw", relx=0.1, rely=0.1, width=100, height=25)
                    self.btn_table2.place(anchor="nw", relx=0.1, rely=0.19, width=100, height=25)
                    self.btn_table3.place(anchor="nw", relx=0.1, rely=0.28, width=100, height=25)
                elif self.curr_page == self.page[3]:
                    title = Label(self.frame_menu, text="Report", width=10, height=1, bg="#aaaaaa")
                    title.place(x=0, y=3)
                elif self.curr_page == self.page[4]:
                    title = Label(self.frame_menu, text="Backup", width=10, height=1, bg="#aaaaaa")
                    title.place(x=0, y=3)

    def _set_main_body(self, db_data, md_data, rp_data, bu_data):
        """
        To set main page body contents
        :param db_data: dict. dictionary of data base information
        :param md_data: dict. dictionary of model information
        :param rp_data: dict. dictionary of report information
        :param bu_data: dict. dictionary of backup information
        :return: None
        """
        self._set_paragraph("DataBase", db_data, (10, 10), self.page[1])
        self._set_paragraph("Model", md_data, (250, 10), self.page[2])
        self._set_paragraph("Report", rp_data, (450, 10), self.page[3])
        self._set_paragraph("Backup", bu_data, (450, 150), self.page[4])

    def _set_db_body(self):
        # Table View
        height = 5
        width = 4
        # TODO: To add padding
        for i in range(height):  # Rows
            for j in range(width):  # Columns
                b = Entry(self.frame_body, text="")
                b.grid(row=i, column=j)

    def _set_model_body(self):
        md_data = {"Information": {"Updated Date": "2019.06.30", "# of Used Data": "25,482", "# of Features": "15",
                                   "Location of Model": "./FaceRecognition", "# of New Data": "4,250", "Accuracy": "80%"}}
        self._set_paragraph("", md_data, (200, 10), "")

    def _set_report_body(self):
        title = Label(self.frame_body, text=self.page[3])
        title.place(x=200, y=200)

    def _set_backup_body(self):
        title = Label(self.frame_body, text=self.page[4])
        title.place(x=200, y=200)

    def _set_paragraph(self, title, contents, location, i):
        margin = 20
        if len(i) == 0:
            margin = 0
        title = Label(self.frame_body, text=title)
        title.place(x=location[0], y=location[1])
        loc_y = location[1]+margin
        for k, vs in contents.items():
            title = Label(self.frame_body, text="  "+k)
            title.place(x=location[0], y=loc_y)
            for v_k, v_v in vs.items():
                loc_y += 20
                title = Label(self.frame_body, text="    "+str(v_k)+": "+str(v_v))
                title.place(x=location[0], y=loc_y)
            loc_y += 25
        print(i, len(i))
        if len(i) > 0:
            btn_nxt_page = Button(self.frame_body, width=3, height=1, command=lambda:self.cbk_change_page(i))
            btn_nxt_page.place(x=location[0]+120, y=location[1])

    def display_login_page(self):
        self.window_login = Toplevel(self.root)
        self.window_login.title("Log In")
        self.window_login.geometry("300x250")

        self.username = StringVar()
        self.password = StringVar()
        username_lable = Label(self.window_login, text="Username * ")
        username_lable.pack()
        username_entry = Entry(self.window_login, textvariable=self.username)
        username_entry.pack()
        password_lable = Label(self.window_login, text="Password * ")
        password_lable.pack()
        password_entry = Entry(self.window_login, textvariable=self.password, show='*')
        password_entry.pack()

        Button(self.window_login, text="Login", width=10, height=1, command = self.cbk_login).pack()

    def cbk_login(self):
        print(self.username.get(), self.password.get())
        # TODO: add login info check code
        self.is_session = True
        if self.is_session:
            self.session_info = {"id": self.username.get(), "password": self.password.get()}
            self.window_login.destroy()
            self.set_header()
            self.set_body()
            self.set_menu()

    def cbk_change_page(self, i):
        self.curr_page = i
        self.set_header()
        self.set_body()
        self.set_menu()

    def cbk_main(self, event):
        if self.is_session:
            self.curr_page = self.page[0]
            self.set_header()
            self.set_body()
            self.set_menu()

    def cbk_logout(self, event):
        self.is_session = False
        self.session_info = None
        self.set_header()
        self.set_body()
        self.set_menu()
        self.display_login_page()

    def cbk_db_table(self, t_name):
        pass

    def cbk_model(self, m_name):
        pass

    def run(self):
        """
        To run the code
        :return:
        """
        if not self.is_session:
            self.display_login_page()
        self.set_header()
        self.set_body()
        self.set_menu()

        #
        # frame2 = tkinter.Frame(self.root, relief="solid", bd=2)
        # frame2.pack(side="right", fill="both", expand=True)
        #
        # button1 = tkinter.Button(frame1, text="프레임1")
        # button1.pack(side="right")
        #
        # button2 = tkinter.Button(frame2, text="프레임2")
        # button2.pack(side="left")

        self.root.mainloop()


if __name__ == '__main__':
    sui = ServerUserInterface()
    sui.run()

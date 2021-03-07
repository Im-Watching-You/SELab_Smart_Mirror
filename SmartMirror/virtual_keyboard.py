import tkinter as tk

keys = [
    [
        # =========================================
        # ===== Keyboard Configurations ===========
        # =========================================

        # [
        #     # Layout Name
        #     ("Function_Keys"),
        #
        #     # Layout Frame Pack arguments
        #     ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
        #     [
        #         # list of Keys
        #         ('esc', " ", 'F1', 'F2', 'F3', 'F4', "", 'F5', 'F6', 'F7', 'F8', "", 'F9', 'F10', 'F11', 'F12')
        #     ]
        # ],

        [
            ("Character_Keys"),
            ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
            [
                ('@', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=',
                 '|\n\\', 'backspace'),
                ('tab', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '{\n[', '}\n]', '   '),
                ('capslock', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ':\n;', "\"\n'", "enter"),
                ("shift", 'z', 'x', 'c', 'v', 'b', 'n', 'm', '<\n,', '>\n.', '?\n/', "shift"),
                ("ctrl", "[+]", 'alt', '\t\tspace\t\t', 'alt', '[+]', '[=]', 'ctrl')
            ]
        ]
    ],
    # [
    #     [
    #         ("System_Keys"),
    #         ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
    #         [
    #             (
    #                 "print\nscreen\nsys",
    #                 "scroll\nlock",
    #                 "pause\nbreak"
    #             )
    #         ]
    #     ],
    #     [
    #         ("Editing_Keys"),
    #         ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
    #         [
    #             (
    #                 "insert",
    #                 "home",
    #                 "page\nup"
    #             ),
    #             ("delete",
    #              "end",
    #              "page\ndown"
    #              ),
    #         ]
    #     ],
    #
    #     [
    #         ("Navigation_Keys"),
    #         ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
    #         [
    #             (
    #                 "up",
    #             ),
    #             ("right",
    #              "down",
    #              "left"
    #              ),
    #         ]
    #     ],
    #
    # ],
    # [
    #
    #     [
    #         ("Numeric_Keys"),
    #         ({'side': 'top', 'expand': 'yes', 'fill': 'both'}),
    #         [
    #             ("num\nlock", "/", "*", "-"),
    #             ("7", "8", "9", "+"),
    #             ("4", "5", "6", " "),
    #             ("0", "1", "2", "3"),
    #             ("0", ".", "enter")
    #         ]
    #     ],
    #
    # ]

]


##  Frame Class
class Keyboard(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        # Function For Creating Buttons
        self.create_frames_and_buttons()
        self.event = None
        self.ent = None

    # Function For Extracting Data From KeyBoard Table
    # and then provide us a well looking
    # keyboard gui

    def close(self):
        tk.Frame.destroy()

    def add_entry(self, ent):
        self.ent = ent

    def create_frames_and_buttons(self):
        # take section one by one
        for key_section in keys:
            # create Sperate Frame For Every Section
            store_section = tk.Frame(self)
            store_section.pack(side='left', expand='yes', fill='both', padx=10, pady=10, ipadx=10, ipady=10)

            for layer_name, layer_properties, layer_keys in key_section:
                store_layer = tk.LabelFrame(store_section)  # , text=layer_name)
                # store_layer.pack(side='top',expand='yes',fill='both')
                store_layer.pack(layer_properties)
                for key_bunch in layer_keys:
                    store_key_frame = tk.Frame(store_layer)
                    store_key_frame.pack(side='top', expand='yes', fill='both')
                    for k in key_bunch:
                        k = k.capitalize()
                        if len(k) <= 3:
                            store_button = tk.Button(store_key_frame, text=k, width=2, height=2)
                        else:
                            store_button = tk.Button(store_key_frame, text=k.center(5, ' '), height=2)
                        if " " in k:
                            store_button['state'] = 'disable'
                        # flat, groove, raised, ridge, solid, or sunken
                        store_button['relief'] = "sunken"
                        store_button['bg'] = "powderblue"
                        store_button['command'] = lambda q=k: self.button_command(q)
                        store_button.pack(side='left', fill='both', expand='yes')
        return

        # Function For Detecting Pressed Keyword.

    def button_command(self, event):
        self.event = event.lower()
        if self.event == 'backspace':
            self.ent.delete(len(self.ent.get())-1)
        else:
            self.ent.insert(tk.END, self.get_command())

        return event

    def get_command(self):

        print(self.ent.get())
        if self.event is not None:
            if self.event == '\t\tspace\t\t':
                return " "
            elif self.event == "backspace":
                return '\b'
            else:
                return self.event
        else:
            return ""


if __name__ == '__main__':
    root = tk.Tk(className='Python Virtual KeyBoard')
    root.overrideredirect(1)
    key = Keyboard(root)
    key.pack()
    root.mainloop()

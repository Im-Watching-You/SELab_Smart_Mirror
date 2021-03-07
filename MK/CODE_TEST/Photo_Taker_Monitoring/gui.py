#!/usr/bin/python

# ---------------- READ ME ---------------------------------------------
# This Script is Created Only For Practise And Educational Purpose Only
# This Script Is Created For http://bitforestinfo.blogspot.com
# This Script is Written By
__author__ = '''

######################################################
                By S.S.B Group                          
######################################################

    Suraj Singh
    Admin
    S.S.B Group
    surajsinghbisht054@gmail.com
    http://bitforestinfo.blogspot.in/

    Note: We Feel Proud To Be Indian
######################################################
'''
# Import Module
try:
    import Tkinter
except:
    import tkinter as Tkinter
import path
from PIL import Image, ImageTk

# # Configurations
# # Enter Path Of Image Directories
# ImageDir = os.path.abspath(os.path.dirname(__file__))
# # print(dir_path)
# ImageDir = os.path.join(ImageDir, 'testdel')
#
# # ImageDir = ["/home/hackwithssb/Pictures/Wallpapers",
# # "/home/hackwithssb/Pictures",
# # "/home/hackwithssb/Pictures/BingWallpapers"]
#
# # Image Extensions Allowed
# Extension = ['JPG', 'BMP', 'PNG']
#
#
# def get_list():
#     Images = []
#     for ImageP in ImageDir:
#         for i in os.listdir(ImageP):
#             Image = os.path.join(ImageP, i)
#             ext = Image.split('.')[::-1][0].upper()
#             if ext in Extension:
#                 Images.append(Image)
#     return Images
#
#
# def tk_image(path, w, h):
#     img = Image.open(path)
#     img = img.resize((w, h))
#     storeobj = ImageTk.PhotoImage(img)
#     return storeobj
#

# Creating Canvas Widget
class PictureWindow(Tkinter.Canvas):
    def __init__(self, *args, **kwargs):
        Tkinter.Canvas.__init__(self, *args, **kwargs)
        self.imagelist = path.get_list()
        self.imagelist_p = []
        self.all_function_trigger()

    def show_image(self, path):
        img = path.tk_image(path, self.winfo_screenwidth(), self.winfo_screenheight())
        self.delete(self.find_withtag("bacl"))
        self.allready = self.create_image(self.winfo_screenwidth() / 2, self.winfo_screenheight() / 2, image=img,
                                          anchor='center', tag="bacl")

        self.image = img
        print
        self.find_withtag("bacl")
        self.master.title("Image Viewer ({})".format(path))
        return

    def previous_image(self):
        try:
            pop = self.imagelist_p.pop()
            self.show_image(pop)
            self.imagelist.append(pop)
        except:
            pass
        return

    def next_image(self):
        try:
            pop = self.imagelist.pop()

            self.show_image(pop)
            self.imagelist_p.append(pop)
        except EOFError as e:
            pass
        return

    def all_function_trigger(self):
        self.create_buttons()
        self.window_settings()
        return

    def window_settings(self):
        self['width'] = self.winfo_screenwidth()
        self['height'] = self.winfo_screenheight()
        return

    def create_buttons(self):
        Tkinter.Button(self, text=" > ", command=self.next_image).place(x=(self.winfo_screenwidth() / 1.1),
                                                                        y=(self.winfo_screenheight() / 2))
        Tkinter.Button(self, text=" < ", command=self.previous_image).place(x=20, y=(self.winfo_screenheight() / 2))
        self['bg'] = "white"
        return


# Main Function
def main():
    # Creating Window
    root = Tkinter.Tk(className=" Image Viewer")
    # Creating Canvas Widget
    PictureWindow(root).pack(expand="yes", fill="both")
    # Not Resizable
    root.resizable(width=0, height=0)
    # Window Mainloop
    root.mainloop()
    return


# Main Function Trigger
if __name__ == '__main__':
    main()
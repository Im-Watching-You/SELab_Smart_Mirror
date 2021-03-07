from tkinter import *
from PIL import ImageTk, Image
import os

root = Tk()
root.title('Image Viewer')

# my_img1 = ImageTk.PhotoImage(Image.open("testdel/sms_20190617_190538_0.png"))
# # my_img2 = ImageTk.PhotoImage(Image.open("D:\\SmartMirrorSystem\\SELab_Smart_Mirror\\MK\\CODE TEST\\Photo_Taker_Monitoring\\testdel\\sms_20190618_100302_0.png"))
# # my_img3 = ImageTk.PhotoImage(Image.open("D:\\SmartMirrorSystem\\SELab_Smart_Mirror\\MK\\CODE TEST\\Photo_Taker_Monitoring\\testdel\\sms_20190618_100626_0.png"))
# #
# # image_list = [my_img1, my_img2, my_img3]
# # print(image_list)
#
# # Enter Path Of Image Directories
# dir_path = os.path.dirname(__file__)
# # print(dir_path)
# dir_path = os.path.join(dir_path, 'testdel')
# # print(dir_path)
#
# image_list = []
# for img in os.listdir(dir_path):
#     img = os.path.join(dir_path, img)
#     if img is not None:
#         image_list.append(ImageTk.PhotoImage(Image.open(img)))
# print(image_list)
#
# def forward(image_number):
#     global my_label
#     global button_forward
#     global button_back
#
#     my_label.grid_forget()
#     my_label = Label(image=image_list[image_number-1])
#     button_forward = Button(root, text=">>", command=lambda: forward(image_number+1))
#     button_back = Button(root, text="<<", command=lambda: forward(image_number-1))
#
#     if image_number == len(image_list):
#         button_forward = Button(root, text=">>", state=DISABLED)
#
#     my_label.grid(row=0, column=0, columnspan=3)
#     button_back.grid(row=1, column=0)
#     button_forward.grid(row=1, column=2)
#
#
# def back(image_number):
#     global my_label
#     global button_forward
#     global button_back
#
#     my_label.grid_forget()
#     my_label = Label(image=image_list[image_number-1])
#     button_forward = Button(root, text=">>", command=lambda: forward(image_number+1))
#     button_back = Button(root, text="<<", command=lambda: forward(image_number-1))
#
#     if image_number == 1:
#         button_back = Button(root, text="<<", state=DISABLED)
#
#     my_label.grid(row=0, column=0, columnspan=3)
#     button_back.grid(row=1, column=0)
#     button_forward.grid(row=1, column=2)
#
#
# my_label = Label(image=image_list[0])
# my_label.grid(row=0, column=0, columnspan=3)
#
# button_back = Button(root, text="<<", command=back, state=DISABLED)
# button_exit = Button(root, text="Exit", command=root.quit)
# button_forward = Button(root, text=">>", command=lambda: forward(2))
#
# button_back.grid(row=1, column=0)
# button_exit.grid(row=1, column=1)
# button_forward.grid(row=1, column=2)
#
# root.mainloop()
# # get_list()


# my_img1 = ImageTk.PhotoImage(Image.open("testdel/sms_20190617_190538_0.png"))
# my_img2 = ImageTk.PhotoImage(Image.open("D:\\SmartMirrorSystem\\SELab_Smart_Mirror\\MK\\CODE TEST\\Photo_Taker_Monitoring\\testdel\\sms_20190618_100302_0.png"))
# my_img3 = ImageTk.PhotoImage(Image.open("D:\\SmartMirrorSystem\\SELab_Smart_Mirror\\MK\\CODE TEST\\Photo_Taker_Monitoring\\testdel\\sms_20190618_100626_0.png"))
#
# image_list = [my_img1, my_img2, my_img3]
# print(image_list)


class MyTest:
    def __init__(self):
        # Enter Path Of Image Directories
        self.dir_path = os.path.dirname(__file__)
        self.dir_path = os.path.join(self.dir_path, 'testdel')
        self.image_list = self.get_list()
        self.gui_initiation()

    def gui_initiation(self):
        self.my_label = Label(image=self.image_list[0])
        self.my_label.grid(row=0, column=0, columnspan=3)

        self.button_back = Button(root, text="<<", command=lambda: self.back(), state=DISABLED)
        self.button_exit = Button(root, text="Exit", command=root.quit)
        self.button_forward = Button(root, text=">>", command=lambda: self.forward(2))

        self.button_back.grid(row=1, column=0)
        self.button_exit.grid(row=1, column=1)
        self.button_forward.grid(row=1, column=2)
        return

    def get_list(self):
        images = []
        for img in os.listdir(self.dir_path):
            img = os.path.join(self.dir_path, img)
            if img is not None:
                images.append(ImageTk.PhotoImage(Image.open(img)))
        return images

    def forward(self, image_number):
        global my_label
        global button_forward
        global button_back

        self.my_label.grid_forget()
        self.my_label = Label(image=self.image_list[image_number-1])
        self.button_forward = Button(root, text=">>", command=lambda: self.forward(image_number+1))
        self.button_back = Button(root, text="<<", command=lambda: self.back(image_number-1))

        if image_number == len(self.image_list):
            button_forward = Button(root, text=">>", state=DISABLED)

        my_label.grid(row=0, column=0, columnspan=3)
        button_back.grid(row=1, column=0)
        button_forward.grid(row=1, column=2)
        return

    def back(self, image_number):
        global my_label
        global button_forward
        global button_back

        self.my_label.grid_forget()
        self.my_label = Label(image=self.image_list[image_number-1])
        self.button_forward = Button(root, text=">>", command=lambda: self.forward(image_number+1))
        self.button_back = Button(root, text="<<", command=lambda: self.back(image_number-1))

        if image_number == 1:
            self.button_back = Button(root, text="<<", state=DISABLED)

        self.my_label.grid(row=0, column=0, columnspan=3)
        self.button_back.grid(row=1, column=0)
        self.button_forward.grid(row=1, column=2)
        return


#
# class PhotoLoading:
#     def __init__(self):
#         # self.dir_path = self.get_pictures()
#         self.image_list = self.get_pictures()
#
#     @staticmethod
#     def get_pictures(user_id=0, mode=''):
#         """
#          Retrieve pictures from the passed user folder for automatic mode or manual mode
#          """
#         dir_path = os.path.abspath(os.path.dirname(__file__))
#         img_list = []
#         # print(dir_path)
#         if mode == 'manual':
#             dir_path = os.path.join(dir_path, 'capture\\manual\\{}'.format(user_id))
#             if not os.path.isdir(dir_path):
#                 os.mkdir(dir_path)
#             for img in os.listdir(dir_path):
#                 img = os.path.join(dir_path, img)
#                 if img is not None:
#                     img_list.append(ImageTk.PhotoImage(Image.open(img)))
#             # print(img_list)
#         elif mode == 'auto':
#             dir_path = os.path.join(dir_path, 'capture\\automatic\\{}'.format(user_id))
#             if not os.path.isdir(dir_path):
#                 os.mkdir(dir_path)
#             for img in os.listdir(dir_path):
#                 img = os.path.join(dir_path, img)
#                 if img is not None:
#                     img_list.append(img)
#             # print(img_list)
#         # print(dir_path)
#         return img_list
#



# Main Function Trigger
if __name__ == '__main__':
    # photo_load = PhotoLoading()
    # photo_load.get_pictures(mode='manual')
    # list_img = photo_load.get_pictures(mode='auto')
    # print('*******************')
    # print(list_img)
    T = MyTest()
    root.mainloop()

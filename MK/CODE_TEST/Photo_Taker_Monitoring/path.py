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
import os
from PIL import Image, ImageTk



def get_list():
    # Configurations
    # Enter Path Of Image Directories
    dir_path = os.path.abspath(os.path.dirname(__file__))
    # print(dir_path)
    dir_path = 'D:\\SmartMirrorSystem\\SELab_Smart_Mirror\\MK\\CODE TEST\\Photo_Taker_Monitoring\\testdel'  # os.path.join(ImageDir, 'Photo_Taker_Monitoring\\testdel')
    print(dir_path)
    # ImageDir = ["/home/hackwithssb/Pictures/Wallpapers",
    # "/home/hackwithssb/Pictures",
    # "/home/hackwithssb/Pictures/BingWallpapers"]

    images = []
    for img in os.listdir(dir_path):
        img = os.path.join(dir_path, img)
        if img is not None:
            images.append(img)
    return images


def tk_image(path,w,h):
  img = Image.open(path)
  img = img.resize((w,h))
  storeobj = ImageTk.PhotoImage(img)
  return storeobj
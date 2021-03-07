import os
import stat
import time
from os import listdir
from os.path import isfile, join
import cv2

from datetime import datetime

#
# now = datetime.now()
# print(now)
# print(now.strftime('%Y-%m-%d_%H%M%S'))
# print (now.date().strftime('%Y%m%d'))
#
print("***************************************")
# print(os.getcwd())
# print(os.path.basename(__file__))                      # fil name
absFilePath = os.path.abspath(__file__)  # Absolute Path of the module
print('absFilePath: ' + absFilePath)
fileDir = os.path.dirname(os.path.abspath(__file__))  # Directory of the Module
print('dir module: ' + fileDir)
parentDir = os.path.dirname(fileDir)  # Directory of the Module directory
print('dir module dir: ' + parentDir)

newPath = os.path.join(parentDir, 'StringFunctions')  # Get the directory for StringFunctions
print('parentdir+adjust: ' + newPath)

print("***************************************")
dirname = os.path.dirname(__file__)
print(dirname)
filename = os.path.join(dirname, 'relative/path/to/file/you/want')
print(filename)

print("***************************************")
my_path = os.path.abspath(os.path.dirname(__file__))
print(my_path)
path = os.path.join(my_path, "../data/test.csv")
print(path)
print("***************************************")
path = os.path.dirname(__file__)
print('Test 1: ', path)
os.path.basename(path)
print('test2:', os.path.basename(path))
print("***************************************")
#
# print("***************************************")
# onlyfiles = [f for f in listdir(fileDir) if isfile(join(fileDir, f))]
# print(onlyfiles)
#
# print("***************************************")
# for root, dirs, files in os.walk("."):
#     for filename in files:
#             if '.png' in filename:
#                 print(filename)

print("***************************************")


def delete_pictures(user_id=0):
    """
     Removes files from the passed in path that are older than or equal
     to the number of minutes or days
     """
    dir_path = os.path.abspath(os.path.dirname(__file__))
    print(dir_path)
    dir_path = os.path.join(dir_path, 'testdel')
    print('Dossier parent :' + dir_path)
    # convert age to sec. 1 day = 24*60*60
    # age = int(2) * 86400
    age = int(30) * 60
    for the_file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, the_file)
        try:
            if os.path.isfile(file_path):
                # print(file_path)
                # Get the file creation time
                fileStatsObj = os.stat(file_path).st_ctime
                # print(fileStatsObj)
                if fileStatsObj <= time.time() - age:
                    if os.path.isfile(file_path):
                        print(file_path)
                        creationTime = time.ctime(fileStatsObj)
                        print("File Creation Time : " + str(creationTime))
                        # os.remove(file_path)
        except Exception as e:
            print(e)


# delete_pictures()


def load_images(dir_path):
    """
     Retrieve images from either automatic folder for training  or manual folder to display
     """
    print('Dossier parent :' + dir_path)

    images = []
    for img in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            cv2.imshow('image', img)
            k = cv2.waitKey(0)
            if k == 27:  # esc key
                cv2.destroyAllWindows()
                break
    return images


dir_path = os.path.abspath(os.path.dirname(__file__))
# print(dir_path)
dir_path = os.path.join(dir_path, 'testdel')

# load_images(dir_path)

"""
Date: 2019.05.27
Programmer: MK
Description: class managing the photo taker module
"""
from skimage.measure import compare_ssim as ssim
import cv2
import datetime
import os
import glob
import time
from ActiveAgingAdvisorySystem.photo import Photo


class PhotoTaker:
    def __init__(self, face_recognizer=None):
        self.img_counter = 0
        self.img_threshold = 58
        self.img_age = 30
        self.face_recognizer = face_recognizer
        # self.dir_path = os.path.abspath(os.path.dirname(__file__))
        self.cur_pt_info = None
        self.db_pt = Photo()
        self.interval = 3
        self.last_taken_time = 0

    def count_new_photo(self, user_id):
        """
        Count the number of pictures captured in the automatic folder and call
        the face recognition training method if it is more or equal to the the threshold.
        """
        dir_path = os.path.abspath(os.path.dirname(__file__))
        dir_path = os.path.join(dir_path, "capture\\automatic\\{}".format(str(user_id)))
        # today = datetime.now().date().strftime('%Y%m%d')
        # for root, dirs, files in os.walk(dir_path):
        #     for filename in files:
        #             if today in filename:
        #                 self.img_counter += 1
        # print(self.img_counter)
        self.img_counter = sum([len(files) for r, d, files in os.walk(dir_path)])
        # print(self.img_counter)
        if self.img_counter == self.img_threshold:
            pass
            # print()
            # print("The model need to be retrain with the {} new photo".format(self.img_counter))
            # self.face_recognizer.train_model()

    @staticmethod
    def detect_similar_photo(new_img, user_id):
        """
        Count the number of pictures captured in the automatic folder and call
        the face recognition training method if it is more or equal to the the threshold.
        """
        # dir_path = '.\\capture\\{}\\automatic\\'.format(str(user_id))s
        file_path = '.\\capture\\automatic\\'
        dir_path = file_path+str(user_id)
        pre_img = ''
        # index for the images
        # using the Structural Similarity Index
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        file_list = glob.glob(dir_path + '*.*')
        if file_list:
            # print(file_list)
            if len(file_list) < 5:
                # If there are less than five pictures, compare all the pictures with the new one
                for file in file_list:
                    pre_img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
            else:
                # If there are more than 5 pictures in the folder, compare the last five pictures with the new one
                file_list = file_list[len(file_list) - 5:]
                # print(file_list)
                for file in file_list:
                    pre_img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
            # print(pre_img)
            s = ssim(new_img, pre_img)
            # print("SSIM: %.2f" % s)
            # Returns True if the similarity is greater than 80 percent.
            if s >= 0.7:
                return True
            else:
                return False

    def save_img(self, frame, info={}, mode='auto'):
        """
        Method to save photos. Only proceed if user is registered and logged in.
        In the auto save mode, it checks whether there is a similar picture in the user's folder,
        and if not, it executes the save method.
        """
        if time.time() - self.last_taken_time >= self.interval:
            user_id = info['user_id']
            sess_id = info['session_id']
            now = datetime.datetime.now()
            if mode and user_id >= -1:
                # Process only if a user has been identified
                if user_id < -1:
                    return
                if mode == 'auto':
                    #  Crop image and save it for model training
                    # add code to call the crop function
                    # Checking similarity of the new photo and existing ones
                    result = self.detect_similar_photo(frame, user_id)
                    if result:
                        # print('Similar photo already exist, captured image will not be saved (Auto)')
                        pass
                    else:
                        # If a similar photo does not exist, execute saving Image process
                        file_path = '.\\capture\\automatic\\'
                        if not os.path.isdir(file_path + str(user_id)):
                            os.mkdir(file_path + str(user_id))
                        for f in os.listdir(file_path + str(user_id)):
                            # ??????
                            f_name = os.path.splitext(f)[0]
                            t = f_name.split("_")
                            if t[2] == str(now.strftime('%H%M%S')):
                                return
                            d_time = datetime.datetime.strptime(str(t[1]) + "_" + str(t[2]), '%Y%m%d_%H%M%S')
                            dd = datetime.datetime.timestamp(d_time)
                            if mode == 'detection' and time.time() - dd > 600:
                                os.remove(file_path + str(user_id) + "\\" + f)
                        filename = file_path + str(user_id) + '\\sms_{}_{}_{}.png'.format(now.strftime("%Y%m%d"),
                                                                                          now.strftime("%H%M%S"),
                                                                                          user_id)
                        cv2.imwrite(filename, frame)
                        self.db_pt.register_photo({'session_id': sess_id, "saved_path": filename})
                        # print('New photo saved! (Auto Mode)')
                        self.count_new_photo(user_id)
                        # self.delete_pictures()
                elif mode == 'manual':
                    # manual saving when user click photo icon
                    file_path = '.\\capture\\manual\\'
                    if not os.path.isdir(file_path + str(user_id)):
                        os.mkdir(file_path + str(user_id))
                    filename = file_path + str(user_id) + '\\sms_{}_{}_{}.png'.format(now.strftime("%Y%m%d"),
                                                                                      now.strftime("%H%M%S"),
                                                                                      user_id)
                    msg = 'Saving Photo (Manual Mode)'
                    cv2.imwrite(filename, frame)
                    self.db_pt.register_photo({'session_id': sess_id, "saved_path": filename})
                    # print(msg)
                elif mode == "face":
                    file_path = '.\\dataset\\train\\'
                    if not os.path.isdir(file_path + str(user_id)):
                        os.mkdir(file_path + str(user_id))
                    filename = file_path + str(user_id) + '\\sms_{}_{}_{}.png'.format(now.strftime("%Y%m%d"),
                                                                                      now.strftime("%H%M%S"),
                                                                                      user_id)
                    msg = 'Saving Photo (Manual Mode)'
                    cv2.imwrite(filename, frame)
                    self.db_pt.register_photo({'session_id': sess_id, "saved_path": filename})
            else:
                # print('The user must be registered to be able to use photo capture module.')
                pass
            self.last_taken_time = time.time()

    def get_curr_photo_info(self, sess_id):
        return self.db_pt.retrieve_latest_photo_id(sess_id)

    @staticmethod
    def delete_pictures(user_id=0, age=30):
        """
         Removes files from the passed user folder that are aged than or equal
         to the number of minutes or days inputted
        """
        dir_path = os.path.abspath(os.path.dirname(__file__))
        # print(dir_path)
        dir_path = os.path.join(dir_path, 'capture\\automatic\\{}'.format(user_id))
        # print('Dossier parent :' + dir_path)
        # convert age to sec. 1 day = 24*60*60
        # age = int(age) * 86400      # x days old with x=age
        age = int(age) * 60  # 30 minutes old
        for the_file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, the_file)
            try:
                if os.path.isfile(file_path):
                    # print(file_path)
                    # Get the file creation time
                    img_creation_time = os.stat(file_path).st_ctime
                    # print(img_creation_time)
                    if img_creation_time <= time.time() - age:
                        if os.path.isfile(file_path):
                            print(file_path)
                            creation_time = time.ctime(img_creation_time)
                            print("File Creation Time : " + str(creation_time))
                            # os.remove(file_path)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    pass

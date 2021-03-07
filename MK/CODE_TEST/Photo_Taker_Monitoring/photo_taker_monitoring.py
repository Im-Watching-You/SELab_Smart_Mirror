from skimage.measure import compare_ssim as ssim
import time
import cv2
import os, os.path
from datetime import datetime

class PhotoTaker:
    img_counter = 0
    sec = 0
    interval = 0
    threshold = 15

    def getInter(self):
        return self.interval

    def setThreshold(self, threshold):
        self.threshold = threshold

    def getThreshold(self):
        return self.threshold

    def setInter(self, interval):
        self.interval = interval

    def capture(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Smart Mirror")

        while True:
            ret, frame = cam.read()
            cv2.imshow("Smart Mirror", frame)
            if not ret:
                break
            k = cv2.waitKey(1) & 0xFF

            if k == 27:                                                          # ESS exit the program
                print("Escape hit, closing...")
                break

            elif k == ord(' '):                                                  # manual capture if Spacebar is pressed
                now = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_name = "picture_{}_{}.png".format(now, self.img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))

                self.CountNewPhoto()
                self.sec = time.time()                                           # assign current time value to sec
                self.img_counter += 1
                continue

            elif time.time()-self.sec > self.getInter():                          # Auto Picture Capture every 10 second
                now = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_name = "picture_{}_{}.png".format(now, self.img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))

                self.CountNewPhoto()
                self.sec = time.time()                                            # assign current time value to sec
                self.img_counter += 1

        cam.release()
        cv2.destroyAllWindows()


    # Method to count the number of new pictures captured in a special day
    # return True if New pictures number reach Threshold value, and call the method to update the model.
    def CountNewPhoto(self):
        file_count = 0
        today = datetime.now().date().strftime('%Y%m%d')
        #
        # piclist = len([pic for pic in os.listdir('.') if os.path.isfile(pic) and today in os.path.isfile(pic)])
        # print(piclist)

        for root, dirs, files in os.walk("."):
            for filename in files:
                    if today in filename:
                        file_count += 1
        #print(file_count)
        if file_count == self.threshold:
            print("The model need to be retrain with the {} new photo".format(file_count))      # Trigger the Event
#
#
# ptaker = PhotoTaker()
# ptaker.setInter(5)                 # Set the interval value
# ptaker.capture()
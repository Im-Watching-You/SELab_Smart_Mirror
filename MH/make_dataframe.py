import math
import os
import cv2
import dlib
from imutils import face_utils
import pandas as pd
import numpy as np


class AgingFactorDetector:
    def __init__(self):
        self.data_dir = "./FGNet_images"
        self.count = 0
        self.val_count = 0
        self.w = 220
        self.h = 220
        self.file_list = os.listdir(self.data_dir)

        self.detector = dlib.get_frontal_face_detector()

        p = "./shape_predictor_81_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(p)
        self.file_list.sort()
        self.frame = {"F1": [],
                 "F2": [],
                 "F3": [],
                 "F4": [],
                 "F5": [],
                 "F6": [],
                 "F7": [],
                 "F8": [],
                 "F9": [],
                 "F10": [],
                 'F11': [],
                 'F12': [],
                 "Age": [],
                 "path": []}

    def compute_distance(self, p1, p2):
        distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        return distance

    def compute_middle(self, p1, p2):
        p3 = ((p2[0] + p1[0]) / 2, (p2[1] + p1[1]) / 2)
        return p3

    def set_w_h(self, w, h):
        self.w = w
        self.h = h

    def compute_factors(self, shape):
        result = {"F1": round(self.compute_distance(shape[45], shape[36])/ self.width, 2),
                  "F2": round(self.compute_distance(shape[42], shape[39])/ self.width, 2),
                  "F3": round(self.compute_distance(self.compute_middle(shape[46], shape[43]),
                                              self.compute_middle(shape[40], shape[37]))/ self.width, 2),
                  "F4": round(self.compute_distance(shape[44], shape[43])/ self.width, 2),
                  "F5": round(self.compute_distance(shape[35], shape[31])/ self.width, 2),
                  "F6": round(self.compute_distance(shape[54], shape[48])/ self.width, 2),
                  "F7": round(self.compute_distance(shape[27], shape[33])/ self.height, 2),
                  "F8": round(self.compute_distance(shape[27], shape[8])/ self.height, 2),
                  "F9": round(self.compute_distance(shape[33], shape[8])/ self.height, 2),
                  "F10": round(self.compute_distance(shape[33], shape[27])/ self.height, 2),
                  'F11': round(self.compute_distance(shape[33], shape[51])/ self.height, 2),
                  'F12': round(self.compute_distance(shape[33], shape[51])/ self.height, 2)}

        return result

    def load_csv(self, path):
        return pd.read_csv(path)

    def preprocessing(self, face_shape=None):
        self.set_w_h(300, 300)
        count = 0
        df_age = self.load_csv('./age_FGNet.csv')
        for i in self.file_list:
            try:
                self.count += 1
                img = cv2.imread(self.data_dir+"/"+i, cv2.IMREAD_COLOR)
                # shape = img.shape
                age = df_age.loc[df_age['SampleID']==i]["Age"].values[0]
                # if shape[0] < 300 or shape[1] < 300:        # check image size
                #     continue
                #
                # f = i.split("_")
                # birth_year = int(f[1].split("-")[0])
                # taken_year = int(f[2].split('.')[0])
                # age = taken_year - birth_year
                # if age > 100 or age < 0:                # check age (0~100)
                #     continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)          # detect faces
                if len(rects) != 1:                     # check only one person
                    continue

                target_face = rects[0]
                self.height = target_face.height()
                self.width = target_face.width()
                # if self.height < 150 or self.width < 150:         # check face part size to detect
                #     continue

                face_shape = self.predictor(gray, target_face)      # find facial landmarks
                mtx_landmarks = np.matrix([[p.x, p.y] for p in face_shape.parts()])
                face_shape = np.squeeze(np.asarray(mtx_landmarks))
                rate = self.compute_distance(face_shape[28], face_shape[0]) \
                       / self.compute_distance(face_shape[0], face_shape[16])
                if 0.40 > rate or rate > 0.60:          # check frontal face
                    continue
                for (x, y) in face_shape:
                    cv2.circle(img,  (x, y), 3, (0, 255, 0), -1)
                    # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
                cv2.rectangle(img, (rects[0].left(), rects[0].top()), (rects[0].right(), rects[0].bottom()), (0, 0, 255), 4)
                img = img[target_face.top()-40:target_face.bottom()+40,
                      target_face.left()-40:target_face.right()+40]   # Crop face part include hear and neck
                self.width = self.compute_distance(face_shape[16], face_shape[0])
                self.height = self.compute_distance(face_shape[8],
                                                    (face_shape[71][0], self.find_lowest_value([face_shape[71][0], face_shape[70][0], face_shape[80][0], face_shape[72][0], face_shape[69][0]])))
                img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_CUBIC)    # Resize detected face
                self.val_count += 1
                result = self.compute_factors(face_shape)
                cv2.imwrite("./results/"+i, img)     # Save chosen image
                print(round((self.count/len(self.file_list))*100, 2))

                self._append_data(result, age, i)
            except:
                count += 1
                print(i, count)
                continue
        print(">>> ", count, len(self.file_list))
    def find_lowest_value(self,l):
        l = sorted(l)
        return l[0]

    def extract_face_landmark(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, img)  # find facial landmarks
        shape = face_utils.shape_to_np(shape)
        return shape

    def set_face_width_height(self, face_shape):
        self.width = self.compute_distance(face_shape[16], face_shape[0])
        self.height = self.compute_distance(face_shape[8],
                                            (face_shape[71][0], self.find_lowest_value(
                                                [face_shape[71][0], face_shape[70][0], face_shape[80][0],
                                                 face_shape[72][0], face_shape[69][0]])))

    def _append_data(self, result, age, img_path):
        self.frame['F1'].append(result['F1'])
        self.frame['F2'].append(result['F2'])
        self.frame['F3'].append(result['F3'])
        self.frame['F4'].append(result['F4'])
        self.frame['F5'].append(result['F5'])
        self.frame['F6'].append(result['F6'])
        self.frame['F7'].append(result['F7'])
        self.frame['F8'].append(result['F8'])
        self.frame['F9'].append(result['F9'])
        self.frame['F10'].append(result['F10'])
        self.frame['F11'].append(result['F11'])
        self.frame['F12'].append(result['F12'])
        self.frame['Age'].append(age)
        self.frame["path"].append("./results/" + img_path)

    def make_dataframe(self):
        return pd.DataFrame(self.frame)

    def write_csv_file(self, frame_df, path="data, FGN, 0730c.csv"):
        frame_df.to_csv(path, mode='a')

    def compute_corr(self, frame_df):
        print(frame_df.corr())
        print(frame_df.corr()["Age"].sort_values(ascending=False))
        #     print(type(i))

    def get_data_size(self):
        print("Count: ", self.val_count)
        print("Length: ", len(self.file_list))


if __name__ == '__main__':
    afd = AgingFactorDetector()
    print("Preprocessing")
    afd.preprocessing()
    print("Make dataframe")
    df = afd.make_dataframe()
    print("write csv file")
    afd.write_csv_file(df)
    print("compute correlation coefficient")
    afd.compute_corr(df)
    print("get data size")
    afd.get_data_size()

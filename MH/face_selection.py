import os
import cv2
import dlib
from imutils import face_utils
import pandas as pd
import numpy as np
import math
from sklearn.externals import joblib

class FacePreprocessor:

    def __init__(self):
        self.list_dataset = ['imdb_crop', 'wiki_crop']
        self.db_location = "D:\\1. Lab\\Dataset\\Age\\"

        self.detector = dlib.get_frontal_face_detector()
        p = "shape_predictor_81_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(p)
        self.age_predicter = joblib.load("./aging_factor/SVC.pkl")

    def compute_distance(self, p1, p2):
        distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        return distance

    def compute_middle(self, p1, p2):
        p3 = ((p2[0] + p1[0]) / 2, (p2[1] + p1[1]) / 2)
        return p3

    def compute_area(self, w, h, type):
        result = 0
        if type=="Square":
            result= w*h
        elif type == "Triangle":
            result = w*h/2
        return result

    def run(self):
        count = 0
        con_1_count = 0
        con_2_count = 0
        con_3_count = 0
        con_4_count = 0
        for db in self.list_dataset:
            list_folder = os.listdir(self.db_location+db)
            for fol in list_folder:
                print(fol)
                if db == "imdb_crop":
                    if int(fol) < 43:
                        continue
                list_cand = os.listdir(self.db_location+db+"\\"+fol)
                for ni in list_cand:
                    count += 1
                    # To check age
                    f = ni.split("_")
                    if db == self.list_dataset[0]:
                        birth_year = int(f[2].split("-")[0])
                        taken_year = int(f[3].split('.')[0])
                    else:
                        birth_year = int(f[1].split("-")[0])
                        taken_year = int(f[2].split('.')[0])
                    age = taken_year - birth_year
                    if age > 100 or age < 0:  # check age (0~100)
                        continue
                    con_1_count += 1

                    # To load image
                    img = cv2.imread(self.db_location + db + "\\" + fol + "\\" + ni, cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    rects = self.detector(gray, 0)
                    # To check # of faces in the image
                    if len(rects) != 1:
                        # print(self.db_location + db + "\\" + fol +"\\"+ni, "    ", len(rects))
                        continue
                    con_2_count += 1

                    curr_face = rects[0]
                    if curr_face.height() < 150 or curr_face.width() < 150:
                        continue
                    con_3_count += 1

                    face_shape = self.predictor(gray, curr_face)  # find facial landmarks
                    face_shape = face_utils.shape_to_np(face_shape)
                    rate = self.compute_distance(face_shape[28], face_shape[0]) \
                           / self.compute_distance(face_shape[0], face_shape[16])
                    if 0.48 > rate or rate > 0.52:  # check frontal face
                        continue

                    con_4_count += 1

                    cv2.imwrite("D:\\2. Project\\Python\\SELab_Smart_Mirror\\ActiveAgingAdvisorySystem\\resultss\\" + ni, img)  # Save chosen image
        print(count)
        print(con_1_count, "    ", con_2_count, "    ", con_3_count, "    ", con_4_count)
        print(round(con_1_count / count, 2), "    ", round(con_2_count / count, 2), "    ", round(con_3_count / count, 2),
              "    ", round(con_4_count / count, 2))

    def compute_factors(self, shape, w, h):
        result = {"F1": round(self.compute_distance(shape[15], shape[1]) / w, 2),
                  "F2": round(self.compute_distance(shape[45], shape[36]) / w, 2),
                  "F3": round(self.compute_distance(shape[42], shape[39]) / w, 2),
                  "F4": round(self.compute_distance(self.compute_middle(shape[46], shape[43]),
                                                    self.compute_middle(shape[40], shape[37])) / w, 2),
                  "F5": round(self.compute_distance(shape[44], shape[43]) / w, 2),
                  "F6": round(self.compute_distance(shape[35], shape[31]) / w, 2),
                  "F7": round(self.compute_distance(shape[54], shape[48]) / w, 2),
                  "F8": round(self.compute_distance(shape[27], shape[33]) / h, 2),
                  "F9": round(self.compute_distance(shape[27], shape[8]) / h, 2),
                  "F10": round(self.compute_distance(shape[33], shape[8]) / h, 2),
                  "F11": round(self.compute_distance(shape[33], shape[27]) / h, 2),
                  'F12': round(self.compute_distance(shape[33], shape[51]) / h, 2),
                  'F13': round(self.compute_area(self.compute_distance(shape[27], shape[33]),
                                                 self.compute_distance(shape[31], shape[35]), "Triangle")
                               / self.compute_area(w, h, "Square"), 2)}

        return result

    def extract_face_landmark(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, img)  # find facial landmarks
        shape = face_utils.shape_to_np(shape)
        return shape

    def compute_age_factors(self, shape):
        mtx_landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        landmarks = np.squeeze(np.asarray(mtx_landmarks))
        width = self.compute_distance(landmarks[15], landmarks[1])
        height = self.compute_distance(landmarks[8], (landmarks[1][0], min(landmarks[72][1], landmarks[69][1])))
        r = self.compute_factors(landmarks, width, height)
        # Add Path
        # r['path'] =
        # Age detection code
        r['age'] = self.age_predicter(r.values())

        return {"Age": 0, "Factor":r}

    def train(self):
        pass

    def predict(self, r):

        return 0

    def renaming(self):
        count = 1
        loc = "D:\\2. Project\\Python\\SELab_Smart_Mirror\\ActiveAgingAdvisorySystem\\resultss\\"
        list_folder = os.listdir(loc)
        for i in list_folder:
            img = cv2.imread(loc+i, cv2.IMREAD_COLOR)

            f = i.split("_")
            name = str(count)+"_"+f[-2]+"_"+f[-1]
            cv2.imwrite("./resultss/"+name, img)
            count += 1

    def compute_age(self):
        p = "shape_predictor_81_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)
        loc = ".\\resultss\\"
        result= {"F1": [],
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
                 'F12': [],"F13":[], 'Age':[], "path":[]}
        list_folder = os.listdir(loc)
        idx = 0
        for i in list_folder:
            idx += 1
            print(str(round((idx/13067)*100, 2))+"%")
            img = cv2.imread(loc+i, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            rect = []
            try:
                rect = rects[0]
            except:
                print(i)
                continue
            shape = predictor(gray, rect)
            mtx_landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
            landmarks = np.squeeze(np.asarray(mtx_landmarks))
            width = self.compute_distance(landmarks[15], landmarks[1])
            height = self.compute_distance(landmarks[8], (landmarks[71][0], min(landmarks[72][1], landmarks[69][1])))
            r = self.compute_factors(landmarks, width, height)
            result['path'].append(loc+i)

            f = i.split("_")
            birth_year = int(f[1].split("-")[0])
            taken_year = int(f[2].split('.')[0])
            result['Age'].append(taken_year-birth_year)
            result['F1'].append(r['F1'])
            result['F2'].append(r['F2'])
            result['F3'].append(r['F3'])
            result['F4'].append(r['F4'])
            result['F5'].append(r['F5'])
            result['F6'].append(r['F6'])
            result['F7'].append(r['F7'])
            result['F8'].append(r['F8'])
            result['F9'].append(r['F9'])
            result['F10'].append(r['F10'])
            result['F11'].append(r['F11'])
            result['F12'].append(r['F12'])
            result['F13'].append(r['F13'])
        return result

    def make_dataframe(self, result):
        return pd.DataFrame(result)

    def write_csv_file(self, frame_df, path="data,0724a.csv"):
        frame_df.to_csv(path, mode='a')


if __name__ == '__main__':
    fp = FacePreprocessor()
    # fp.run()
    # fp.renaming()
    result = fp.compute_age()
    df = fp.make_dataframe(result)
    fp.write_csv_file(df)
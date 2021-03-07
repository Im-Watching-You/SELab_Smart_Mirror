"""
Date: 2019. 06. 11
Programmer: MH
Description:
"""
import threading

import tensorflow as tf
import pickle
import cv2
import os
from scipy import misc
import numpy as np
from packages import facenet, detect_face
from os import listdir
from packages.classifier import Training


class Detector:
    def __init__(self):
        pass

    def _prepare_model(self):
        pass

    def predict(self, obj):
        pass

    def train_model(self):
        pass

    def estimate_model(self):
        accuracy = 0
        return accuracy


class FaceDetector:
    def __init__(self):
        self.dir_model = './models/20170511-185253.pb'
        self.npy = ''
        self.nrof_faces = 0
        self._prepare_model()

    def load_cand(self):
        pass

    def _prepare_model(self):
        # To prepare registered people list

        # self.load_cand()
        # To prepare recognition model
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, self.npy)

                self.minsize = 20  # minimum size of face
                self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                self.factor = 0.709  # scale factor
                self.image_size = 182
                self.input_image_size = 160

                print('Loading Modal')
                facenet.load_model(self.dir_model)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

    def detect(self, frame):
        # def th(frame):
        try:
            # To make RGB image from grayscale or BGR Image
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            frame = frame[:, :, 0:3]
            # To detect face
            self.bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet,
                                                             self.threshold, self.factor)
            self.nrof_faces = self.bounding_boxes.shape[0]
            # print('Detected_FaceNum: %d' % self.nrof_faces)
        except:
            print("ERROR")
            self.nrof_faces = None
        # threading.Thread(target=th, args=(frame,)).start()

    def choose_face(self, frame, x=None, y=None):
        face = None
        # try:
        if self.nrof_faces is None:
            return frame, None
        if self.nrof_faces > 0:
            det = self.bounding_boxes[:, 0:4]
            bb = np.zeros((self.nrof_faces, 4), dtype=np.int32)
            for i in range(self.nrof_faces):
                self.emb_array = np.zeros((1, self.embedding_size))
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                if self.nrof_faces > 1:    # more than two people are detected
                    if x is None and y is None:
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                    elif bb[i][0] <= x <= bb[i][2] and bb[i][1] <= y <= bb[i][3]:
                        # cv2.circle(frame, (int((bb[i][0]+bb[i][2])/2), int((bb[i][1]+bb[i][3])/2)),2, (255, 255, 0), -1)
                        face = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                else:       # One person Detected
                    face = frame[bb[i][1] - 40:bb[i][3] + 40, bb[i][0] - 40:bb[i][2] + 40, :]
                    # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
        # except:
        #     print("ERROR")
        #     face = None
        return frame, face


class FaceRecognizer:
    def __init__(self):
        self.dir_train_set = "./dataset/train"
        self.dir_test_set = "./dataset/test"
        self.dir_model = './models/20180402-114759.pb'
        self.file_classifier = './models/class/classifier.pkl'
        self.npy = ''
        self.num_gathered_data = 0
        self.nrof_faces = None
        self.accuracy = 0
        self.image_size = 182
        self.input_image_size = 160
        self._prepare_model()

    def make_cand_list(self):
        self.cand_ids = [1,2,3,4,5]

    def load_cand(self):
        # up = UserProfile()
        # up.load_user_list_db()
        # self.cand_id = []
        # for u in up.list_user:
        #     # print(u.read_first_name())
        #     self.cand_id.append(u.read_id())
        #     if not os.path.exists('.\\image\\'+str(0)):
        #         os.mkdir(".\\image\\" + str(0))
        #     if not os.path.exists('.\\image\\'+str(u.read_id())):
        #         os.mkdir(".\\image\\"+str(u.read_id()))
        self.cand_ids = [1,2,3,4,5]

    def _prepare_model(self):
        # To prepare registered people list

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, self.npy)

                self.minsize = 20  # minimum size of face
                self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                self.factor = 0.709  # scale factor
                self.image_size = 182
                self.input_image_size = 160

                print('Loading Modal')
                facenet.load_model(self.dir_model)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

        classifier_filename_exp = os.path.expanduser(self.file_classifier)
        with open(classifier_filename_exp, 'rb') as infile:
            (self.model, self.class_names) = pickle.load(infile,  encoding="latin1")

    # def detect(self, frame):
    #     try:
    #         # To make RGB image from grayscale or BGR Image
    #         if frame.ndim == 2:
    #             frame = facenet.to_rgb(frame)
    #         frame = frame[:, :, 0:3]
    #         # To detect face
    #         self.bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet,
    #                                                          self.threshold, self.factor)
    #         self.nrof_faces = self.bounding_boxes.shape[0]
    #         # print(detection_count, 'Detected_FaceNum: %d' % nrof_faces)
    #     except:
    #         print("ERROR")
    #         self.nrof_faces = None
    #
    #     return self.bounding_boxes
    #
    # def choose_face(self, frame, bounding_box, x=None, y=None):
    #     face = None
    #     try:
    #         if self.nrof_faces is None:
    #             return frame, None
    #         if self.nrof_faces > 0:
    #             det = bounding_box[:, 0:4]
    #             bb = np.zeros((self.nrof_faces, 4), dtype=np.int32)
    #             for i in range(self.nrof_faces):
    #                 bb[i][0] = det[i][0]
    #                 bb[i][1] = det[i][1]
    #                 bb[i][2] = det[i][2]
    #                 bb[i][3] = det[i][3]
    #                 if x is None and y is None:
    #                     face = frame[bb[i][1]-20:bb[i][3]+20, bb[i][0]-20:bb[i][2]+20, :]
    #                     cv2.rectangle(frame, (bb[i][0]-20, bb[i][1]-20), (bb[i][2]+20, bb[i][3]+20), (0, 255, 0), 2)
    #                 # else:
    #                 #     cv2.rectangle(frame, (bb[i][0]-20, bb[i][1]-20), (bb[i][2]+20, bb[i][3]+20), (0, 255, 0), 2)
    #                 #     # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
    #                 #     if bb[i][0] <= x <= bb[i][2] and bb[i][1] <= y <= bb[i][3]:
    #                 #         face = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
    #     except:
    #         face = None
    #     return frame, face

    def recognize(self, face):
        """
        To recognize input face
        :param face: ndarray, face image data
        :return: int: index of detected user, int: detected user id
        """
        try:
            self.load_cand()    # To load whole user id
            if self.nrof_faces is None or face is None:     # when input data is None
                return 0, 0
            if self.nrof_faces > 0:
                self.img_size = np.asarray(face.shape)[0:2]     # To get image size

                # To do preprocessing
                emb_array = np.zeros((1, self.embedding_size))
                cropped = facenet.flip(face, False)
                scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                scaled = cv2.resize(scaled, (self.input_image_size, self.input_image_size),
                                    interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, self.input_image_size, self.input_image_size, 3)
                feed_dict = {self.images_placeholder: scaled_reshape, self.phase_train_placeholder: False}
                emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
                predictions = self.model.predict_proba(emb_array)   # To predict confidence rate for whole labels
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                if best_class_probabilities > 0.50:     # If confidence rate is higher than 0.5
                    return best_class_indices[0], self.cand_ids[best_class_indices[0]]  #
                else:
                    return 0, 0
                    # else:
                    #     for H_i in self.cand_ids:
                    #         if self.cand_ids[best_class_indices[0]] == H_i:
                    #             detected_id = self.cand_ids[best_class_indices[0]]
                # print(">>>>> ", detected_id)
            # except:
            #     # Situation of detected unknown person
            #     print("ERROR")
            #     return  -1, detected_id
            # if detected_id is "":
            #     return  -1, detected_id
            # return self.cand_ids.index(detected_id), detected_id
        except:
                print("ERROR")
                return  -1, -1

    def estimate_model(self):
        """
        To estimate current model accuracy using gathered test set
        :return: float, accuracy
        """
        # To set the number of detected face as 1
        self.nrof_faces = 1
        # To initialize counting variables
        whole_count, num_right = 0.0, 0.0
        folders = os.listdir(self.dir_test_set) # To load test set folder
        for f_name in folders:
            for f in os.listdir(self.dir_test_set + "/" + f_name):  # To load list each user's images
                test_img = cv2.imread(self.dir_test_set + "/" + f_name + "/" + f, cv2.IMREAD_COLOR)
                id, ids = self.recognize(test_img)  # To recognize the face image
                if int(f_name) == ids:
                    num_right += 1    # To count correction
                whole_count += 1    # To count whole number of test case
        self.accuracy = round(num_right / whole_count, 3)*100   # To calculate accuracy
        return self.accuracy

    def train_model(self):
        """
        To train people recognition model using gathered training set
        :return: None
        """
        obj = Training(self.dir_train_set, self.dir_model, self.file_classifier)    # To initialize training object
        get_file = obj.main_train()     # To start training model
        print('Saved classifier model to file "%s"' % get_file)


if __name__ == '__main__':
    fd = FaceDetector()
    cap = cv2.VideoCapture("./dummy/S124_001_00000014.png")

    # while True:
    _, frame = cap.read()
    rect = fd.detect(frame)
    f, f1 = fd.choose_face(frame)

    cv2.imshow("Face", f1)
    cv2.imwrite("./dummy/crop_S124_001_00000014.png", f1)
    cv2.waitKey(0)
    cv2.destroyAllWIndows()

    # key_input = cv2.waitKey(1)
    # if key_input & 0xFF == ord('q'):

    # c = FaceRecognizer()
    # # c.train_model()
    # print(c.estimate_model())

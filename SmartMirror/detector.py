"""
Date: 2019.03.21
Programmer: MH
Description: Detector module
"""
import os
import shutil
from threading import Thread

import cv2
import pickle
from scipy import misc
import numpy as np
import tensorflow as tf
from gtts import gTTS
from time import sleep
import pyglet

from packages import facenet, detect_face

from db_connector import User, UserSession, UserProfile
from packages.preprocess import preprocesses
from packages.classifier import Training


class Detector:
    def __init__(self):
        """
        To initialize the object
        """
        self.input_video = 0
        self.modeldir = './models/20180402-114759.pb'
        self.classifier_filename = './models/class/classifier.pkl'
        self.npy = ''
        self.train_img = "./train_img"
        self.detection_count = 0
        self.clicked = False
        self.select_x = -1
        self.select_y = -1

        self.person_selected = False
        self.is_greeting = False
        self.greeting_start = 0
        self.nrof_faces = None
        self.name = ""
        self.result = []
        self.cand_id = []
        pyglet.options['debug_gl'] = False
        self._prepare_model()
        self.detected_people = []

    def load_cand(self):
        up = UserProfile()
        up.load_user_list_db()
        self.cand_id = []
        for u in up.list_user:
            # print(u.read_first_name())
            self.cand_id.append(u.read_id())
            if not os.path.exists('.\\image\\'+str(0)):
                os.mkdir(".\\image\\" + str(0))
            if not os.path.exists('.\\image\\'+str(u.read_id())):
                os.mkdir(".\\image\\"+str(u.read_id()))

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
                facenet.load_model(self.modeldir)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                classifier_filename_exp = os.path.expanduser(self.classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (self.model, self.class_names) = pickle.load(infile,  encoding="latin1")

    def detect_people(self, frame):
        def th(frame):
            try:
                # To make RGB image from grayscale or BGR Image
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                # To detect face
                self.bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet,
                                                                 self.threshold, self.factor)
                self.nrof_faces = self.bounding_boxes.shape[0]
                # print(detection_count, 'Detected_FaceNum: %d' % nrof_faces)
            except:
                print("ERROR")
                self.nrof_faces = None
        Thread(target=th, args=(frame,)).start()

    def get_nrof_faces(self):
        return self.nrof_faces

    def choose_face(self, frame, x=-1, y=-1):
        face = None
        try:
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
                    if x is None and y is None:
                        face = frame[bb[i][1]-40:bb[i][3]+40, bb[i][0]-40:bb[i][2]+40, :]
                    else:
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        if bb[i][0] <= x <= bb[i][2] and bb[i][1] <= y <= bb[i][3]:
                            face = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
        except:
            face = None
        return frame, face

    def identify_face(self, frame, input_face):
        # def th(frame, idx):
        try:
            # self.load_cand()
            detected_id = ""
            if self.nrof_faces is None or input_face is None:
                return frame, ""
            if self.nrof_faces > 0:
                self.img_size = np.asarray(frame.shape)[0:2]

                emb_array = np.zeros((1, self.embedding_size))
                cropped = facenet.flip(input_face, False)
                scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                scaled = cv2.resize(scaled, (self.input_image_size, self.input_image_size),
                                       interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, self.input_image_size, self.input_image_size, 3)
                feed_dict = {self.images_placeholder: scaled_reshape, self.phase_train_placeholder: False}
                emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
                predictions = self.model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                if best_class_probabilities > 0.50:
                    self.load_cand()
                    if best_class_indices[0] >= len(self.cand_id) or best_class_indices[0] == 0:
                        # print(">>>>> ", best_class_indices[0], len(self.cand_id))
                        return best_class_indices[0], 0
                    else:
                        for H_i in self.cand_id:
                            if self.cand_id[best_class_indices[0]] == H_i:
                                detected_id = self.cand_id[best_class_indices[0]]
                # print(">>>>> ", detected_id)
        except :
            # Situation of detected unknown person
            print("ERROR")
            return frame, ""
        if detected_id is "":
            return frame, ""
        return self.cand_id.index(detected_id), detected_id

    def voice(self, content, f_name=".\\tmp\\temp.wav"):
        def th():
            try:
                tts = gTTS(text=content, lang='en')
                tts.save(f_name)
                music = pyglet.media.load(f_name, streaming=False)
                music.play()
                sleep(music.duration)  # prevent from killing
            except:
                pass
        Thread(target=th).start()

    def train_model(self, id=20):
        def th(id):
            os.mkdir('.\\emotion_data\\train_img\\'+str(id)+"\\")
            src_files = os.listdir('.\\image\\0\\')
            for file_name in src_files:
                full_file_name = os.path.join('.\\image\\0\\', file_name)
                if (os.path.isfile(full_file_name)):
                    shutil.copy(full_file_name, '.\\emotion_data\\train_img\\'+str(id)+"\\")
            input_datadir = '.\\emotion_data\\train_img'
            output_datadir = '.\\emotion_data\\pre_img'

            obj = preprocesses(input_datadir, output_datadir)
            obj.collect_data()
            datadir = './emotion_data/pre_img'
            modeldir = './models/20180402-114759.pb'
            classifier_filename = '.models/class/classifier.pkl'
            obj = Training(datadir, modeldir, classifier_filename)
            obj.main_train()
            self.load_cand()
        Thread(target=th, args=[id,]).start()

if __name__ == '__main__':
    t = Detector()
    t.train_model()
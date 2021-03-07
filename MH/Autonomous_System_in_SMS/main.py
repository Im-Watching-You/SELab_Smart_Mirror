import cv2
from face_detections import FaceRecognizer
import threading


class Video:
    cap = None
    input_video = 0
    acq = True

    def __init__(self):
        self.cap = cv2.VideoCapture(self.input_video)
        self.face_detector = FaceRecognizer()

    def acquire(self):
        while self.acq:
            self.acq, frame = self.cap.read()
            rect = self.face_detector.detect(frame)
            _, faces = self.face_detector.choose_face(frame, rect)
            user_id, ids = self.face_detector.detect(faces)
            print(ids)
            # threading.Thread(target=self.th_detect, args=(frame,)).start()
            # print(faces)
            # if faces is not None:
            #     user_id, ids = self.face_detector.recognize(faces)
            #     print(ids)
            #     cv2.imshow("Face", faces)
            cv2.imshow("Smart Mirror System", frame)  # display image

            key_input = cv2.waitKey(1)
            if key_input & 0xFF == ord('q'):
                break
            elif key_input & 0xFF == ord('n'):
                print("N")
            elif key_input & 0xFF == ord('y'):
                print("Y")

    def th_detect(self, frame):
        rect = self.face_detector.detect(frame)
        _, faces = self.face_detector.choose_face(frame, rect)


if __name__ == '__main__':
    v = Video()
    v.acquire()

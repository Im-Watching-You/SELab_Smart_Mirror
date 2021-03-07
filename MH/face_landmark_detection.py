from imutils import face_utils
import dlib
import cv2
import numpy as np

# Vamos inicializar um detector de faces (HOG) para então
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.

import math

def compute_distance(p1, p2):
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return distance

def compute_middle(p1, p2):
    p3 = ((p2[0]+p1[0])/2, (p2[1]+p1[1])/2)
    return p3

p = "shape_predictor_81_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
img = cv2.imread("./faces/test.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

count = 0
# Get faces into webcam's image
rects = detector(gray, 0)
for (i, rect) in enumerate(rects):
    # Make the prediction and transfom it to numpy array
    shape = predictor(gray, rect)
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    # shape = face_utils.shape_to_np(shape)
    for num in range(shape.num_parts):
        x = shape.parts()[num].x
        y = shape.parts()[num].y
        cv2.putText(img, str(num), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
        # cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    # print("1. zy-zy: ", compute_distance(shape[15], shape[1]))
    # print("2. ec-ec: ", compute_distance(shape[45], shape[36]))
    # print("3. en-en: ", compute_distance(shape[42], shape[39]))
    # print("4. pu-pu: ", compute_distance(compute_middle(shape[46],shape[43]), compute_middle(shape[40], shape[37])))
    # print("5. Iris: ", compute_distance(shape[44], shape[43]))
    # print("6. al-al: ", compute_distance(shape[35], shape[31]))
    # print("7. ch-ch: ", compute_distance(shape[54], shape[48]))
    # print("8. n-sn: ", compute_distance(shape[27], shape[33]))
    # print("9. n-gn: ", compute_distance(shape[27], shape[8]))
    # print("10. sn-gn: ", compute_distance(shape[33], shape[8]))
    #
    # # Draw on our image, all the finded cordinate points (x,y)
    #
    # i = 0
    # for (x, y) in shape:
    #     # cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
    #     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    # #     i += 1
    cv2.rectangle(img, (rects[0].left(),rects[0].top()), (rects[0].right(), rects[0].bottom()), (0, 0, 255), 4)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2. destoryAllWindows()

# cap = cv2.VideoCapture('./faces/123123123.png')
#
# while True:
#     # Getting out image by webcam
#     _, image = cap.read()
#     # Converting the image to gray scale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Get faces into webcam's image
#     rects = detector(gray, 0)
#
#     # For each detected face, find the landmark.
#     for (i, rect) in enumerate(rects):
#         # Make the prediction and transfom it to numpy array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#
#         # Draw on our image, all the finded cordinate points (x,y)
#         i = 0
#         for (x, y) in shape:
#             cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
#             # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#             i+=1
#
#     # Show the image
#     cv2.imshow("Output", image)
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()
# cap.release()
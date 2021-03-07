"""
Date: 2019.07.12
Programmer: Maksym Chernozhukov
Description: Code for Aging Analytics "Wrinkle Factor"
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
import imutils
from imutils import face_utils

"""
    1. Sobel Filter
"""


def apply_sobel(img, ksize, thres=None):
    """
    Apply Sobel operator [ksizexksize] to image.
    @param  img:    input image
    @param  ksize:  Sobel kernel size
                    @pre odd integer >= 3
    @param  thres:  binary threshold, if None do not threshold
                    @pre integer >= 0 & <= 255

    @return:    image of Sobel magnitude
    """
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    Im = cv2.magnitude(Ix, Iy)
    if thres is not None:
        _, It = cv2.threshold(Im, thres, 1, cv2.THRESH_BINARY)
        return It
    else:
        return Im


"""
    2. Facial Landmarks
"""


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def draw_rectangles(gray, cour):
    print('Min x1:', cour[0])
    print('Min y1:', cour[1])
    print('Max x2:', cour[2])
    print('Max y2:', cour[3])
    print(cour)
    cv2.rectangle(gray, (cour[0], cour[1]), (cour[2], cour[3]), (0, 0, 255), 2)
    plt.subplot(111), plt.imshow(gray, cmap="gray")
    plt.title("Wrinkle Sections"), plt.xticks([]), plt.yticks([])
    plt.show()


# For Facial Parts Coordinates
def find_coordinates(gray, shape):
    k = 0
    print(shape)
    x1, x2, y1, y2 = 1000, 0, 1000, 0
    for (x, y) in shape:
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x)
        y2 = max(y2, y)
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(image, str(k), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
        k += 1
    coor = [x1, y1, x2, y2]
    coorwh = [x1, y1, x2 - x1, y2 - y1]
    draw_rectangles(gray, coor)

    return coor, coorwh


predictor_path = '../shape_predictor_68_face_landmarks.dat'
faces_folder_path = './img/man2.jpg'

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(faces_folder_path)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

shape = predictor(gray, rects[0])
shape = face_utils.shape_to_np(shape)


def draw_wrinkle_sections(gray, dots):
    print(dots)
    x1 = dots[0]
    y1 = dots[1]
    x2 = dots[2]
    y2 = dots[3]
    cv2.rectangle(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)


corner_left_eye = [shape[2][0], min(shape[37][1], shape[38][1]), shape[18][0], shape[28][1]]
draw_wrinkle_sections(gray, corner_left_eye)

corner_right_eye = [shape[25][0], min(shape[43][1], shape[44][1]), shape[14][0], shape[28][1]]
draw_wrinkle_sections(gray, corner_right_eye)

forehead = [shape[19][0], shape[19][1] - (shape[42][0] - shape[39][0]),
            shape[24][0], min(shape[19][1], shape[24][1])]
# shape[42][0]-shape[39][0] = h (dist_eyes)
draw_wrinkle_sections(gray, forehead)

cheek_left = [shape[17][0], shape[28][1], shape[49][0], shape[2][1]]
draw_wrinkle_sections(gray, cheek_left)

cheek_right = [shape[53][0], shape[28][1], shape[26][0], shape[14][1]]
draw_wrinkle_sections(gray, cheek_right)

neck = [shape[41][0] + 5, shape[8][1], shape[46][0], shape[8][1] + (shape[57][1] - shape[33][1])]
draw_wrinkle_sections(gray, neck)

plt.subplot(111), plt.imshow(gray, cmap="gray")
plt.title("Wrinkle Sections"), plt.xticks([]), plt.yticks([])
plt.show()

"""
    3. Wrinkle Feature Extraction
"""


def wrinkle_density(img, threshold):
    Wa = np.sum(img >= threshold)
    Pa = img.shape[0] * img.shape[1]
    result = Wa/Pa
    return result


def wrinkle_depth(img, threshold):
    Wa = img[img >= threshold]
    M = np.sum(Wa)
    result = M / (255*len(Wa))
    return result


def avg_skin_variance(img):
    M = np.sum(img)
    Pa = img.shape[0] * img.shape[1]
    result = M / (255*Pa)
    return result


# Function used to calculate the wrinkle features for the 5 different parts of the face:
# forehead, left eye, right eye, left cheek, right cheek
def extract_wrinkles(img, corner_left_eye, corner_right_eye, forehead, cheek_left, cheek_right, neck):
    threshold = 40

    #Apply sobel to get wrinkled image
    wrinkled = apply_sobel(img, 3)

    # Sobel Filter
    plt.subplot(121), plt.imshow(wrinkled, cmap="gray")
    plt.title("Sobel Filter"), plt.xticks([]), plt.yticks([])
    plt.show()

# Define the sections for each part of the face and calculate their wringle features
# Corner Left Eye
#   [y1:y2, x1:x2]
    print(corner_left_eye)
    window = wrinkled[corner_left_eye[1]:corner_left_eye[3], corner_left_eye[0]:corner_left_eye[2]]
    left_eye_wr = wrinkle_density(window, threshold)
    d1 = wrinkle_depth(window, threshold)
    v1 = avg_skin_variance(window)

    # Show rectangle
    plt.subplot(121), plt.imshow(window, cmap="gray")
    plt.title("Corner Left Eye"),plt.xticks([]),plt.yticks([])
    plt.show()

# Corner Right Eye
    print(corner_right_eye)
    window = wrinkled[corner_right_eye[1]:corner_right_eye[3], corner_right_eye[0]:corner_right_eye[2]]
    right_eye_wr = wrinkle_density(window, threshold)
    d2 = wrinkle_depth(window, threshold)
    v2 = avg_skin_variance(window)

    # Show rectangle
    plt.subplot(121), plt.imshow(window, cmap="gray")
    plt.title("Corner Right Eye"),plt.xticks([]),plt.yticks([])
    plt.show()

# Forehead
    print(forehead)
    window = wrinkled[forehead[1]:forehead[3], forehead[0]:forehead[2]]
    forehead_wr = wrinkle_density(window, threshold)
    d3 = wrinkle_depth(window, threshold)
    v3 = avg_skin_variance(window)

    # Show rectangle
    plt.subplot(121), plt.imshow(window, cmap="gray")
    plt.title("Forehead"),plt.xticks([]),plt.yticks([])
    plt.show()

# Cheek Left
    print(cheek_left)
    window = wrinkled[cheek_left[1]:cheek_left[3], cheek_left[0]:cheek_left[2]]
    cheek_left_wr = wrinkle_density(window, threshold)
    d4 = wrinkle_depth(window, threshold)
    v4 = avg_skin_variance(window)

    # Show rectangle
    plt.subplot(121), plt.imshow(window, cmap="gray")
    plt.title("Cheek Left"),plt.xticks([]), plt.yticks([])
    plt.show()

# Cheek Right
    print(cheek_right)
    window = wrinkled[cheek_right[1]:cheek_right[3], cheek_right[0]:cheek_right[2]]
    cheek_right_wr = wrinkle_density(window, threshold)
    d5 = wrinkle_depth(window, threshold)
    v5 = avg_skin_variance(window)

    # Show rectangle
    plt.subplot(121), plt.imshow(window, cmap="gray")
    plt.title("Cheek Right"),plt.xticks([]),plt.yticks([])
    plt.show()

# Neck
    print(neck)
    window = wrinkled[neck[1]:neck[3], neck[0]:neck[2]]
    neck_wr = wrinkle_density(window, threshold)
    d6 = wrinkle_depth(window, threshold)
    v6 = avg_skin_variance(window)

    # Show rectangle
    plt.subplot(121), plt.imshow(window, cmap="gray")
    plt.title("Neck"), plt.xticks([]), plt.yticks([])
    plt.show()

    d = [d1, d2, d3, d4, d5, d6]
    v = [v1, v2, v3, v4, v5, v6]
    result = [left_eye_wr, right_eye_wr, forehead_wr, cheek_left_wr, cheek_right_wr, neck_wr]
    return result, d, v


# Show Wrinkle Density for Each Sections
wrinkle, d, v = extract_wrinkles(gray, corner_left_eye, corner_right_eye, forehead, cheek_left, cheek_right, neck)
print('Wrinkle Density:', wrinkle)
print('Wrinkle Depth:', d)
print('Wrinkle Variance:', v)

import cv2
import numpy as np
from math import sqrt, atan, degrees
import matplotlib.pyplot as plt


def improve_quality(plate):
    """

    :param plate: detected palte numpy nd array : (h , w , c)
    :return: result : nd array : (h , w , c)
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    improve_model_path = "/your_path/EDSR_x4.pb"
    sr.readModel(improve_model_path)
    sr.setModel("edsr", 4)
    result = sr.upsample(plate)
    return result


def find_longest_line(plate_img, plate_img_gr):
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(plate_img_gr, (kernel_size, kernel_size), 0)

    low_threshold = 150
    high_threshold = 200

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(plate_img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    if lines is None:
        return None

    lls = []
    for indx, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            line_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            lls.append((indx, line_length))
    lls.sort(key=lambda x: x[1])
    linessorted = []
    for (indx, ll) in lls:
        linessorted.append(lines[indx])

    return linessorted


def find_line_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = degrees(atan(((y2 - y1) / (x2 - x1))))
    return angle


def rotate_image(plate_img_gr, angle):
    (h, w) = plate_img_gr.shape[0], plate_img_gr.shape[1]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(plate_img_gr, M, (w, h))
    return rotated


def adjust_cropping(rotated_img):
    h, w = rotated_img.shape[0], rotated_img.shape[1]
    targ_h = int(w / 4)
    crop_h = int((h - targ_h) / 2)
    cropped_rotated_img = rotated_img[crop_h:h - crop_h, :]
    return cropped_rotated_img


def adjust_plate(plate):
    """

    :param plate: car_plate numpy nd array (h , w, c)
    :return: cropped_rotated_img numpy nd array (h , w, c)
    """
    plate_img = plate.copy()
    plate_img_gr = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    linessorted = find_longest_line(plate_img, plate_img_gr)
    if linessorted is None:
        return plate_img
    rot_angle = find_line_angle(linessorted[-1])

    if rot_angle <= 5.0:
        return plate_img
    rotated_img = rotate_image(plate_img, rot_angle)
    cropped_rotated_img = adjust_cropping(rotated_img)
    cw = cropped_rotated_img.shape[1]

    return cropped_rotated_img

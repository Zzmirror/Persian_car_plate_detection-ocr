from skimage import measure
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def apply_threshold(image):
    """
    :param image: np nd array shape  : (w ,h , c)
    :return: thresh np nd array shape : (w ,h)

    convert image to grayscale , apply adaptiveThreshold
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 199, 5)
    return thresh


def label_contour(thresh):
    """
    :param thresh: np nd array shape : (w ,h )
    :return: labels, char_candidates, inverted_thresh
    """
    # invert black number to white number
    inverted_thresh = np.invert(thresh)
    labels = measure.label(inverted_thresh, background=0)
    char_candidates = np.zeros(inverted_thresh.shape, dtype="uint8")
    return labels, char_candidates, inverted_thresh


def prepare_boding_box_dataset(labels, char_candidates, inverted_thresh):
    """

    :param labels:
    :param char_candidates:
    :param inverted_thresh:
    :return: [bonding_box[-1]] single & last found bonding_box : list of array
    """
    bonding_box = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        label_mask = np.zeros(inverted_thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255
        cnts = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:
            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solidity, and height ratio for the component
            aspect_ratio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            height_ratio = boxH / float(inverted_thresh.shape[0])

            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests
            keep_aspect_ratio = aspect_ratio < 3.0
            keep_solidity = solidity > 0.09
            keep_height = height_ratio > 0.01 and height_ratio < 2.0

            # check to see if the component passes all the tests
            if keep_aspect_ratio and keep_solidity and keep_height:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)
                cv2.drawContours(char_candidates, [hull], -1, 255, -1)
                bonding_box.append(cv2.boundingRect(char_candidates))

    return [bonding_box[-1]]


def prepare_boding_box_plate(labels, char_candidates, inverted_thresh, resized_image):
    new_bondingbox = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        label_mask = np.zeros(inverted_thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255
        cnts = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:
            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solidity, and height ratio for the component
            aspect_ratio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            height_ratio = boxH / float(resized_image.shape[0])

            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests

            keepAspectRatio = aspect_ratio < 3
            keepSolidity = solidity > 0.13
            keepHeight = height_ratio > 0.3 and height_ratio < 0.9

            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)
                cv2.drawContours(char_candidates, [hull], -1, 255, -1)
                new_bondingbox.append(cv2.boundingRect(c))
    return sorted(new_bondingbox)


def crop_image(thresh, bonding_box):
    """

    :param thresh:
    :param bonding_box: list of array : last found bonging_box
    :return: crop : np nd array :cropped image
    """
    target_w = 32
    target_h = 32
    image = thresh.copy()
    for rect in bonding_box:
        # Get the coordinates from the bounding box
        x, y, w, h = rect

        # Crop the character from the mask
        # and apply bitwise_not because in our training data for pre-trained model
        # the characters are black on a white background
        crop = image[y:y + h, x:x + w]
        # crop = cv2.bitwise_not(crop)

        # Get the number of rows and columns for each cropped image
        # and calculate the padding to match the image input of pre-trained model
        rows = crop.shape[0]
        columns = crop.shape[1]
        padding_y = (target_h - rows) // 2 if rows < target_h else int(0.17 * rows)
        padding_x = (target_w - columns) // 2 if columns < target_w else int(0.45 * columns)

        # Apply padding to make the image fit for neural network model
        crop = cv2.copyMakeBorder(crop, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT, None, 255)

        #  resize image
        crop = cv2.resize(crop, (target_w, target_h))

    return crop


def display_bonding_box(image, bonding_box):
    """

    :param image: original image np nd array
    :param bonding_box: list of boding boxes
    :return:
    """
    img = image.copy()
    for rect in bonding_box:
        # Get the coordinates from the bounding box
        x, y, w, h = rect
        cv2.rectangle(img, (x - 1, y - 3), (x + w + 1, y + h + 3), (255, 0, 0), 1)
    cv2.imshow(img , 'img')

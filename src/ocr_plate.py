import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from preprocess_plate import improve_quality, adjust_plate
from preprocess_image import apply_threshold, label_contour, prepare_boding_box_plate


def predict_model(crop):
    """

    :param crop: cropped area of plate shape : (32 , 32 )
    :return:
    """
    model = load_model('saved_model_path')
    y_df = pd.read_csv('saved_y_df/y_df.csv')
    # preprocess cropped image
    crop = crop.reshape(1, 32, 32, 1)
    crop = crop / 255
    pre = np.argmax(model.predict(crop, verbose=0))
    decoded_pre = y_df[y_df['labels_nums'] == pre].class_names.values[0]
    return decoded_pre


def ocr_prediction(original_image, original_thresh, bonding_box):
    TARGET_WIDTH = 32
    TARGET_HEIGHT = 32

    image = original_image.copy()
    thresh = original_thresh.copy()
    prediction = []

    # Loop over the bounding boxes
    for rect in bonding_box:
        # Get the coordinates from the bounding box
        x, y, w, h = rect

        # Crop the character from the mask
        # and apply bitwise_not because in our training data for pre-trained model
        # the characters are black on a white background
        crop = thresh[y:y + h, x:x + w]
        # crop = cv2.bitwise_not(crop)

        # Get the number of rows and columns for each cropped image
        # and calculate the padding to match the image input of pre-trained model
        rows = crop.shape[0]
        columns = crop.shape[1]
        paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

        # Apply padding to make the image fit for neural network model
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        # Convert and resize image
        # crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

        pre = predict_model(crop)

        # idx = np.argsort(prob)[-1]
        # vehicle_plate += chars[idx]

        # Show bounding box and prediction on image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # cv2.putText(image, chars[idx], (x,y+15), 0, 0.8, (0, 0, 255), 2)
        prediction.append(pre)

    # Show final image
    cv2.imshow(image , 'car_plate')
    print("predicted plate: ", prediction)
    cv2.waitKey(0)
    return prediction, image


def ocr_plate(detected_plate):
    """

    :param detected_plate: original detected plate : numpy nd array (h , w, c)
    :return: prediction : prediction list
            image_with_bonding_box : numpy nd array (h , w, c)
    """
    plate = detected_plate.copy()
    improved_plate = improve_quality(plate)
    adjusted_plate = adjust_plate(improved_plate)
    resized_plate = cv2.resize(adjusted_plate, (300, 50))
    thresh = apply_threshold(resized_plate)
    labels, char_candidates, inverted_thresh = label_contour(thresh)
    bonding_box = prepare_boding_box_plate(labels, char_candidates, inverted_thresh, resized_plate)
    prediction, image_with_bonding_box = ocr_prediction(resized_plate, thresh, bonding_box)
    return prediction, image_with_bonding_box

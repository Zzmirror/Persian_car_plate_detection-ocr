import cv2
import pandas as pd

from preprocess_image import apply_threshold, label_contour, prepare_boding_box_dataset, crop_image
import numpy as np


def prepare_cropped_image(image_path):
    """

    :param image_path: str
    :return: crop : cropped_image np nd array
    """
    image = cv2.imread(image_path)
    thresh = apply_threshold(image)
    labels, char_candidates, inverted_thresh = label_contour(thresh)
    bonding_box = prepare_boding_box_dataset(labels, char_candidates, inverted_thresh)
    # display_bonding_box(image , bonding_box )
    crop = crop_image(thresh, bonding_box)
    return crop


def prepare_cropped_dataset():
    cropped_image_path_list = []
    cropped_y = []
    count = 0
    files_path = 'your_files_path'
    image_info_df = pd.read_csv(files_path + '/your_dataset_info_csv_path')
    target_h, target_x = 32, 32
    number_of_images = image_info_df.shape[0]

    cropped_X = np.empty([number_of_images, target_h, target_x])

    for image_path in image_info_df.image_path:
        cropped_image = prepare_cropped_image(image_path)
        cropped_X[count, :, :] = cropped_image
        y = image_info_df[image_info_df['image_path'] == image_path].label[count]
        cropped_y.append(y)
        cropped_image_path_list.append(image_path)
        count += 1

    cropped_data_pd = pd.DataFrame(
        {'croped_image_path_list': cropped_image_path_list,
         'croped_y': cropped_y,
         })

    classes_name = list(set(cropped_y))
    labels_nums = [i for i in range(0, len(classes_name))]
    y_df = pd.DataFrame(
        {'class_names': classes_name,
         'labels_nums': labels_nums,
         })

    np.save(files_path + '/new_croped_X.npy', cropped_X)
    np.save(files_path + '/new_croped_y.npy', cropped_y)
    cropped_data_pd.to_csv(files_path + '/croped_data_pd.csv')
    y_df.to_csv(files_path + '/y_df.csv')

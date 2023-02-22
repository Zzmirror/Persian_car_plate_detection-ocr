from copy import deepcopy
import os
from utils.torch_utils import select_device, TracedModel
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
import torch
from utils.datasets import letterbox
import numpy as np
import cv2
import random
from ocr_plate import ocr_plate

from utils.plots import plot_one_box, plot_one_box_PIL

input_path = 'D:/projects/car_plate_ditection_yolov7/input'
image_path = os.path.join(input_path, '5.jpg')

save_path = 'D:/projects/car_plate_ditection_yolov7/saved_result'
weights = 'weights/best.pt'
device_id = 'cpu'
image_size = 640
trace = True

# Initialize
device = select_device(device_id)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(image_size, s=stride)

if trace:
    model = TracedModel(model, device, image_size)

if half:
    model.half()  # to FP16
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


def detect_plate(source_image):
    image_size = 640
    strides = 32
    img = letterbox(source_image, image_size, stride=stride)[0]

    #     convert

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        # Inference
        pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

    plate_detections = []
    det_confidences = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()

            # Return results
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
                det_confidences.append(conf.item())

    return plate_detections, det_confidences


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=2.0, threshold=0):
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def crop(image, coord):
    cropped_image = image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
    return cropped_image


def get_plates_from_image(input):
    if input is None:
        print("this is null :", input)
        return None
    plate_detections, def_confidences = detect_plate(input)
    plate_texts = []
    ocr_confiences = []
    detected_image = deepcopy(input)
    for coords in plate_detections:
        plate_region = crop(input, coords)
        detected_image, prediction = ocr_plate(plate_region)
        name = ''.join([str(item + '-') for item in prediction]) + '.png'
        cv2.imwrite(os.path.join(save_path, name), detected_image)

    return detected_image


plate_image = cv.imread(image_path)
detected_plate_image = get_plates_from_image(plate_image)
cv.waitKey(0)
cv.destroyAllWindows

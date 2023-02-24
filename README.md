# Persian_car_plate_detection&ocr

In this project we will detec persian car plates and then read them.
This project consists of three main stages:
**1. Car plate detection**
**2. Character Segmentation**
**3. Training a new OCR model on persian dataset**

## Car plate detection :
For detecting car plates I used yolov7 pre_trained model , and fine tuned it on two datasets .
One include persian car plates :
[Persian car plates](https://www.kaggle.com/datasets/skhalili/iraniancarnumberplate).
The second one contain similar car plates to persian ones. :
[Non persian car plates](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).
I got a huge help from this [video](https://www.youtube.com/watch?v=bgAUHS1Adzo) , thanks to @mrymsadeghi

## Character Segmentation :
For this part befor anything , we should improve detected car plate quality and adjust it .

Adjusted plate :

![Adjusted plate](https://github.com/Zzmirror/Persian_car_plate_detection-ocr/blob/main/files/adjusted_plate.png)


For this purpose I used [x4_EDSR pre_trained model](https://github.com/Saafke/EDSR_Tensorflow/tree/master/models)
Quality improved image : Second image is quality improved :

![Quality improved image](https://github.com/Zzmirror/Persian_car_plate_detection-ocr/blob/main/files/quality.png)

After all I applied some thresholding with opencv , and I could get Character bonding boxes . you can see more detailes in src/preprocess_image.py and src/preprocess_plate.py .
Segmented Characters :

![Segmented Characters](https://github.com/Zzmirror/Persian_car_plate_detection-ocr/blob/main/files/character_segmentation.png)

## Training a new OCR model on persian dataset :
I used [persian-alpha](https://www.kaggle.com/datasets/mehdisahraei/persian-alpha) dataset .
And applied a cnn on it .You can see cnn architecture in src/Model/ocr_model.py .

**You should pay attentaion that reshaping dataset images to small dimention like 32 , 32 , will decrease image quality and you will not get a good result .To overcome this problem , again I used Character Segmentation on dataset images and I cropped the area I wanted and trained model on new ones.**

This model got **0.98 acc** on validation data and has a good result on test set too just with **50 epoches**.

But on plates images it doesn't have ideal result for some classes.
In future by gaining more data especially car plate images , model performance can be increased.

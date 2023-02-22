import numpy as np
from ocr_model import LeNet5v2
from preprocess_input import load_dataset, train_test_val_split, preprocess
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.models import save_model


def build_model():
    # Build model
    print("[INFO] Compiling model...")
    shape = (32, 32, 1)
    model_LeNet5v2 = LeNet5v2(input_shape=shape, classes=43)
    model_LeNet5v2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model_LeNet5v2.summary())

    return model_LeNet5v2


def train_model():
    model = build_model()
    X, y = load_dataset()
    train_size = 0.8
    test_size = 0.2
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_val_split(train_size, test_size, X, y)
    pre_X_train, pre_y_train = preprocess(X_train, y_train)
    pre_X_val, pre_y_val = preprocess(X_valid, y_valid)
    pre_X_test, pre_y_test = preprocess(X_test, y_test)
    history_model = model.fit(pre_X_train, pre_y_train, epochs=50, batch_size=64,
                              validation_data=(pre_X_val, pre_y_val))

    LeNet5v2_acc = history_model.history['accuracy']
    LeNet5v2_val_acc = history_model.history['val_accuracy']
    LeNet5v2_loss = history_model.history['loss']
    LeNet5v2_val_loss = history_model.history['val_loss']

    plt.figure(figsize=(6, 4))

    fig, ax = plt.subplots(1, 2, sharey='row')
    # plt.subplot(1 ,2,1)
    ax[0].plot(LeNet5v2_acc, color='blue', label="Training Accuracy")
    ax[0].plot(LeNet5v2_val_acc, color='red', label="Val Accuracy")
    ax[0].set_title('Train & validation acc')

    # plt.subplot(1 ,2,2)
    ax[1].plot(LeNet5v2_loss, color='blue', label="Training loss")
    ax[1].plot(LeNet5v2_val_loss, color='red', label="Val loss")
    ax[1].set_title('Train & validation loss')

    plt.show()

    y_test_index = np.argmax(pre_y_test, axis=1)  # Convert one-hot to index

    y_pred_crop = np.zeros(pre_y_test.shape[0])
    for i in range(pre_y_test.shape[0]):
        x = np.expand_dims(pre_X_test[i], axis=0)
        predict = np.argmax(model.predict(x, verbose=False))
        y_pred_crop[i] = predict

    print(classification_report(y_test_index, y_pred_crop))
    save_model(model, "your_path/model_LeNet5v2_ocr_crop.h5")

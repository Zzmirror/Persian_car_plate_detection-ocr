from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


def LeNet5v2(input_shape=(32, 32, 1), classes=43):
    """
      Implementation of a modified LeNet-5.
      Only those layers with learnable parameters are counted in the layer numbering.

      Arguments:
      input_shape -- shape of the images of the dataset
      classes -- integer, number of classes

      Returns:
      model -- a Model() instance in Keras

    """
    model = Sequential(
        [
            # layer

            Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', input_shape=input_shape,
                   kernel_regularizer=l2(0.0005), name='convolution_1'),

            #  layer 2
            Conv2D(filters=32, kernel_size=5, strides=1, name='convolution_2', use_bias=False),
            BatchNormalization(name='batchnorm_1'),
            Activation("relu"),
            MaxPooling2D(pool_size=2, strides=2, name='max_pool_1'),
            Dropout(0.25, name='dropout_1'),

            #  layer 3
            Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', input_shape=(32, 32, 1),
                   kernel_regularizer=l2(0.0005), name='convolution_3'),

            #  layer 4
            Conv2D(filters=64, kernel_size=3, strides=1, name='convolution_4', use_bias=False),
            BatchNormalization(name='batchnorm_2'),
            Activation("relu"),
            MaxPooling2D(pool_size=2, strides=2, name='max_pool_2'),
            Dropout(0.25, name='dropout_2'),
            Flatten(name='flatten'),

            # layer 6
            Dense(units=256, name='fully_connected_1', use_bias=False),
            BatchNormalization(name='batchnorm_3'),
            Activation("relu"),

            # Layer 7
            Dense(units=128, name='fully_connected_2', use_bias=False),
            BatchNormalization(name='batchnorm_4'),
            Activation("relu"),

            # Layer 8
            Dense(units=84, name='fully_connected_3', use_bias=False),
            BatchNormalization(name='batchnorm_5'),
            Activation("relu"),
            Dropout(0.25, name='dropout_3'),

            # Output
            Dense(units=classes, activation='softmax', name='output')

        ]
    )

    model._name = 'LeNet5v2'

    return model

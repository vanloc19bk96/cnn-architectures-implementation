import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ZeroPadding2D, MaxPool2D, Flatten, Dense, Dropout    
from tensorflow.keras.models import Model

image_size = 227
chanel = 3

def alexnet12(input_x):
    # first layer
    x = Conv2D(filters=96, kernel_size=(11, 11), strides=4, activation="relu", padding="valid")(input_x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding="valid")(x)
    x = BatchNormalization()(x)

    # second layer
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation="relu", padding="same")(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding="valid")(x)
    x = BatchNormalization()(x)

    # third layer
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation="relu", padding="same")(x)
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation="relu", padding="same")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation="relu", padding="same")(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding="valid")(x)

    x = Flatten()(x)
    # fc
    x = Dense(units=4096, activation="relu", use_bias=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(units=4096, activation="relu", use_bias=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(units=1000, activation="softmax")(x)
    return x

input_x = Input(shape=(image_size, image_size, chanel))
alexnet = alexnet12(input_x)

model = Model(inputs=input_x, outputs=alexnet)
model.summary()
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Cropping2D, Conv2DTranspose, MaxPool2D, Concatenate, UpSampling2D
import numpy as np
from tensorflow.keras.models import Model


def down_block(input_x, filters, kernel_size=(3, 3)):
    c = Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding='same')(input_x)
    c = Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding='same')(c)
    m = MaxPool2D(pool_size=(2, 2), strides=2)(c)

    return c, m

def bottleneck(input_x, filters=1024, kernel_size=(3, 3)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding='same')(input_x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding='same')(x)

    return x

def up_block(input_x, skip, filters, kernel_size=(3, 3)):
    x = UpSampling2D((2, 2))(input_x)
    x = Concatenate()([x, skip])
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding='same')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding='same')(x)
    return x

def build_unet(image_size, chanel, num_classes=2):
    f = [64, 128, 256, 512, 1024]

    input_x = Input(shape=(image_size, image_size, chanel))
    
    c1, m1 = down_block(input_x, f[0])
    c2, m2 = down_block(m1, f[1])
    c3, m3 = down_block(m2, f[2])
    c4, m4 = down_block(m3, f[3])

    bn = bottleneck(m4, f[4])

    x = up_block(bn, c4, f[3])  
    x = up_block(x, c3, f[2])   
    x = up_block(x, c2, f[1])   
    x = up_block(x, c1, f[0])   

    outputs = Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(x)
    
    model = Model(input_x, outputs)

    return model

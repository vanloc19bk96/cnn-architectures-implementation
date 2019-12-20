import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Lambda, Dense, ZeroPadding2D, Reshape, Permute, multiply, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Flatten
from tensorflow.keras.backend import relu
import tensorflow.keras.backend as backend
import os
import numpy as np
import tensorflow.keras.backend as K

image_size = 224
chanels = 3
batch_size = 32
FILTERS = {0: [128, 128, 256], 1: [256, 256, 512], 2: [512, 512, 1024], 3: [1024, 1024, 2048]} 

os.environ["PATH"] += os.pathsep + r'C:\Users\Admin\Downloads\graphviz-2.38\release\bin'

def slice_tensor(x, start, stop, axis):
    if axis == 3:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))


def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation='linear',
                padding='SAME',
                **kwargs):

    slice_axis = 3 

    def layer(input_tensor):
        inp_ch = int(np.shape(input_tensor)[-1] // groups)
        out_ch = int(filters // groups) 
        blocks = []
        for c in range(groups):
            slice_arguments = {
                'start': c * inp_ch,
                'stop': (c + 1) * inp_ch,
                'axis': slice_axis,
            }
            x = Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = Conv2D(out_ch,
                              kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_initializer,
                              use_bias=use_bias,
                              activation=activation,
                              padding=padding)(x)
            blocks.append(x)

        x = Concatenate(axis=slice_axis)(blocks)
        return x

    return layer

class ResNeXt50(object):
    def __init__(self, input_x):
        self.model = self.build_resneXt50(input_x)

    def first_layer(self, input_x):
        x = BatchNormalization()(input_x)
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPool2D(pool_size=(3, 3), strides=2, padding='VALID')(x)
        return x

    def conv_block(self, input_x, stage, block, strides):
        x = Conv2D(filters=FILTERS[stage][0], kernel_size=(1, 1), strides=strides, padding="SAME")(input_x)
        x = BatchNormalization()(x)
        x = relu(x)

        x = GroupConv2D(FILTERS[stage][1], kernel_size=(3, 3), strides=1)(x)
        
        x = Conv2D(filters=FILTERS[stage][2], kernel_size=(1, 1), padding="SAME")(x)
        x = BatchNormalization()(x)
        
        shortcut = input_x
        if block == 0:
            shortcut = Conv2D(filters=FILTERS[stage][2], kernel_size=(1, 1), strides=strides)(input_x)
            shortcut = BatchNormalization()(shortcut)

        x = relu(x + shortcut)

        return x

    def build_resneXt50(self, input_x):
        x = self.first_layer(input_x)
        blocks = (3, 4, 6, 3)
        for stage, block in enumerate(blocks):
            for i in range(block):
                if stage == 0 and i == 0:
                    x = self.conv_block(input_x=x, stage=stage, block=i, strides=1)
                elif i == 0:
                    x = self.conv_block(input_x=x, stage=stage, block=i, strides=2)
                else:
                    x = self.conv_block(input_x=x, stage=stage, block=i, strides=1)

        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(units=1000, use_bias=False, activation="softmax")(x)
        
        return x

    def plot_model(self, model):
        tf.keras.utils.plot_model(
            model,
            to_file='model3.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False
        )

input_x = Input(shape=(image_size, image_size, chanels))
resNeXt = ResNeXt50(input_x)
model = Model(inputs=input_x, outputs=resNeXt.model)
model.summary()
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten

image_size = 28
chanel = 1
num_classes = 10
batch_size = 32

def lenet98(input_x):
	# first layer
	x = Conv2D(filters=6, kernel_size=(5, 5), strides=1)(input_x)
	x = MaxPool2D(pool_size=(2, 2), strides=2)(x)

	# second layer
	x = Conv2D(filters=16, kernel_size=(5, 5), strides=1)(x)
	x = MaxPool2D(pool_size=(2, 2), strides=2)(x)

	x = Flatten()(x)

	# fc
	x = Dense(units=400, use_bias=False)(x)
	x = Dense(units=120, use_bias=False)(x)
	x = Dense(units=84, use_bias=False)(x)
	x = Dense(units=num_classes, use_bias=False, activation="softmax")(x)

	return x

input_x = Input(shape=(image_size, image_size, 1))
lenet98 = lenet98(input_x)
model = Model(inputs=input_x, outputs=lenet98)
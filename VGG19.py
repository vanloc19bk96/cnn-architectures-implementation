import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ZeroPadding2D, MaxPool2D, Flatten, Dense, Dropout	
from tensorflow.keras.models import Model

image_size = 224
chanel = 3
batch_size = 32

def vgg_block(input_x, repetition, filters):
	x = input_x
	for i in range(repetition):
		x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)	
	x = MaxPool2D(pool_size=(2, 2), strides=2)(x)

	return x

def VGG19(input_x):
	x = vgg_block(input_x, 2, 64)
	x = vgg_block(x, 2, 128)
	x = vgg_block(x, 4, 256)
	x = vgg_block(x, 4, 512)
	x = vgg_block(x, 4, 512)

	x = Flatten()(x)
	x = Dense(units=4096, activation="relu")(x)
	x = Dense(units=4096, activation="relu")(x)

	x = Dense(units=1000, activation="softmax")(x)
	return x

input_x = Input(shape=(image_size, image_size, chanel))
vgg = VGG19(input_x)
model = Model(inputs=input_x, outputs=vgg)
model.summary()

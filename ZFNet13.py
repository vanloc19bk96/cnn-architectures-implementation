import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout	
from tensorflow.keras.models import Model

image_size = 224
chanel = 3

def zfnet13(input_x):
	# first layer
	x = Conv2D(filters=96, kernel_size=(7, 7), strides=2, use_bias=False, activation="relu", padding="valid")(input_x)
	x = MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME")(x)
	x = BatchNormalization()(x)

	# second layer
	x = Conv2D(filters=256, kernel_size=(5, 5), strides=2, use_bias=False, activation="relu", padding="valid")(x)
	x = MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME")(x)
	x = BatchNormalization()(x)
	
	# third layer
	x = Conv2D(filters=384, kernel_size=(3, 3), strides=1, use_bias=False, activation="relu", padding="SAME")(x)
	x = Conv2D(filters=384, kernel_size=(3, 3), strides=1, use_bias=False, activation="relu", padding="SAME")(x)
	x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, use_bias=False, activation="relu", padding="SAME")(x)
	x = MaxPool2D(pool_size=(3, 3), strides=2)(x)
	x = Flatten()(x)
	print(x)
	# fc
	x = Dense(units=4096, activation="relu", use_bias=False)(x)
	x = Dropout(0.5)(x)
	x = Dense(units=4096, activation="relu", use_bias=False)(x)
	x = Dropout(0.5)(x)
	x = Dense(units=1000, activation="softmax")(x)
	return x

input_x = Input(shape=(image_size, image_size, chanel))
alexnet = zfnet13(input_x)

model = Model(inputs=input_x, outputs=alexnet)
model.summary()
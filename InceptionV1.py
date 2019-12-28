import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, MaxPool2D, concatenate, Dropout, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def inception_module(input_x, filter1x1, filter3x3, filter5x5, filter3x3_reduce, filter5x5_reduce, filters_pool_proj):
	conv1x1 = Conv2D(filters=filter1x1, kernel_size=(1, 1), padding="same", activation="relu")(input_x)

	conv3x3 = Conv2D(filters=filter3x3_reduce, kernel_size=(1, 1), padding="same", activation="relu")(input_x)
	conv3x3 = Conv2D(filters=filter3x3, kernel_size=(3, 3), padding="same",activation="relu")(conv3x3)

	conv5x5 = Conv2D(filters=filter5x5_reduce, kernel_size=(1, 1), padding="same", activation="relu")(input_x)
	conv5x5 = Conv2D(filters=filter5x5, kernel_size=(5, 5), padding="same", activation="relu")(conv5x5)

	pool_proj  = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input_x)
	pool_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), padding="same", activation="relu")(pool_proj)

	x = tf.concat([conv1x1, conv3x3, conv5x5, pool_proj], axis=3)

	return x

def auxiliary_classifier(input_x, scope_name):
	auxiliary = AveragePooling2D(pool_size=(5, 5), strides=3)(input_x)
	auxiliary = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(auxiliary)
	auxiliary = Flatten()(auxiliary)
	auxiliary = Dense(units=1024)(auxiliary)
	auxiliary = Dropout(0.4)(auxiliary)
	auxiliary = Dense(units=1000, activation="softmax", name=scope_name)(auxiliary)
	return auxiliary

def build_inceptionv1(input_x):
	x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", activation="relu")(input_x)
	x = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)
	x = tf.nn.local_response_normalization(x)
	x = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding="same", activation="relu")(x)
	x = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(x)
	x = tf.nn.local_response_normalization(x)
	x = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)

	# 3a
	x = inception_module(x, filter1x1=64, filter3x3=128, filter5x5=32, filter3x3_reduce=96, filter5x5_reduce=16, filters_pool_proj=32)

	#3b
	x = inception_module(x, filter1x1=128, filter3x3=192, filter5x5=96, filter3x3_reduce=128, filter5x5_reduce=32, filters_pool_proj=64)

	x = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)
	
	# 4a
	x = inception_module(x, filter1x1=192, filter3x3=208, filter5x5=48, filter3x3_reduce=96, filter5x5_reduce=16, filters_pool_proj=64)
	
	# intermediate layer 1 => auxiliary classifier
	auxiliary1 = auxiliary_classifier(x, "auxilliary_output_1")

	# 4b
	x = inception_module(x, filter1x1=160, filter3x3=224, filter5x5=64, filter3x3_reduce=112, filter5x5_reduce=24, filters_pool_proj=64)

	# 4c
	x = inception_module(x, filter1x1=128, filter3x3=256, filter5x5=64, filter3x3_reduce=128, filter5x5_reduce=24, filters_pool_proj=64)
	
	# 4d
	x = inception_module(x, filter1x1=112, filter3x3=288, filter5x5=64, filter3x3_reduce=144, filter5x5_reduce=32, filters_pool_proj=64)

	auxiliary2 = auxiliary_classifier(x, "auxilliary_output_2")

	# 4e
	x = inception_module(x, filter1x1=256, filter3x3=320, filter5x5=128, filter3x3_reduce=160, filter5x5_reduce=32, filters_pool_proj=128)
	
	x = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)

	# 5a
	x = inception_module(x, filter1x1=256, filter3x3=320, filter5x5=128, filter3x3_reduce=160, filter5x5_reduce=32, filters_pool_proj=128)

	# 5b
	x = inception_module(x, filter1x1=384, filter3x3=384, filter5x5=128, filter3x3_reduce=192, filter5x5_reduce=48, filters_pool_proj=128)

	x = AveragePooling2D(pool_size=(5, 5), strides=1)(x)
	x = Dropout(0.4)(x)
	x = Dense(units=1000, activation="softmax")(x)

	model = Model(inputs=input_x, outputs=[x, auxiliary1, auxiliary2], name='inception_v1')
	return model

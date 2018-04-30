import numpy as np
import sys
import os
from keras.utils import np_utils
from keras.models import Sequential,Model,load_model
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import  Adam,Nadam,Adamax
from keras.layers import *
from keras.callbacks import ModelCheckpoint,EarlyStopping
import h5py
from keras.regularizers import l2
from keras.utils import np_utils

weight_decay = 0
classes = 7
size = 512
batch = 10

def VGG_FCN_model(mode=32):
	path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	img_input = Input(shape=(size,size,3))

	b1_c1 = Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv1')(img_input)
	b1_c2 = Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv2')(b1_c1)
	b1_p = MaxPooling2D((2,2),strides=(2,2),name='block1_pool')(b1_c2)

	b2_c1 = Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv1')(b1_p)
	b2_c2 = Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv2')(b2_c1)
	b2_p = MaxPooling2D((2,2),strides=(2,2),name='block2_pool')(b2_c2)

	b3_c1 = Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv1')(b2_p)
	b3_c2 = Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv2')(b3_c1)
	b3_c3 = Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv3')(b3_c2)
	b3_p = MaxPooling2D((2,2),strides=(2,2),name='block3_pool')(b3_c3)

	b4_c1 = Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv1')(b3_p)
	b4_c2 = Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv2')(b4_c1)
	b4_c3 = Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv3')(b4_c2)
	b4_p = MaxPooling2D((2,2),strides=(2,2),name='block4_pool')(b4_c3)

	b5_c1 = Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv1')(b4_p)
	b5_c2 = Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv2')(b5_c1)
	b5_c3 = Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv3')(b5_c2)
	b5_p = MaxPooling2D((2,2),strides=(2,2),name='block5_pool')(b5_c3)

	fc_1 = Conv2D(4096, (2, 2),activation='relu', padding='same',  name='fc_1')(b5_p)
	#x = Dropout(0.4)(x)
	fc_2 = Conv2D(4096, (1, 1),activation='relu', padding='same', name='fc_2')(fc_1)
	#x = Dropout(0.4)(x)

	fc_3 = Conv2D(classes, (1, 1),  padding='same',activation='relu', strides=(1, 1))(fc_2)
	
	if (mode == '32'):
		x = Conv2DTranspose(filters=classes,kernel_size=(64,64),strides=(32,32),use_bias=False, padding='same')(fc_3)#(img_input)

	elif (mode == '16'):
		up_1 = Conv2DTranspose(filters=classes,kernel_size=(4,4),strides=(2,2),use_bias=False, padding='same')(fc_3)#(img_input)
		b4_fc =  Conv2D(classes, (1, 1),  padding='same',activation='relu', strides=(1, 1))(b4_p)
		x = Add()([up_1,b4_fc])
		x = Conv2DTranspose(filters=classes,kernel_size=(32,32),strides=(16,16),use_bias=False, padding='same')(x)#(img_input)

	elif (mode == '8'):
		up_1 = Conv2DTranspose(filters=classes,kernel_size=(4,4),strides=(2,2),use_bias=False, padding='same')(fc_3)#(img_input)

		b4_fc =  Conv2D(classes, (1, 1),  padding='same',activation='relu', strides=(1, 1))(b4_p)
		ad_1 = Add()([up_1,b4_fc])
		up_2 = Conv2DTranspose(filters=classes,kernel_size=(4,4),strides=(2,2),use_bias=False, padding='same')(ad_1)#(img_input)

		b3_fc = Conv2D(classes, (1, 1),  padding='same',activation='relu', strides=(1, 1))(b3_p)
		x = Add()([up_2,b3_fc])
		x = Conv2DTranspose(filters=classes,kernel_size=(16,16),strides=(8,8),use_bias=False, padding='same')(x)

	# U-Net

	elif (mode == 'U'):
		up_1 = Conv2DTranspose(filters=512,kernel_size=(4,4),strides=(2,2),use_bias=False, padding='same')(fc_2)
		u1_c0 = Concatenate(axis=3)([b5_c3,up_1])
		u1_c1 = Conv2D(512,(3,3),activation='relu',padding='same')(u1_c0)
		u1_c2 = Conv2D(512,(3,3),activation='relu',padding='same')(u1_c1)

		up_2 = UpSampling2D(size=(2,2))(u1_c2)
		u2_c0 = Concatenate(axis=3)([b4_c3,up_2])
		u2_c1 = Conv2D(512,(3,3),activation='relu',padding='same')(u2_c0)
		u2_c2 = Conv2D(512,(3,3),activation='relu',padding='same')(u2_c1)

		up_3 = UpSampling2D(size=(2,2))(u2_c2)
		u3_c0 = Concatenate(axis=3)([b3_c3,up_3])
		u3_c1 = Conv2D(256,(3,3),activation='relu',padding='same')(u3_c0)
		u3_c2 = Conv2D(256,(3,3),activation='relu',padding='same')(u3_c1)

		up_4 = UpSampling2D(size=(2,2))(u3_c2)
		u4_c0 = Concatenate(axis=3)([b2_c2,up_4])
		u4_c1 = Conv2D(128,(3,3),activation='relu',padding='same')(u4_c0)
		u4_c2 = Conv2D(128,(3,3),activation='relu',padding='same')(u4_c1)

		up_5 = UpSampling2D(size=(2,2))(u4_c2)
		u5_c0 = Concatenate(axis=3)([b1_c2,up_5])
		u5_c1 = Conv2D(64,(3,3),activation='relu',padding='same')(u5_c0)
		u5_c2 = Conv2D(64,(3,3),activation='relu',padding='same')(u5_c1)
		x = Conv2D(classes,(3,3),activation='relu',padding='same')(u5_c2)

	#x = UpSampling2D(size=(32, 32))(x)
	x = Reshape((size*size,7))(x)
	x = Activation('softmax')(x)
	model = Model(img_input,x)
	model.load_weights(path,by_name=True)
	return model

if __name__ == '__main__':

	filepath = 'data/' + sys.argv[1]
	mode = sys.argv[2]
	model = VGG_FCN_model(mode)

	print ('loading...')

	train = np.load('data/train_'+str(size)+'.npy').astype(np.float32)
	y_train = np.load('data/y_train_'+str(size)+'.npy').astype(np.uint8)
	valid = np.load('data/valid_'+str(size)+'.npy').astype(np.float32)
	y_valid = np.load('data/y_valid_'+str(size)+'.npy').astype(np.uint8)

	train /= 255
	valid /= 255

	print ('done~')

	model.compile(loss='categorical_crossentropy',optimizer='Adamax',metrics=['accuracy'])
	model.summary()
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
	early_stopping = EarlyStopping(monitor='val_acc', patience=10, mode='max')
	callbacks_list = [checkpoint]

	model.fit(train,y_train,batch_size=batch,epochs=1,validation_data=(valid,y_valid),callbacks=[checkpoint,early_stopping])

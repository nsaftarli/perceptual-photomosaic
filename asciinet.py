import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import *
from keras.models import Sequential, Model
from keras import Input
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import imgdata


(x_train, y_train) = imgdata.load_data(batch_size=10000)

x_train = x_train.astype('float32')
x_train /= 255
# x_train -= np.mean(x_train,axis=3, keepdims=True)




samples = 202533
text_rows = 224
text_cols = 224
dims = 16
split = 1000

x_test = x_train[:split]
x_val = x_train[split:(2 * split)]
x_train = x_train[(2 * split):]

y_test = y_train[:split]
y_val = y_train[split:(2 * split)]
y_train = y_train[(2 * split):]

#Median Frequency Balancing:
class_freq = imgdata.get_class_weights(size=8000, targets=y_train)
print("CLASS FREQUENCY: " + str(class_freq))
median_freq = np.median(class_freq)
print("MEDIAN FREQUENCY: " + str(median_freq))
balanced_freq = median_freq/class_freq
print("BALANCED FREQUENCIES: " + str(balanced_freq))



input_tensor = Input(shape=(224,224,3))

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(text_rows,text_cols,3))

# Freeze VGG layers
for layer in vgg.layers:
	layer.trainable = False

VGG16(weights='imagenet', include_top=False, input_shape=(text_rows,text_cols,3)).summary()

l1_1 = Model.get_layer(vgg, 'block1_conv1')
l1_2 = Model.get_layer(vgg, 'block1_conv2')
l1_p = Model.get_layer(vgg, 'block1_pool')

l2_1 = Model.get_layer(vgg, 'block2_conv1')
l2_2 = Model.get_layer(vgg, 'block2_conv2')
l2_p = Model.get_layer(vgg, 'block2_pool')

l3_1 = Model.get_layer(vgg, 'block3_conv1')
l3_2 = Model.get_layer(vgg, 'block3_conv2')
l3_3 = Model.get_layer(vgg, 'block3_conv3')
l3_p = Model.get_layer(vgg, 'block3_pool')

l4_1 = Model.get_layer(vgg, 'block4_conv1')
l4_2 = Model.get_layer(vgg, 'block4_conv2')
l4_3 = Model.get_layer(vgg, 'block4_conv3')
l4_p = Model.get_layer(vgg, 'block4_pool')

l5_1 = Model.get_layer(vgg, 'block5_conv1')
l5_2 = Model.get_layer(vgg, 'block5_conv2')
l5_3 = Model.get_layer(vgg, 'block5_conv3')
l5_p = Model.get_layer(vgg, 'block5_pool')


#Encoder: Basically re-building VGG layer by layer, because Keras's concat only takes tensors, not layers
x = l1_1(input_tensor)
o1 = l1_2(x)
x = l1_p(o1)
x = l2_1(x)
o2 = l2_2(x)
x = l2_p(o2)
x = l3_1(x)
x = l3_2(x)
o3 = l3_3(x)
x = l3_p(o3)
x = l4_1(x)
x = l4_2(x)
o4 = l4_3(x)
x = l4_p(o4)
x = l5_1(x)
x = l5_2(x)
o5 = l5_3(x)
x = l5_p(o5)

# x = vgg(input_tensor)

# def hed_net_models():
# 	up1 = UpSampling2D()(x)
# 	conv1 = Conv2D(512,3,activation='relu',padding='same')(up1)
# 	soft = Conv2D(16,1,activation='softmax',padding='valid')(conv1)

# 	return soft 
#Decoder layers: VGG architecture in reverse with skip connections and dropout layers

# #Block 1
up1 = UpSampling2D()(x)

conv1 = Conv2D(512, 3, activation='relu', padding='same')(up1)
conv1 = Conv2D(512, 3, activation='relu', padding='same')(conv1)
conv1 = Conv2D(512, 3, activation='relu', padding='same')(conv1)
conv1 = add([conv1,o5])
batch1 = BatchNormalization()(conv1)


# #Block 2
up2 = UpSampling2D()(batch1)

conv2 = Conv2D(512, 3, activation='relu', padding='same')(up2)
conv2 = Conv2D(512, 3, activation='relu', padding='same')(conv2)
conv2 = Conv2D(512, 3, activation='relu', padding='same')(conv2)
conv2 = add([conv2,o4])
batch2 = BatchNormalization()(conv2)


# #Block 3
up3 = UpSampling2D()(batch2)

conv3 = Conv2D(256, 3, activation='relu', padding='same')(up3)
conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
conv3 = add([conv3,o3])
batch3 = BatchNormalization()(conv3)

drop3 = Dropout(0.5)(batch3)

# #Block 4
up4 = UpSampling2D()(drop3)

conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
conv4 = add([conv4,o2])
batch4 = BatchNormalization()(conv4)

drop4 = Dropout(0.5)(batch4)

# #Block 5
up5 = UpSampling2D()(drop4)

conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
conv5 = add([conv5,o1])
batch5 = BatchNormalization()(conv5)

drop5 = Dropout(0.5)(batch5)

# x = UpSampling2D()(x)
# x = Conv2D(512,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(512,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(512,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
 
# x = UpSampling2D()(x)
# x = Conv2D(512,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(512,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(512,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
 
# x = UpSampling2D()(x)
# x = Conv2D(256,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(256,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(256,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
 
# x = UpSampling2D()(x)
# x = Conv2D(128,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(128,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
 
# x = UpSampling2D()(x)
# x = Conv2D(64,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(64,3, activation='relu', padding='same')(x)
# x = BatchNormalization()(x)

#Final prediction layer
soft5 = Conv2D(dims, 1, activation='softmax', padding='same')(drop5)


model = Model(input_tensor,soft5)
model.summary()
model.compile(
	loss='categorical_crossentropy',
	optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
	metrics=['accuracy']
	
)

history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val,y_val), class_weight=balanced_freq)

	# class_weight=[ 
	# 0.0931816,   0.07889178,  0.0683163,  0.0599477,  0.0576133,  
	# 0.0596811, 0.0649174,  0.0713340,  0.0777491,  0.0805642,  
	# 0.0774501,  0.0663205, 0.0514884,  0.0368726,  0.0278029,  
	# 0.0278682
	# ]

model.save('ascii_nn4.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo')
plt.plot(epochs, val_acc, 'b')
plt.title('Train/Val Accuracy')

plt.figure()

plt.plot(epochs,loss,'bo')
plt.plot(epochs,val_loss,'b')
plt.title('Train/Val Loss')

plt.show()


# x = l1_1(input_tensor)
# o1 = l1_2(x)
# x = l1_p(o1)
# x = l2_1(x)
# o2 = l2_2(x)
# x = l2_p(o2)
# x = l3_1(x)
# x = l3_2(x)
# o3 = l3_3(x)
# x = l3_p(o3)
# x = l4_1(x)
# x = l4_2(x)
# o4 = l4_3(x)
# x = l4_p(o4)
# x = l5_1(x)
# x = l5_2(x)
# o5 = l5_3(x)
# x = l5_p(o5)




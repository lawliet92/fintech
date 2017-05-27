
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras import callbacks
from keras.callbacks import ModelCheckpoint,  EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from numpy import loadtxt
import cv2
from keras.utils import np_utils
import numpy as np
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model



vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(192, 192,3))

output_vgg16 = vgg16.output
x = Flatten(name='flatten')(output_vgg16)
x = Dense(4096, activation="relu", name='fc1')(x)
x = Dense(4096, activation="relu", name='fc2')(x)
predictions = Dense(7, activation='softmax', name='predictions')(x)
model = Model(inputs=vgg16.input, outputs=predictions)
#model.load_weights('/Volumes/JasonWork/Data/fer2013/TF_Vgg16_FC_weights-improvement-99-0.64.hdf5')

sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


for layer in model.layers:
    if layer.name in ['fc1', 'fc2', 'predictions']:
        continue
    layer.trainable = False


model.summary()

print "Finish Loading Weights"


# load data
my_data = loadtxt('/home/jiay/fintech/fer2013.csv', delimiter=",")
my_data = my_data.reshape((35887, 48, 48))
my_data = np.float32(my_data)
my_data = [cv2.cvtColor(cv2.resize(i,(192, 192)), cv2.COLOR_GRAY2BGR) for i in my_data]
my_data = np.concatenate([arr[np.newaxis] for arr in my_data]).astype('float32')

label = loadtxt('/home/jiay/fintech/labels.csv', delimiter=",")
label = np.float32(label)

print "Finish Loading Data"


label = np_utils.to_categorical(label)
x_train = my_data[0:28709]
y_train = label[0:28709]

x_val = my_data[28709:]
y_val = label[28709:]


train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 0.2,
    horizontal_flip=True,
    fill_mode='nearest')


validation_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=100)

validation_datagen.fit(x_val)
validation_generator = validation_datagen.flow(x_val, y_val, batch_size=100)

print "Finish Data Augmentation"



filepath='/home/jiay/fintech/results/face224_weights-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
callbacks_list = [checkpoint,early]

model.fit_generator(train_generator,
                    steps_per_epoch=len(x_train) / 100,
                    validation_data=validation_generator,
                    validation_steps=len(x_val) / 100,
                    epochs=100,
                    callbacks=callbacks_list,
                    verbose=1)

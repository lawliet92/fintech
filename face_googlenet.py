from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from numpy import loadtxt
import cv2
from keras.utils import np_utils
import numpy as np




# create the base pre-trained model
inceptionv3 = InceptionV3(weights='imagenet', include_top=False)
keras_input = Input(shape=(96,96, 3), name='image_input')
# Use the generated model
output_inceptionv3 = inceptionv3(keras_input)
x = GlobalAveragePooling2D(name='gap2d')(output_inceptionv3)
x = Dense(1024, activation="relu", name='fc1')(x)
predictions = Dense(7, activation='softmax', name='predictions')(x)

# Create your own model
model = Model(inputs=keras_input, outputs=predictions)

for layer in model.layers:
    if layer.name in ['fc1', 'gap2d', 'predictions']:
        continue
    layer.trainable = False

sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print "Finish Loading Weights"

model.summary()

# load data
my_data = loadtxt('/home/jiay/TransferLearning/facial/fer2013.csv', delimiter=",")
my_data = my_data.reshape((35887, 48, 48))
my_data = np.float32(my_data)
my_data = [cv2.cvtColor(cv2.resize(i,(96,96)), cv2.COLOR_GRAY2BGR) for i in my_data]
my_data = np.concatenate([arr[np.newaxis] for arr in my_data]).astype('float32')
#my_data = np.transpose(my_data, (0,3,1,2))

label = loadtxt('/home/jiay/TransferLearning/facial/labels.csv', delimiter=",")
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
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 0.2,
    horizontal_flip=True,
    fill_mode='nearest')


validation_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

train_datagen.fit(x_train)
train_generator = train_datagen.flow(x_train, y_train, batch_size=8)

validation_datagen.fit(x_val)
validation_generator = validation_datagen.flow(x_val, y_val, batch_size=8)



filepath='/home/jiay/TransferLearning/facial/TF_GoogleNet_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
callbacks_list = [checkpoint,early]

model.fit_generator(train_generator,
                    steps_per_epoch=len(x_train) / 8,
                    validation_data=validation_generator,
                    validation_steps=len(x_val) / 8,
                    epochs=100,
                    callbacks=callbacks_list,
                    verbose=1)


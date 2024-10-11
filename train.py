# image classification: Bear dataset
# https://www.kaggle.com/datasets/hoturam/bear-dataset?resource=download
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras
import tensorflow as tf                
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.applications import VGG16

logger.add('log.log')

#  activate GPU(s)
print(tf.__version__)
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))

# training dataset
trpath = "./Kaggle/data/"

# load dataset
trdata = ImageDataGenerator(validation_split=0.2)
traindata = trdata.flow_from_directory(directory=trpath, 
                                       target_size=(256,256),
                                       shuffle=True,
                                       subset='training')
valdata = trdata.flow_from_directory(directory=trpath, 
                                     target_size=(256,256), 
                                     shuffle=True,
                                     subset='validation')

spe = traindata.samples // traindata.batch_size # steps_per_epoch
vs = valdata.samples // traindata.batch_size # validation_steps

# build model
inputs = tf.keras.Input(shape=(256, 256, 3))

# data augmentation layer
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)
base_model = data_augmentation(inputs)
base_model = VGG16(include_top=False, weights='imagenet', input_tensor=base_model)

x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(4096, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Dense(4096, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
outputs = layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# freeze
for layer in base_model.layers:
    layer.trainable = False

# compile model
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])
model.summary()
checkpoint = ModelCheckpoint("model.h5", 
                              monitor="val_accuracy", 
                              verbose=1, 
                              save_best_only=True, 
                              save_weights_only=False, 
                              mode='max', 
                              period=1)
csv_logger = CSVLogger("training.log")
early = EarlyStopping(monitor="val_accuracy", 
                      min_delta=0, 
                      patience=20, 
                      verbose=1, 
                      mode="max")
hist = model.fit_generator(steps_per_epoch=spe, 
                            generator=traindata, 
                            validation_data=valdata, 
                            validation_steps=vs, 
                            epochs=100,
                            callbacks=[checkpoint, csv_logger, early])

# plot training result
plt.figure(figsize=(9, 4))
plt.subplot(121)
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy"])
plt.subplot(122)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss","Validation Loss"])
# plt.show()
plt.savefig("plot.png")

# evaluate model
logger.info('Max Validation Accuracy: '+str(max(hist.history["val_accuracy"])))

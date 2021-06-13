from inspect import Arguments
from PIL.Image import Image
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.python.keras.applications.densenet import preprocess_input
from tensorflow.python.ops.gen_array_ops import fill
from helper_func import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default='plot.png',
                help="The path to save the accuracy plot")
args = vars(ap.parse_args())

INIT_LR = 1e-4
EPOCHS = 5
BS = 32

print('[INFO] loading images')
imagePaths = list(paths.list_images(config.BASE_PATH))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = load_img(imagePath, target_size=config.INPUT_DIMS)
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
#headModel = Dense(2, activation="softmax")(headModel)
headModel = (Dense(2, kernel_regularizer=tf.keras.regularizers.l2(
    0.0001), activation='softmax'))(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")

opt = Adam(lr=INIT_LR)
#model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
model.compile(optimizer='adam', loss='squared_hinge', metrics=['accuracy'])

print("[INFO] training head...")
H = model.fit(aug.flow(X_train, y_train, batch_size=BS), steps_per_epoch=len(
    X_train)//BS, validation_data=(X_test, y_test), validation_steps=len(X_test)//BS, epochs=EPOCHS)

print("[INFO] evaluating network..")
predIdxs = model.predict(X_test, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(y_test.argmax(
    axis=1), predIdxs, target_names=lb.classes_))

print("[INFO] saving detector model...")
model.save(config.MODEL_PATH, save_format="h5")

print("[INFO] saving label encoder..")
f = open(config.ENCODER_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, N), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, N), H.history['accuracy'], label="train_acc")
plt.plot(np.arange(0, N), H.history['val_accuracy'], label="val_acc")
plt.title("Training Loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])

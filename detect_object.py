from tensorflow.python.keras.backend import dtype
from helper_func.nms import non_max_suppresion
from helper_func import config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

image = cv2.imread(args['image'])
image = imutils.resize(image, width=600)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

proposals = []
boxes = []
lst = []
for i in range(len(rects)):
    if (rects[i][2] == image.shape[1]) and (rects[i][3] == image.shape[0]):
        lst.append(i)


for i, (x, y, w, h) in enumerate(rects):

    if(i not in lst):
        roi = image[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)

        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        proposals.append(roi)
        boxes.append((x, y, x+w, y+h))


proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")
print("[INFO] proposals shape: {}".format(proposals.shape))

print("[INFO] classifying proposals...")
proba = model.predict(proposals)

labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == "table")[0]

boxes = boxes[idxs]
proba = proba[idxs][:, 1]

idxs = np.where(proba >= config.MIN_PROBA)
boxes = boxes[idxs]
proba = proba[idxs]

clone = image.copy()

for (box, prob) in zip(boxes, proba):

    (startX, startY, endX, endY) = box
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY+10
    text = "Table : {:.2f}%".format(prob*100)
    cv2.putText(clone, text, (startX, y),
                cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 255, 0), 2)

cv2.imshow("BEFORE NMS", clone)
# cv2.waitKey(0)

boxIdxs = non_max_suppresion(boxes, proba)

for i in boxIdxs:
    (startX, startY, endX, endY) = boxes[i]

    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text = "Table : {:.2f}%".format(prob*100)
    cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 255, 0), 2)


cv2.imshow("After NMS", image)
cv2.waitKey(0)

from tensorflow.python.keras.backend import dtype
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from helper_func.nms import non_max_suppresion
from tensorflow.python.keras.backend import reset_uids
from tensorflow.python.ops.gen_math_ops import Imag
from helper_func.slidingWindowAndImagePyramid import slidingWindow
from helper_func.slidingWindowAndImagePyramid import ImagePyramid
from helper_func import config
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
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import time
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200,150)")
ap.add_argument("-c", "--min_conf", default=0.9, type=float,
                help="Minimum probability score/Threshold")
ap.add_argument("-v", "--visualize", type=int, default=-1,
                help="Whether or not to visulize the sliding window")

args = vars(ap.parse_args())


def get_rois(image):
    WIDTH = 600
    PYR_SCALE = 1.5
    WIN_STEP = 10
    ROI_SIZE = (eval(args["size"]))
    INPUT_SIZE = config.INPUT_DIMS  # MODEL INPUT SIZE

    orig = cv2.imread(args['image'])
    orig = imutils.resize(orig, width=WIDTH)
    (H, W) = orig.shape[:2]

    pyramid = ImagePyramid(orig, scale=PYR_SCALE, minImageSize=ROI_SIZE)

    rois = []
    locs = []

    start = time.time()

    for image in pyramid:
        scale = W/float(image.shape[1])

        for (x, y, roiOrig) in slidingWindow(image, WIN_STEP, ROI_SIZE):

            x = int(x*scale)
            y = int(y*scale)
            w = int(ROI_SIZE[0]*scale)
            h = int(ROI_SIZE[1]*scale)

            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)

            rois.append(roi)
            locs.append((x, y, x+w, y+h))

            if args['visualize'] > 0:
                clone = orig.copy()
                cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imshow("Visualizations", clone)
                cv2.imshow("ROI", roiOrig)
                cv2.waitKey(0)

    end = time.time()

    print("LOOPING OVER PYRAMID WINDOW TOOK: {:.2f}".format(end-start))

    rois = np.array(rois, dtype='float32')
    locs = np.array(locs, dtype="int32")
    return rois, locs


model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

image = cv2.imread(args['image'])
image = imutils.resize(image, width=600)


proposals, boxes = get_rois(image)

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

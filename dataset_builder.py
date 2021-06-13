from helper_func.iou import compute_iou
from helper_func import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os


'''
Generate Comumn and Table mask from Marmot Data
'''

import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image

# Returns if columns belong to same table or not


def sameTable(ymin_1, ymin_2, ymax_1, ymax_2):
    min_diff = abs(ymin_1 - ymin_2)
    max_diff = abs(ymax_1 - ymax_2)

    if min_diff <= 5 and max_diff <= 5:
        return True
    elif min_diff <= 4 and max_diff <= 7:
        return True
    elif min_diff <= 7 and max_diff <= 4:
        return True
    return False


def get_regions(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    size = root.find('size')

    # Parse width
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Create grayscale image array
    col_mask = np.zeros((height, width), dtype=np.int32)
    table_mask = np.zeros((height, width), dtype=np.int32)

    got_first_column = False
    i = 0
    table_xmin = 10000
    table_xmax = 0

    table_ymin = 10000
    table_ymax = 0

    regions = []
    for column in root.findall('object'):
        bndbox = column.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # col_mask[ymin:ymax, xmin:xmax] = 255

        if got_first_column:
            if sameTable(prev_ymin, ymin, prev_ymax, ymax) == False:
                i += 1
                got_first_column = False
                # table_mask[table_ymin:table_ymax,
                #            table_xmin:table_xmax] = 255

                regions.append(
                    (table_xmin, table_ymin, table_xmax, table_ymax))

                table_xmin = 10000
                table_xmax = 0

                table_ymin = 10000
                table_ymax = 0

        if got_first_column == False:
            got_first_column = True
            first_xmin = xmin

        prev_ymin = ymin
        prev_ymax = ymax

        table_xmin = min(xmin, table_xmin)
        table_xmax = max(xmax, table_xmax)

        table_ymin = min(ymin, table_ymin)
        table_ymax = max(ymax, table_ymax)

    # table_mask[table_ymin:table_ymax, table_xmin:table_xmax] = 255
    regions.append((table_xmin, table_ymin, table_xmax, table_ymax))

    #im = Image.fromarray(col_mask.astype(np.uint8), 'L')
    #im.save(final_col_directory + filename + ".jpeg")

    # im = Image.fromarray(table_mask.astype(np.uint8), 'L')
    # im.save(final_table_directory + filename + ".jpeg")
    return regions


for dirPath in (config.POSITIVE_PATH, config.NEGATIVE_PATH):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

imagePaths = list(paths.list_images(config.ORIG_IMAGES))

totalPostitive = 0
totalNegative = 0

for(i, imagePath) in enumerate(imagePaths):
    try:
        print("[INFO] processing image: {}/{}...".format(i+1, len(imagePaths)))

        filename = imagePath.split(os.path.sep)[-1]
        filename = filename[:filename.rfind('.')]
        annotPath = os.path.sep.join(
            [config.ORIG_ANNOTS, "{}.xml".format(filename)])

        contents = open(annotPath).read()

        soup = BeautifulSoup(contents, 'html.parser')

        gtBoxes = []

        # w = int(soup.find('width').string)
        # h = int(soup.find('height').string)

        # for o in soup.find_all("object"):
        #     label = o.find("name").string
        #     xMin = int(o.find('xmin').string)
        #     yMin = int(o.find('ymin').string)
        #     xMax = int(o.find('xmax').string)
        #     yMax = int(o.find('ymax').string)

        #     xMin = max(0, xMin)
        #     yMin = max(0, yMin)
        #     xMax = min(w, xMax)
        #     yMax = min(h, yMax)

        #     gtBoxes.append((xMin, yMin, xMax, yMax))

        # gtTables = []
        # ind_done = []
        # for i in range(len(gtBoxes)):
        #     if(i not in ind_done):
        #         x1mn = gtBoxes[i][0]
        #         y1mn = gtBoxes[i][1]
        #         x1mx = gtBoxes[i][2]
        #         y1mx = gtBoxes[i][3]
        #         table = (x1mn, y1mn, x1mx, y1mx)
        #         for j in range(i+1, len(gtBoxes)):
        #             x2mn = gtBoxes[j][0]
        #             y2mn = gtBoxes[j][1]
        #             x2mx = gtBoxes[j][2]
        #             y2mx = gtBoxes[j][3]

        #             if(abs(y1mn-y2mn) < 10):
        #                 table = (int(min(table[0], x2mn)), int(min(table[1], y2mn)), int(
        #                     max(table[2], x2mx)), int(max(table[3], y2mx)))
        #                 # gtBoxes.pop(j)
        #                 ind_done.append(i)
        #                 ind_done.append(j)

        #         gtTables.append(table)

        # gtBoxes = gtTables

        gtBoxes = get_regions(annotPath)
        image = cv2.imread(imagePath)

        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        proposedRects = []

        for(x, y, w, h) in rects:
            proposedRects.append((x, y, x+w, y+h))

        positiveROIs = 0
        negativeROIs = 0

        for proposedRect in proposedRects[:config.MAX_PROPOSALS]:
            (propStartX, propStartY, propEndX, propEndY) = proposedRect

            for gtBox in gtBoxes:
                iou = compute_iou(gtBox, proposedRect)
                (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

                roi = None
                outputPath = None

                if iou > 0.7 and positiveROIs <= config.MAX_POSITIVE:
                    roi = image[propStartY:propEndY, propStartX:propEndX]
                    filename = "{}.png".format(totalPostitive)
                    outputPath = os.path.sep.join(
                        [config.POSITIVE_PATH, filename])

                    positiveROIs += 1
                    totalPostitive += 1

                fullOverlap = propStartX >= gtStartX
                fullOverlap = fullOverlap and propStartY >= gtStartY
                fullOverlap = fullOverlap and propEndX <= gtEndX
                fullOverlap = fullOverlap and propEndY <= gtEndY

                if not fullOverlap and iou < 0.05 and negativeROIs <= config.MAX_NEGATIVE:
                    roi = image[propStartY:propEndY, propStartX:propEndX]
                    filename = "{}.png".format(totalNegative)
                    outputPath = os.path.sep.join(
                        [config.NEGATIVE_PATH, filename])

                    negativeROIs += 1
                    totalNegative += 1

                if (roi is not None) and (outputPath is not None):
                    roi = cv2.resize(roi, config.INPUT_DIMS,
                                     interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(outputPath, roi)

        # cv2.imwrite(os.path.sep.join(
        #                 [config.NEGATIVE_PATH, filename]))

    except:
        print("[INFO]----- FILE NOT FOUND")

    #print(imagePath, gtTables)

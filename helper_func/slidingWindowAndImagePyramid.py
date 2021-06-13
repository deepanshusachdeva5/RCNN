import imutils


def slidingWindow(image, step, window):
    for y in range(0, image.shape[0]-window[1], step):
        for x in range(0, image.shape[1]-window[0], step):
            yield(x, y, image[y:y+window[1], x:x+window[0]])


def ImagePyramid(image, scale=1.5, minImageSize=(224, 224)):

    yield image
    while True:

        w = int(image.shape[1]/scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < minImageSize[1] or image.shape[1] < minImageSize[0]:
            break
        yield image

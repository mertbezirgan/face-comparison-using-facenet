import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
# from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model

### cv2 cascade for detecting faces
cascade_path = 'model\cv2\haarcascade_frontalface_alt2.xml'
### image size that for resizing image to only face after detection
image_size = 160

### pretrained facenet model
model = load_model("model\\facenet_keras.h5")

labeled_data = pd.read_csv("formatted-7000-data.csv")
real_results = np.array(labeled_data["label"])
image1_paths = np.array(labeled_data["image1"])
image2_paths = np.array(labeled_data["image2"])


# print(len(image1_paths), " image 1 count")
# print(len(image2_paths), " image 2 count")
# image1_paths = image1_paths[0:4000]
# image2_paths = image2_paths[0:4000]
# print(len(image1_paths), " image 1 count")
# print(len(image2_paths), " image 2 count")

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def load_and_align_images(filepaths, margin):
    counter = 0
    cascade = cv2.CascadeClassifier(cascade_path)

    aligned_images = []
    for filepath in filepaths:
        if counter % 20 == 0:
            print('align stage {}/{}'.format(counter, len(filepaths)))
        img = imread(filepath)

        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        (x, y, w, h) = faces[0]
        # cropped = img[y - margin // 2:y + h + margin // 2, x - margin // 2:x + w + margin // 2, :]
        cropped = img[max(y - 10 // 2, 0):y + h + 10 // 2, max(x - 10 // 2, 0):x + w + 10 // 2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
        counter += 1
    return np.array(aligned_images)

### main function that returns feature vector from given image
def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
    pd = []
    current = 0
    for start in range(0, len(aligned_images), batch_size):
        total = len(aligned_images)
        if current % 5 == 0:
            print('{} / {}'.format(current, total))
        pd.append(model.predict_on_batch(aligned_images[start:start + batch_size]))
        current += 1
    embs = l2_normalize(np.concatenate(pd))

    return embs


def calc_cosine_dist(v1, v2):
    return distance.cosine(v1, v2)


min_im1_paths = []
min_im2_paths = []
min_labels = []

###
for i in range(0, len(image1_paths)):
    min_im1_paths.append(image1_paths[i])
    min_im2_paths.append(image2_paths[i])
    min_labels.append(real_results[i])
# im1_paths = []
# im2_paths = []
# for i in range(0, len(image1_paths)):
#     im1_paths.append(image1_paths[i])
#     im2_paths.append(image2_paths[i])


### calculate vectors
im1_embs = calc_embs(min_im1_paths)
im2_embs = calc_embs(min_im2_paths)
# im1_embs = calc_embs(im1_paths)
# im2_embs = calc_embs(im2_paths)


### write cosine distances of vectors that must be same person to same_data_length txt file and different people people to different_data_length file
with open("same_7000.txt", "w") as same_file:
    with open("different_7000.txt", "w") as different_file:
        # for i in range(0, len(real_results)):
        for i in range(0, len(min_labels)):
            if i == 0:
                same_file.write(str(calc_cosine_dist(im1_embs[i], im2_embs[i])) + "\n")
            elif i % 2 == 0:
                same_file.write(str(calc_cosine_dist(im1_embs[i], im2_embs[i])) + "\n")
            else:
                different_file.write(str(calc_cosine_dist(im1_embs[i], im2_embs[i])) + "\n")

# for i in range(0, 11):
#     print(calc_cosine_dist(im1_embs[i], im2_embs[i]), " real result was", str(min_labels[i]))

# with open("results.txt", "w") as output:
#     for i in range(1, len(im1_embs)):
#         output.write(str(calc_cosine_dist(im1_embs[i], im2_embs[i])))

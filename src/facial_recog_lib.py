#!/usr/bin/python3
import numpy as np
from PIL import Image
import os

def read_image(picture):
    img = Image.open(picture)
    img = img.convert("L")
    img_matrix = np.asarray(img, dtype=np.uint8)
    img_vec = np.array([])
    for row in img_matrix:
        img_vec = np.append(img_vec, row)
    return img_vec


def covariance(path):
    vectors = []
    folders = os.listdir(path)
    folders.remove("README")
    for folder in folders:
        newFolder = os.listdir(path + '/' + folder)
        for picture in newFolder:
            picture = path + '/' + folder + '/' + picture
            vectors.append(read_image(picture))
    
    vectors = np.vstack(vectors)
    avg = np.mean(vectors, axis=0)
    
    for index in range(len(vectors)):
        vectors[index,:] = vectors[index,:] - avg
    
    covar = np.dot(vectors, vectors.T)/len(vectors)
    
    return covar
        


if __name__ == '__main__':
    covariance("/home/ubuntu/workspace/faces")
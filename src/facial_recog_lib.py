#!/usr/bin/python3
import numpy as np
from PIL import Image
import os


def read_image(picture):
    img = Image.open(picture)
    # img = img.convert("L")
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
        vectors[index, :] = vectors[index, :] - avg

    covar = np.dot(vectors, vectors.T) / len(vectors)

    return vectors.T, covar, avg


def eigenStuff(vectors, covar, epsilon):
    evals, evecs = np.linalg.eig(covar)
    edict = {}
    for index in range(len(evals)):
        edict[evals[index]] = evecs[index]
    variance = sum(evals) * len(vectors)
    principle_components = np.array([evals[0]])
    # this epsilon stuff makes it so that we only use our most important eigen
    # values
    evals = sorted(evals, reverse=True)
    index = 1
    while (len(vectors) * sum(principle_components)) / variance <= epsilon:
        principle_components = np.append(principle_components, evals[index])
        index += 1

    newEvecs = []
    for val in principle_components:
        mult = np.dot(vectors, edict[val])
        newEvecs.append(mult)

    newEvecs = np.vstack(newEvecs).T
    # print(newEvecs.shape)
    return principle_components, newEvecs


def find_weight(evecs, x, mean):
    return np.dot(evecs.T, x-mean)

def reconstruct(evecs, weights, mean):
    return np.dot(evecs, weights) + mean

def vectorToImage(vector):
    """This function will convert the vector to images"""
    vec = []
    for index in range(112):
        vec.append(vector[index * 92:index * 92 + 92])
    vec = np.vstack(vec)
    print(np.max(vec))
    img = Image.fromarray(vec)
    img.show()


if __name__ == '__main__':
    vectors, covar, avg = covariance("/home/ubuntu/workspace/faces")
    pca, newEvecs = eigenStuff(vectors, covar, 0.85)

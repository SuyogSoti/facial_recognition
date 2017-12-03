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
    folders.remove("newFaces")
    # folders.remove("avery")
    for folder in folders:
        newFolder = os.listdir(path + '/' + folder)
        for picture in newFolder:
            picture = path + '/' + folder + '/' + picture
            vectors.append(read_image(picture))

    vectors = np.vstack(vectors)
    avg = np.mean(vectors, axis=0)

    for index in range(len(vectors)):
        vectors[index] = vectors[index] - avg

    covar = np.dot(vectors, vectors.T) / len(vectors)

    return vectors.T, covar, avg


def eigenStuff(vectors, covar, k):
    evals, evecs = np.linalg.eigh(covar)
    edict = {}
    for index in range(len(evals)):
        edict[evals[index]] = evecs[index]
    principle_components = np.array([evals[0]])
    # this epsilon stuff makes it so that we only use our most important eigen
    # values
    evals = sorted(evals, reverse=True)
    principle_components = evals[:k]
    newEvecs = []
    print("Number of eigen vectors: " + str(len(principle_components)))
    for val in principle_components:
        mult = np.dot(vectors, edict[val])
        newEvecs.append(mult)

    newEvecs = np.vstack(newEvecs).T
    # print(newEvecs.shape)
    return principle_components, newEvecs


def find_weight(evecs, x, mean=0):
    weight = np.dot(evecs.T, x - mean)
    return weight


def reconstruct(evecs, weights, mean):
    og = np.dot(evecs, weights) + mean
    # return (og*255)/max(og)
    return og


def vectorToImage(vector):
    """This function will convert the vector to images"""
    vec = []
    for index in range(112):
        vec.append(vector[index * 92:index * 92 + 92])
    vec = np.vstack(vec)
    img = Image.fromarray(vec)
    img.show()


def recognize_face(inputFace, k=100):
    """
        inputFace is a path to a picture
        epsilon is 0.85 by default but user can specify
    """
    path = os.path.realpath(__file__).split("/")
    path = path[0:len(path) - 1]
    path = "/".join(path)
    path += "/../faces"
    vectors, covar, avg = covariance(path)
    transVec = vectors.T
    pca, newEvecs = eigenStuff(vectors, covar, k)
    in_vec = read_image(inputFace)
    in_weight = find_weight(newEvecs, in_vec, avg)
    weights = np.array([find_weight(newEvecs, x) for x in transVec])

    err = [np.linalg.norm(in_weight - row) for row in weights]
    index = np.argmin(err)
    min_err = err[index]
    print("Min Err:", min_err)
    if min_err < 5700000:
        print("This face exists in our database")
    elif min_err < 7000000:
        print("This image is most likely a face but does not exist in our database")
    else:
        print("Please make sure the input image is a face")
    vectorToImage(in_vec)
    vectorToImage(reconstruct(newEvecs, in_weight, avg))
    vectorToImage(transVec[index] + avg)
    # [vectorToImage(x) for x in newEvecs.T]


if __name__ == '__main__':
    path = os.path.realpath(__file__).split("/")
    path = path[0:len(path) - 1]
    path = "/".join(path)
    path += "/../faces/newFaces/4.jpg"
    recognize_face(path, k=10)

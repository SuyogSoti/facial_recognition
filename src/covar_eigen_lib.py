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

def normalize(X, low, high):
    minX, maxX = np.min(X), np.max(X)
    X = X-minX
    X = X/(maxX-minX)
    X = X*(high-low)
    X = X + low
    return X

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
    evals, evecs = np.linalg.eig(covar)
    edict = {}
    
    for index in range(len(evals)):
        edict[evals[index]] = evecs[index]
    # this epsilon stuff makes it so that we only use our most important eigen
    # values
    evals = sorted(evals, reverse=True)
    principle_components = evals[:k]
    newEvecs = []
    print("Number of eigen vectors: " + str(len(principle_components)))
    for val in principle_components:
        mult = np.dot(vectors, edict[val])
        newEvecs.append(normalize(mult, 0, 255))

    newEvecs = np.vstack(newEvecs).T

    np.savetxt("eigen_matrix.csv", newEvecs, delimiter=',')
    # print(newEvecs.shape)
    return principle_components, newEvecs

def find_weight(evecs, x, mean=0):
    weight = np.dot(evecs.T, x - mean)
    return weight/np.linalg.norm(weight)


def reconstruct(evecs, weights, mean):
    og = normalize(np.dot(evecs, weights) + mean, 0 ,255)
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


def count_faces():
    files = os.walk('../faces/')
    count = 0
    for f in files:
        if '/s' in f[0]:
            count += len(f[2])
    return int(count)


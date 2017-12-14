#!/usr/bin/python3
import numpy as np
from PIL import Image
import os
import sys


def normalize(X, low, high):
    minX, maxX = np.min(X), np.max(X)
    X = X-minX
    X = X/(maxX-minX)
    X = X*(high-low)
    X = X + low
    return X


# Print iterations progress
def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def read_image(picture):
    img = Image.open(picture)
    img = img.convert("L")
    baseheight = 100
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    img_matrix = np.asarray(img, dtype=np.uint8)
    img_vec = np.array([])
    for row in img_matrix:
        img_vec = np.append(img_vec, row)
    if len(img_vec) != 15200:
        return None
    return img_vec


def covariance(path):
    vectors = []
    folders = ["benign", "malignant"]
    printProgressBar(0, 1807, prefix='Progress:', suffix='Complete', length=50)
    safe = 0
    for folder in folders:
        newFolder = os.listdir(path + '/' + folder)
        for cancer in newFolder:
            if "." not in cancer:
                cancer = path + '/' + folder + '/' + cancer
                sobs = os.listdir(cancer)
                for sob in sobs:
                    sob = cancer + "/" + sob
                    zooms = [sob + "/" + x + "/400X" for x in os.listdir(sob)]
                    for zoom in zooms:
                        temp = os.listdir(zoom)
                        for x in temp:
                            x = zoom + "/" + x
                            tmp = read_image(x)
                            if tmp is not None:
                                vectors.append(tmp)
                            printProgressBar(
                                len(vectors),
                                1807,
                                prefix='Progress:',
                                suffix='Complete',
                                length=50)
        if folder is folders[0]:
            safe = len(vectors)
            print(folder, safe)

    vectors = np.vstack(vectors)
    print("Shape of the vector is", vectors.shape)
    avg = np.mean(vectors, axis=0)

    for index in range(len(vectors)):
        vectors[index] = vectors[index] - avg

    covar = np.dot(vectors, vectors.T) / len(vectors)

    return [vectors.T, covar, avg, safe]


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

    # print(newEvecs.shape)
    return principle_components, newEvecs


def find_weight(evecs, x, mean=0):
    weight = np.dot(evecs.T, x - mean)
    return weight / np.linalg.norm(weight)


def reconstruct(evecs, weights, mean):
    og = normalize(np.dot(evecs, weights) + mean, 0, 255)
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

#!/usr/bin/python3
import numpy as np
import covar_eigen_boobs_lib as cel
import os
import sys


def recognize_face(inputFace, cov, newEvecs, pca, weights, transVec):
    """
        inputFace is a path to a picture
        epsilon is 0.85 by default but user can specify
    """
    avg = cov[2]
    safe = cov[3]
    in_vec = cel.read_image(inputFace)
    in_weight = cel.find_weight(newEvecs, in_vec, avg)
    err = [np.linalg.norm(in_weight - row) for row in weights]
    index = np.argmin(err)
    min_err = err[index]
    print("Min Err:", len(pca), ":", min_err)
    if index < safe:
        print("is NOT cancer")
    else:
        print("is CANCER")
    # [cel.vectorToImage(x) for x in newEvecs.T[:k]]


if __name__ == '__main__':
    path = os.path.realpath(__file__).split("/")
    path = path[0:len(path) - 1]
    path = "/".join(path)
    path += "/../breast"
    covar = cel.covariance(path)
    path = os.path.realpath(__file__).split("/")
    path = path[0:len(path) - 1]
    path = "/".join(path)
    # path += "/../faces/newFaces/4.jpg"
    # path += "/../faces/newFaces/avery.jpg"
    # path += "/../cancer/SOB_M_DC-14-12312-400-001.png"
    pca, newEvecs = cel.eigenStuff(covar[0], covar[1], 1000)
    transVec = covar[0].T
    weights = np.array([cel.find_weight(newEvecs, x) for x in transVec])

    path += "/../cancer"
    folder = os.listdir(path)
    for pic in folder:
        recognize_face(path + "/" + pic, covar, newEvecs, pca, weights, transVec)
    # for i in range(1, 50):
    #     recognize_face(path, covar, k=i)

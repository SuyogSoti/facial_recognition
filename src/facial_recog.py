#!/usr/bin/python3
import numpy as np
from PIL import Image
import covar_eigen_lib as cel
import os

def recognize_face(inputFace, k=100):
    """
        inputFace is a path to a picture
        epsilon is 0.85 by default but user can specify
    """
    path = os.path.realpath(__file__).split("/")
    path = path[0:len(path) - 1]
    path = "/".join(path)
    path += "/../faces"
    vectors, covar, avg = cel.covariance(path)
    transVec = vectors.T
    #pca, newEvecs = cel.eigenStuff(vectors, covar, k)
    
    currdir = os.listdir()
    curr_k = len(np.genfromtxt("eigen_matrix.csv", delimiter=',').T)
    face_count = cel.count_faces()
    if "eigen_matrix.csv" not in currdir or face_count != len(transVec) or k != curr_k:
        print("Eigenmatrix is not correct, creating eigenmatrix")
        cel.eigenStuff(vectors, covar, k)
    newEvecs = np.genfromtxt("eigen_matrix.csv", delimiter=',')
    
    in_vec = cel.read_image(inputFace)
    in_weight = cel.find_weight(newEvecs, in_vec, avg)
    weights = np.array([cel.find_weight(newEvecs, x) for x in transVec])

    err = [np.linalg.norm(in_weight - row) for row in weights]
    index = np.argmin(err)
    min_err = err[index]
    print("Min Err:", min_err)
    if min_err < 0.07:
        print("This face exists in our database")
    elif min_err < 0.14:
        print("This image is most likely a face but does not exist in our database")
    else:
        print("Please make sure the input image is a face")
    cel.vectorToImage(in_vec)
    #cel.vectorToImage(cel.reconstruct(newEvecs, in_weight, avg))
    cel.vectorToImage(transVec[index] + avg)
    #[cel.vectorToImage(x) for x in newEvecs.T[:k]]


if __name__ == '__main__':
    path = os.path.realpath(__file__).split("/")
    path = path[0:len(path) - 1]
    path = "/".join(path)
    #path += "/../faces/newFaces/4.jpg"
    path += "/../faces/newFaces/4.jpg"
    recognize_face(path, k=10)

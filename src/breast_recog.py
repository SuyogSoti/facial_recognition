#!/usr/bin/python3
import numpy as np
import covar_eigen_boobs_lib as cel
import os, sys


def recognize_face(inputFace, k=100):
    """
        inputFace is a path to a picture
        epsilon is 0.85 by default but user can specify
    """
    path = os.path.realpath(__file__).split("/")
    path = path[0:len(path) - 1]
    path = "/".join(path)
    path += "/../breast"
    transVec, covar, avg, safe = cel.covariance(path)
    myf = open("overall.csv", 'w')
    myf.write("index, predic, real, accurate, err\n")
    # for vec_index in range(100):
    #     in_vec = transVec[vec_index]
    #     transVec = np.delete(transVec, vec_index, 0)
    #     vectors = transVec
    #     avg = np.mean(vectors, axis=0)

    #     for index in range(len(vectors)):
    #         vectors[index] = vectors[index] - avg

    #     covar = np.dot(vectors, vectors.T) / len(vectors)

    #     vectors = transVec.T
    #     tempSafe = 0
    #     if vec_index < safe:
    #         tempSafe = safe - 1

    #     principal, newEvecs = cel.eigenStuff(vectors, covar, k)
    #     in_weight = cel.find_weight(newEvecs, in_vec, avg)
    #     weights = np.array([cel.find_weight(newEvecs, x) for x in transVec])

    #     err = [np.linalg.norm(in_weight - row) for row in weights]
    #     index = np.argmin(err)
    #     predic = 0
    #     real = 0
    #     acc = 0
    #     if index < tempSafe:
    #         print("Is Probably Not cancerous")
    #     else:
    #         print("Is probably cancerous")
    #         predic = 1

    #     if vec_index < safe:
    #         print("Real: not cancer")
    #     else:
    #         print("Real: cancer")
    #         real = 1
    #     if real is predic:
    #         acc = 1
    #     min_err = err[index]
    #     mystr = str(vec_index) + ", " + str(predic) + ", " + str(
    #         real) + ", " + str(acc) + ", " + str(min_err) + "\n"
    #     myf.write(mystr)
    #     # print("Min Err:", min_err)
    #     # if min_err < 5700000:
    #     #     print("Our guess is most likeyly correct")
    #     # elif min_err < 7000000:
    #     #     print(
    #     #         "a mamogram but does not exist in our db... our guess is probably correct?"
    #     #     )
    #     # else:
    #     #     print("not a mamogram")
    #     # cel.vectorToImage(in_vec)
    #     # cel.vectorToImage(cel.reconstruct(newEvecs, in_weight, avg))
    #     # cel.vectorToImage(transVec[index] + avg)
    #     # [cel.vectorToImage(x) for x in newEvecs.T[:10]]

    #     # insert the image back here
    #     np.insert(transVec, vec_index, in_vec)
    for vec_index in range(safe+1, safe+21):
        in_vec = transVec[vec_index]
        transVec = np.delete(transVec, vec_index, 0)
        vectors = transVec
        avg = np.mean(vectors, axis=0)

        for index in range(len(vectors)):
            vectors[index] = vectors[index] - avg

        covar = np.dot(vectors, vectors.T) / len(vectors)

        vectors = transVec.T
        tempSafe = 0
        if vec_index < safe:
            tempSafe = safe - 1

        principal, newEvecs = cel.eigenStuff(vectors, covar, k)
        in_weight = cel.find_weight(newEvecs, in_vec, avg)
        weights = np.array([cel.find_weight(newEvecs, x) for x in transVec])

        err = [np.linalg.norm(in_weight - row) for row in weights]
        index = np.argmin(err)
        predic = 0
        real = 0
        acc = 0
        if index < tempSafe:
            print("Is Probably Not cancerous")
        else:
            print("Is probably cancerous")
            predic = 1

        if vec_index < safe:
            print("Real: not cancer")
        else:
            print("Real: cancer")
            real = 1
        if real is predic:
            acc = 1
        min_err = err[index]
        mystr = str(vec_index) + ", " + str(predic) + ", " + str(
            real) + ", " + str(acc) + ", " + str(min_err) + "\n"
        myf.write(mystr)
        # print("Min Err:", min_err)
        # if min_err < 5700000:
        #     print("Our guess is most likeyly correct")
        # elif min_err < 7000000:
        #     print(
        #         "a mamogram but does not exist in our db... our guess is probably correct?"
        #     )
        # else:
        #     print("not a mamogram")
        # cel.vectorToImage(in_vec)
        # cel.vectorToImage(cel.reconstruct(newEvecs, in_weight, avg))
        # cel.vectorToImage(transVec[index] + avg)
        # [cel.vectorToImage(x) for x in newEvecs.T[:10]]

        # insert the image back here
        np.insert(transVec, vec_index, in_vec)
    myf.close()


if __name__ == '__main__':
    path = os.path.realpath(__file__).split("/")
    path = path[0:len(path) - 1]
    path = "/".join(path)
    path += "/../faces/newFaces/4.jpg"
    # path = "/home/suyog/Documents/matrix_methods/facial_recognition/src/../breast/malignant/papillary_carcinoma/SOB_M_PC_14-19440/400X/SOB_M_PC-14-19440-400-024.png"
    # path = "/home/suyog/Documents/matrix_methods/facial_recognition/breast/benign/fibroadenoma/SOB_B_F_14-14134/400X/SOB_B_F-14-14134-400-008.png"
    path = "/home/suyog/Documents/matrix_methods/facial_recognition/SOB_B_A-14-22549AB-400-008.png"
    recognize_face(path, k=50)

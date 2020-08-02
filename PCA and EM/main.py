import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv
import math
import operator
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
import sklearn as sk
import datetime
import random
import sys

dim = 2
tolerance = 0.01

def initialize(K, N):

    W_array = []
    mew_array = []
    sigma_array = []
    P_array = []


    W_array = np.random.rand(K)

    for i in range(K):
        #arr = np.random.randint(low=1,high=9,size=dim)
        arr = np.random.rand(dim)
        arr = np.array(arr).reshape(dim, 1)
        mew_array.append(arr)

        # indv_mew = []
        # for j in range(dim):
        #     arr = np.random.rand(1)
        #     indv_mew.append(arr)
        # mew_array.append(indv_mew)

    for i in range(K):

        #sigma_array.append(sk.datasets.make_spd_matrix(dim))
        while 1:
            indv_covar = []
            for j in range(dim):
                #arr = np.random.randint(low=1,high=9,size=dim)
                arr = np.random.rand(dim)
                indv_covar.append(arr)
            indv_covar = np.array(indv_covar).reshape(dim, dim)
            det = np.linalg.det(indv_covar)
            if(det > 0):
                break

        sigma_array.append(indv_covar)

        # indv_covar = []
        # for j in range(dim):
        #     arr = np.random.rand(dim)
        #     indv_covar.append(arr)
        # indv_covar = np.array(indv_covar).reshape(dim,dim)
        # sigma_array.append(indv_covar)

    for i in range(N):
        arr = np.random.rand(K)
        P_array.append(arr)

    # print("W array:",W_array)
    # print("mew array:", mew_array)
    # print("sigma array:", sigma_array)
    # print("P array:", P_array)

    return W_array, mew_array, sigma_array, P_array

def getGaussian(xi, mewk, sigmak):

    det_sigmak = np.linalg.det(sigmak)
    det_sigmak = np.absolute(det_sigmak)
    #print("det_sigmak",det_sigmak)

    if(det_sigmak == 0):
        print("determinant zero")
        det_sigmak = 0.0000001

    inv_sigmak = np.linalg.inv(sigmak)
    constant = 1.0 / (np.sqrt((np.power(2*np.pi,dim)) * det_sigmak))
    #print("deno const",np.sqrt((np.power(2*np.pi,dim)) * det_sigmak))
    #print("constant:", constant)

    # print("xi", xi)
    # print("mewk", mewk)

    xi_min_mewk = np.subtract(xi,mewk)

    # print(xi_min_mewk,"\nhh",inv_sigmak)

    #print("trans",np.transpose(xi_min_mewk))
    #print("inve sigmak", inv_sigmak)
    temp = np.dot(np.transpose(xi_min_mewk),inv_sigmak)
    #print("temp",temp)
    #print("ximinmewk", xi_min_mewk)
    exp_val = np.dot(temp,xi_min_mewk)
    exp_val = -0.5 * exp_val
    #print("exp val:",exp_val)

    #print("sigmak",sigmak)

    #print(np.multiply(5,mewk))

    if(exp_val < -500):
        exp_val = -500
    elif (exp_val > 500):
        exp_val = 500
    #else:
    ans = constant*np.exp(exp_val)
    #print("gaussian",ans)
    return ans


def calcLogLikelihood(reducedDataset, K, N, mew_array, sigma_array, W_array):

    logLikelihood = 0.0

    for i in range(N):

        xi = reducedDataset[i]
        xi = np.array(xi).reshape(2, 1)

        indv_logLikelihood = 0.0
        for k in range(K):
            Nk = getGaussian(xi, mew_array[k], sigma_array[k])
            # print("Nk",Nk)
            # print("prob", W_array[k])
            indv_logLikelihood += W_array[k] * Nk
            # print("indv log likelihood", indv_logLikelihood)
            # print("log of indv likelihood", np.log(indv_logLikelihood))

        if(indv_logLikelihood == 0):
            indv_logLikelihood == 0.00000000001
        logLikelihood += np.log(indv_logLikelihood)

    return logLikelihood


def showPlot(N,reducedDataset,P_array):
    # sep = []
    # for k in range(K):
    #     sep.append([])

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    for i in range(N):
        row = reducedDataset[i]
        prob_arr = P_array[i]
        # print(prob)
        probable_val = np.max(prob_arr)
        probable_idx = np.argmax(prob_arr)
        # print("val:",probable_val)
        # print("idx:", probable_idx)
        # sep[probable_idx].append(row)
        if (probable_idx == 0):
            plt.plot(row[0], row[1], 'bo')
        elif (probable_idx == 1):
            plt.plot(row[0], row[1], 'ro')
        if (probable_idx == 2):
            plt.plot(row[0], row[1], 'go')
        if (probable_idx == 3):
            plt.plot(row[0], row[1], 'mo')

    plt.show()
    #plt.pause(0.00001)


    # print("sep",pd.DataFrame(sep[0]))
    # print("sep",pd.DataFrame(sep[1]))
    # print("sep",pd.DataFrame(sep[2]))

#def getGaussian2(xi,mewk,sigmak):



def runEM(reducedDataset, K, N):

    W_array, mew_array, sigma_array, P_array = initialize(K, N)

    # print("W array:", W_array)
    # print("mew array:", mew_array)
    # print("sigma array:", sigma_array)
    # print("P array:", P_array)

    #calculate initial log likelihood
    init_log = calcLogLikelihood(reducedDataset, K, N, mew_array, sigma_array, W_array)
    print("initial log likelihood:", init_log)

    old_log = init_log
    c = 0

    while 1:

        c += 1
        print("Iteration:",c)

        #E step
        #print("P before calc:", P_array)

        for i in range(N):
            for k in range(K):

                xi = reducedDataset[i]
                xi = np.array(xi).reshape(2, 1)

                mewk = mew_array[k]
                sigmak = sigma_array[k]
                wk = W_array[k]

                #print(xi,"\n",mewk,"\n",sigmak,"\n",wk)

                num = wk * getGaussian(xi,mewk,sigmak)
                #print("num:",num)

                den = 0.0
                for newk in range(K):
                    den += W_array[newk] * getGaussian(xi,mew_array[newk],sigma_array[newk])

                #print("den:",den)
                P_array[i][k] = num/den

        #print("P after calc:",P_array)

        #M step
        # print("W before calc:", W_array)
        # print("mew before calc:", mew_array)
        # print("sigma before calc:", sigma_array)

        # updating mew_array
        for k in range(K):
            # first num to add succesive
            x0 = reducedDataset[0]
            x0 = np.array(x0).reshape(2, 1)
            num = np.multiply(P_array[0][k], x0)
            # print("num:",num)

            for i in range(1, N):
                xi = reducedDataset[i]
                xi = np.array(xi).reshape(2, 1)
                num += np.multiply(P_array[i][k], xi)
            # print("num:",num)

            den = 0.0
            for i in range(N):
                den += P_array[i][k]

            updated_mew = np.divide(num, den)
            # print("updated mew k ",updated_mew)
            mew_array[k] = updated_mew


        # updating sigma array
        for k in range(K):
            # first num to add succesive
            x0 = reducedDataset[0]
            x0 = np.array(x0).reshape(2, 1)

            x0minmewk = np.subtract(x0, mew_array[k])
            square = np.dot(x0minmewk, np.transpose(x0minmewk))
            num = np.multiply(P_array[0][k], square)
            # print("num:",num)

            for i in range(1, N):
                xi = reducedDataset[i]
                xi = np.array(xi).reshape(2, 1)

                ximinmewk = np.subtract(xi, mew_array[k])
                square = np.dot(ximinmewk, np.transpose(ximinmewk))
                num += np.multiply(P_array[i][k], square)

            # print("num:", num)

            den = 0.0
            for i in range(N):
                den += P_array[i][k]

            updated_sigma = np.divide(num, den)
            #print("updated sigma k ", updated_sigma)
            sigma_array[k] = updated_sigma


        #updating W_array
        for k in range(K):
            num = 0.0
            for i in range(N):
                num += P_array[i][k]
            den = N
            updated_w = num/den
            #print("updated w k ",updated_w)
            W_array[k] = updated_w

        # print("W after calc:", W_array)
        # print("mew after calc:", mew_array)
        #print("sigma after calc:", sigma_array)

        new_log = calcLogLikelihood(reducedDataset, K, N, mew_array, sigma_array, W_array)
        print("new log likelihood:", new_log)

        diff = new_log - old_log
        print("diff:",diff)

        if(np.absolute(diff) <= tolerance):
            break
            # if(new_log < -2100):
            #     W_array, mew_array, sigma_array, P_array = initialize(K, N)
            # else:
            #     break

        old_log = new_log

        # if(c%3 == 0):
        #     showPlot(N, reducedDataset, P_array)
        #print("updated P array:", P_array)

    print("final P array:",P_array)
    print("final W array:",W_array)
    print("final mew array:", mew_array)
    print("final sigma array:", sigma_array)
    #showPlot(N, reducedDataset, P_array)

    print("cluster portion:")
    for k in range(K):
        sum = 0.0
        for i in range(N):
            sum += P_array[i][k]
        print("Portion of points from cluster ",k,": ",sum)

    #plt.show()
    showPlot(N, reducedDataset, P_array)



def main():

    dataset = pd.read_csv("data_online.txt", delimiter='\t', header=None)
    #print("original dataset",dataset)

    # covar_matrix = calcCov(dataset)
    # print(pd.DataFrame(covar_matrix))

    #dataset -= dataset.mean(axis=0)
    # print(dataset)

    dataset = np.transpose(dataset)
    covar_matrix = np.cov(dataset)

    # print(dataset)
    # print(covar_matrix)
    eigenValues, eigenVectors = np.linalg.eig(covar_matrix)

    # print("before sort")
    # print(eigenValues)
    # print(eigenVectors)
    # print("after sort")
    #
    # idx = eigenValues.argsort()[::-1]
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:, idx]
    # print(eigenValues)
    # print(eigenVectors)

    idx = eigenValues.argsort()[-dim:][::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    #print(eigenValues)
    #print(eigenVectors)

    eigenVectors = np.transpose(eigenVectors)
    #print(eigenVectors)
    reducedDataset = eigenVectors.dot(dataset) #2*100 cross 100*500
    reducedDataset = np.transpose(reducedDataset) #500*2

    #reducedDataset = pd.DataFrame(reducedDataset)
    #print("reduced dataset\n",reducedDataset)

    # for row in reducedDataset:
    #
    #     x = row[0]
    #     y = row[1]
    #     #print(x, " ", y)
    #     plt.plot(x, y,'bo')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.plot(reducedDataset[:,0], reducedDataset[:,1],'mo')
    #plt.pause(0.0001)

    plt.show()

    # for i in range(len(reducedDataset)):
    #     xi = reducedDataset[i]
    #     xi = np.array(xi).reshape(2, 1)
    #     reducedDataset[i] = np.array(xi)
    # print("new:\n",reducedDataset)

    np.random.seed(12)

    K = 4
    N = len(reducedDataset)
    #print(sys.float_info)
    runEM(reducedDataset, K, N)


main()


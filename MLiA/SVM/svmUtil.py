import random
from numpy import *


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    file = open(filename)
    for line in file.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([lineArr[0], lineArr[1]])
        labelMat.append(lineArr[2])
    return dataMat, labelMat


def randomSelectJ(i, m):
    j = i  # 随机返回一个不等于i的J
    while j == i:
        j = random.uniform(0, m)
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def simpleSMO(dataMatIn, classLabels, c, toler, maxIter):
    # 将输入的特征矩阵化，shape为100*2；标签矩阵化后转置，shape为100*1
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    # 初始化b，样本数量m，特征数量n
    b = 0
    m, n = shape(dataMat)
    # 初始化参数α为全0向量 m*1 即100*1
    alpha = mat(zeros(m, 1))
    # 初始化当前遍历次数iter为0 ，当iter == maxIter时，结束循环
    iter = 0
    # while(iter < maxIter):


import numpy as np


def load_data():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage']
    ]
    classVec = [0, 1, 0, 1]
    return postingList, classVec


def createVocabList(postingList):
    vocabSet = set([])
    for posting in postingList:
        vocabSet = vocabSet | set(posting)
    return list(vocabSet)


def setOfWord2Vec(vocalList, input):
    returnVec = [0] * len(vocalList)
    for item in input:
        if item in vocalList:
            returnVec[vocalList.index(item)] = 1
        else:
            print(item + "is not in vocalList!")
    return returnVec


def train(trainMatrix, label):
    docuNum = len(trainMatrix)
    wordsNum = len(trainMatrix[0])
    pAbusive = np.sum(label) / docuNum
    p0Num = np.ones(wordsNum);
    p1Num = np.ones(wordsNum)  # p0,p1分子 初始化为1，避免有一项为0时，相乘结果为0
    p0Denom = 2.0;
    p1Denom = 2.0  # p0,p1 分母，初始化为2（原因未知）
    for i in range(docuNum):
        if label[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive


def classsifyNB(vec2classify, p0Vec, p1Vec, pclass1):
    p1 = np.sum(vec2classify * p1Vec) + np.log(pclass1)


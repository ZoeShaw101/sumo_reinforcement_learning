# using encoding=utf-8

'''
状态空间离散化

对于一天中的每个时刻h，每个信号灯相位，可以采取动作k
'''

import conf
import numpy as np
import math

class Discretizator:

    __mapDiscreteStates = {}
    __stateData = {}  ##原始状态空间矩阵数据

    __dictClusterObjects = {}
    __numClusterObjects = {}


    def init(self):
        for i in range(24):
            self.__stateData[i] = {}
            for j in range(len(conf.actionPhases)):
                self.__stateData[i][conf.actionPhases[j]] = np.array([])

    def learnDiscretization(self, daysToTrain):
        self.init()

        pass

    def getMapDiscreteStates(self):
        return __mapDiscreteStates

    

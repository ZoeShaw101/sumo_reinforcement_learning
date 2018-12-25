# using encoding=utf-8

'''
启动仿真流程

使用SPSA算法优化信号灯固定配时参数，使得目标函数最小
'''
import conf
from RLalgorithm import RL

import os, sys
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:   
     sys.exit("please declare environment variable 'SUMO_HOME'")

import subprocess
import traci
import random
import pandas as pd
import numpy as np
import math
from numpy import random
import numpy.matlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import heapq

import sklearn
from sklearn.cluster import KMeans
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import conf


class Simulator :

    __laneQueueTracker = {}
    __laneWaitingTracker = {}
    __laneNumVehiclesTracker = {}

    __algo = RL()


    def init(self):
        print 'Init simulation status...'
        for lane in conf.listLanes:
            self.__laneNumVehiclesTracker[lane] = 0
            self.__laneWaitingTracker[lane] = 0
            self.__laneQueueTracker[lane] = 0
        pass

    def start(self):
        self.init()

        print 'Start simulation...'
        ###以一天中的时间为步长仿真，首先是固定配时方案
        for day in range(conf.totalDays):
            sumoProcess = subprocess.Popen([conf.sumoBinaryPath, "-c", "../palm.sumocfg", \
            "--remote-port", str(conf.PORT)], stdout=sys.stdout, stderr=sys.stderr)

            traci.init(conf.PORT)

            curSeconds = 0
            curHour = 0
            action = 0
            lastAction = 0
            curPhaseID = 0
            secondsInCurPhase = 0

            while curSeconds < conf.secondsInDay:
                ##如果仍是当前的相位
                if curPhaseID == int(traci.trafficlights.getPhase(conf.SL)) and curSeconds != 0:
                    secondsInCurPhase += 1
                else :
                    secondsInCurPhase = 0
                    curPhaseID = int(traci.trafficlights.getPhase(conf.SL))
                ##刚转到黄灯时搜集环境信息
                if (curPhaseID % 2 == 0) and secondsInCurPhase == 0:
                    if curHour != curSeconds / conf.secondsInHour:
                        curHour = int(curSeconds / conf.secondsInHour)
                    for lane in conf.listLanes:
                        self.__laneQueueTracker[lane] = traci.lane.getLastStepHaltingNumber(str(lane))
                        self.__laneWaitingTracker[lane] = traci.lane.getWaitingTime(str(lane))/60
                        self.__laneNumVehiclesTracker[lane] = traci.lane.getLastStepVehicleNumber(str(lane))
                    curObjValue = self.__algo.computeObjValue(self.__laneQueueTracker, self.__laneQueueTracker)
                    print 'curObjValue=' + str(curObjValue)

                curSeconds += 1

                traci.simulationStep()
            
            traci.close()
        pass

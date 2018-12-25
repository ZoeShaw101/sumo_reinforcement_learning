# using encoding=utf-8

'''
强化算法实现
'''

import conf

class RL:

    def computeObjValue(self, laneQueueTracker, laneWaitingTracker):
        currObjValue = 0
        for key in conf.listLanes:
            currObjValue -= ((1 * laneQueueTracker[key])**1.75 + (2 * laneWaitingTracker[key])**1.75) #TODO - include waitingTracker
        return currObjValue 
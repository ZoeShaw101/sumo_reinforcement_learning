# using encoding=utf-8

'''
一些仿真参数、模型参数
'''
PORT = 8813
sumoBinaryPath = "/Users/shaw/Documents/sumo-0_32_0/bin/sumo"

listLanes = ['8949170_0', '8949170_1', \
			'-164126513_0', '-164126513_1',\
			'52016249_0', '52016249_1',\
			'-164126511_0', '-164126511_1']

listEdges = ['8949170', '-164126513', '52016249', '-164126511']
tupEdges = ('8949170', '-164126513', '52016249', '-164126511')

actionPhases = [0,2,4,6]

# pick the thresholds from small, medium, long-sized queues
numPhasesForAction = 4 # 8 including the yellow phases
numEdges = 4
numLanes = 8
numQueueSizeBuckets = 3
numwaitingBuckets = 3

secondsInDay = 24*60*60
secondsInHour = 60*60
totalDays = 1 # days to run simulation

alpha = 0.5 # learning rate
SL = "65546898" # 当前这个交叉口信号灯的id

# counters
currSod = 0
currPhaseID = 0
secsThisPhase = 0

# state objects and boolean helpers
phaseNum = 0
lastObjValue = 0
lastAction = 0
stepThru = 1
arrivalTracker = 0
waitingTime = 0
currState = 0
lastState = 0

# discretization parameters
numPhasesForAction = 4 # 8 including the yellow phases
numEdges = 4
numLanes = 8
numQueueSizeBuckets = 4
numwaitingBuckets = 4
hoursInDay = 24 #
numActions = 2 # 1 = switch to yellow phase; stay in current phase
secsPerInterval = 4
minPhaseTime = 4
maxPhaseTime = 36
yellowPhaseTime = 4

numStates = numPhasesForAction*(numQueueSizeBuckets*numwaitingBuckets)**numEdges

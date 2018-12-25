# Python接口



# Java接口
注意，Java接口的client TraCI4J和TraaS的版本较低，在sumo最新的1.0.1版本时更改了traci协议，所以使用java 接口，应该重新编译低版本（0.32.0）的sumo

## it.polito.appeal.traci 包： 
### SumoTraciConnection类
Models a TCP/IP connection to a local or remote SUMO server via the TraCI protocol.
初始化：可以通过(sumo_bin_path, config_file)初始化，也可以通过SocketAddress初始化


## de.tudresden.sumo.cmd 包 : 一些控制命令
### 车辆控制类 Vehicle 
- 控制仿真过程中车辆行为，可以动态添加车辆
- 获得仿真过程中车辆信息，比如累积等待时长，已经行驶的距离等

### 道路控制类 Lane
- 得到仿真过程中道路信息，比如车道上到上一仿真时刻停顿的车辆数
- 可以动态改变车道最大车速

### 行人控制类 Person
- 支持动态添加行人，并制定行人的行为及行为参数：驾驶、等待、步行

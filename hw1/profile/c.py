import matplotlib.pyplot as plt
import numpy as np

# 进程数量
process_count = list(range(1, 13))

# 每个进程数量对应的运行时间（示例数据，需要替换成实际数据）
cpu_time = [28.1, 15.65, 11.2, 9.225, 7.56, 6.6, 6, 5.99, 5.8, 5.7, 5.8, 5.925]
# io_time = [27.7, 14.12, 13, 9.19, 12.5,
#            12.84, 11.56, 11.7, 11.95, 12.5, 10.8, 11.8]
# communication_time = [0.00002, 0.708, 4.538,
#                       1.13, 1.38, 1.59, 1.2, 0.78, 0.65, 0.67, 0.668, 0.575]
Allreduce = [2.29e-05, 0.29256509999999997, 1.4865, 1.3343699999999998, 0.9561040000000001,
             0.7579725, 0.9645714285714285, 1.198875, 0.7373999999999999, 1.1396, 0.7929727272727274, 0.9468]
Sendrecv = [0.0, 1.0365, 1.982, 1.054, 1.5176, 0.9241666666666667, 1.2045,
            1.1003999999999998, 1.1131333333333333, 1.42877, 1.6693727272727275, 1.4596]
io_time = [27.668537, 14.1352474, 13.190148666666666, 9.19943025, 12.51660854, 12.347942016666662, 10.894484242857143,
           11.711320275000002, 11.935491555555558, 12.521248200000002, 11.04275809090909, 11.982606933333324]
# communication time = Allreduce + Sendrecv
communication_time = []
for i in range(len(Allreduce)):
    communication_time.append(Allreduce[i] + Sendrecv[i])


# 创建子图
fig, ax = plt.subplots()

# 使用柔和的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 绘制堆叠条形图
# plt.bar(process_count, cpu_time, label='CPU Time', color=colors[0])
# plt.bar(process_count, io_time, bottom=cpu_time,
#         label='I/O Time', color=colors[1])
# plt.bar(process_count, communication_time, bottom=np.array(cpu_time) +
#         np.array(io_time), label='Communication Time', color=colors[2])

plt.bar(process_count, communication_time,
        label='Communication Time', color=colors[2])
plt.bar(process_count, io_time, bottom=communication_time,
        label='I/O Time', color=colors[1])
plt.bar(process_count, cpu_time, bottom=np.array(communication_time) +
        np.array(io_time), label='CPU Time', color=colors[0])

plt.title("Time Profile")

# 设置X轴标签
plt.xlabel('# of Process')

# 设置Y轴标签
plt.ylabel('Time')

# 添加图例
plt.legend()

# 显示图
plt.show()

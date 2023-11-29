import matplotlib.pyplot as plt
import numpy as np

# 进程数量
process_count = list(range(2, 13))

# 每个进程数量对应的运行时间（示例数据，需要替换成实际数据）
cpu_time = []
total_time = [31.1, 25.4, 24.3, 19.5, 17.5,
              19.8, 18.25, 19.8, 18.6, 18.8, 15.6]
# io_time = [27.7, 14.12, 13, 9.19, 12.5,
#            12.84, 11.56, 11.7, 11.95, 12.5, 10.8, 11.8]
# communication_time = [0.00002, 0.708, 4.538,
#                       1.13, 1.38, 1.59, 1.2, 0.78, 0.65, 0.67, 0.668, 0.575]
Allreduce = [0.289929, 0.9092000000000001, 1.4310729, 0.7528, 0.8338333333333333, 0.8390714285714286,
             1.1045999999999998, 0.9721333333333334, 1.1239199999999998, 0.8722636363636364, 1.2500666666666667]
Sendrecv = [1.6125, 1.827, 1.018, 1.4968, 1.2785, 1.2650714285714286, 1.1783249999999998,
            1.3155333333333334, 1.38663, 1.6268636363636364, 1.5140666666666671]
io_time = [13.736587649999999, 11.990238999999999, 13.140714575000002, 9.78128616, 8.740652500000001,
           11.722450285714286, 10.07759325, 11.817990444444444, 10.520375500000002, 10.749747999999999, 7.161848166666668]

# communication time = Allreduce + Sendrecv
communication_time = []
for i in range(len(Allreduce)):
    communication_time.append(Allreduce[i] + Sendrecv[i])

for i in range(len(total_time)):
    cpu_time.append(total_time[i] - io_time[i] - communication_time[i])

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
        label='Communication time', color=colors[2])
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

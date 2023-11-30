import matplotlib.pyplot as plt
import numpy as np

# 进程数量
process_count = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 每个进程数量对应的运行时间（示例数据，需要替换成实际数据）
cpu_time = [12.4, 7.68, 6.05, 6.01, 5.925, 5.84, 5.51, 5.55, 5.025, 5.04]
# io_time = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
# Sendrecv = [0.24, 0.212, 0.277, 0.283,
#             0.339, 0.412, 0.441, 0.474, 0.533, 0.594]
# Allreduce = [0.365, 0.308, 0.341, 0.348,
#              0.269, 0.248, 0.215, 0.219, 0.266, 0.290]
Allreduce = [4.306666666666667, 2.3145499999999997, 1.6711, 1.18455, 0.9468,
             0.7954333333333334, 0.7036679999999998, 0.7672583333333334, 0.88578, 0.9469541666666665]
Sendrecv = [1.9165333333333334, 1.5428916666666663, 1.4591, 1.3581000000000003, 1.4596,
            1.562983333333333, 1.7208333333333334, 1.911, 2.995416666666667, 3.3366666666666664]
IO_ = [17.997400575, 10.257974333333337, 12.01795665, 19.765427241666664, 11.982606933333324,
       10.104087283333332, 10.285497816666672, 12.495476958333331, 10.377523324999997, 12.494577]
iteration = [32, 17, 12, 9, 8, 7, 7, 7, 7, 7]

# 创建子图
fig, ax = plt.subplots()

# 使用柔和的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 绘制堆叠条形图

plt.bar(process_count, Sendrecv, label='Sendrecv', color=colors[0], width=3)
plt.bar(process_count, Allreduce, bottom=Sendrecv,
        label='Allreduce', color=colors[1], width=3)
plt.bar(process_count, cpu_time, bottom=np.array(Sendrecv) +
        np.array(Allreduce), label='CPU Time', color=colors[2], width=3)

# plt.bar(process_count, cpu_time, label='CPU Time', color=colors[2], width=3)
# plt.bar(process_count, Sendrecv, bottom=cpu_time,
#         label='Sendrecv', color=colors[0], width=3)
# plt.bar(process_count, Allreduce, bottom=np.array(Sendrecv) +
#         np.array(cpu_time), label='Allreduce', color=colors[1], width=3)

# thicker bar


# iteration 用折線圖

# plt.plot(process_count, iteration, color='#d62728',
# marker='o', label='Iteration')

plt.title('Time Profile for different send size \n (1 node 12 process)')

# 设置X轴标签
plt.xlabel('Percentage of task size')

# 设置Y轴标签
plt.ylabel('Time')

# 添加图例
plt.legend()

# 显示图
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 进程数量
process_count = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 每个进程数量对应的运行时间（示例数据，需要替换成实际数据）
totoal_time = [16, 24.6, 17.9, 18.5, 15, 21, 18.3, 18.3, 18.2, 15.5]
# io_time = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
# Sendrecv = [0.24, 0.212, 0.277, 0.283,
#             0.339, 0.412, 0.441, 0.474, 0.533, 0.594]
# Allreduce = [0.365, 0.308, 0.341, 0.348,
#              0.269, 0.248, 0.215, 0.219, 0.266, 0.290]
Allreduce = [0.6432916666666666, 0.8111666666666667, 1.0041666666666667, 1.0508333333333333, 1.0896666666666666,
             1.1108333333333333, 0.9760416666666667, 1.1241666666666668, 1.0913333333333333, 1.0385833333333334]
Sendrecv = [0.9771666666666666, 1.0654166666666665, 1.269875,
            1.4645, 1.59425, 1.743, 1.875375, 2.11625, 2.09625, 2.255]
IO_ = [11.510212374999998, 20.526123866666662, 13.420734499999996, 13.907940083333337, 10.14402844166667,
       15.967959775, 13.373679149999996, 12.964085166666667, 12.840450124999998, 10.159711433333335]
cpu_time = []

for i in range(len(totoal_time)):
    cpu_time = totoal_time[i] - IO_[i] - Allreduce[i] - Sendrecv[i]

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

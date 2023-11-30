import matplotlib.pyplot as plt
import numpy as np

# partion form 10% to 100%
iteration = [32, 17, 12, 9, 8, 7, 7, 7, 7, 7]
send_unit_data = [1.79e+07, 3.579e+07, 5.369e+07, 7.158e+07,
                  8.948e+07, 1.074e+08, 1.253e+08, 1.432e+08, 1.611e+08, 1.79e+08]
send_total_data = [5.727e+08, 6.085e+08, 6.442e+08, 6.442e+08,
                   7.158e+08, 7.516e+08, 8.769e+08, 1.002e+09, 1.127e+09, 1.253e+09]

fig, ax1 = plt.subplots()

# 设置x轴刻度为1到5
x_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 绘制迭代折线图，使用淺藍色线条
ax1.plot(x_values, iteration, marker='o',
         color='#00808c', label='Iteration')

# 设置第一个y轴标签
ax1.set_xlabel('Percentage of task size')
ax1.set_ylabel('Iterations', color='black')
ax1.tick_params('y', colors='black')

# 创建第二个y轴，与第一个y轴共享x轴
ax2 = ax1.twinx()

# 绘制发送大小和发送总大小的条形图，使用红色和绿色条形
width = 3  # 条形宽度
ax2.bar(x_values, send_unit_data, width=width,
        color='r', alpha=0.7, label='Send Size')
ax2.bar([i + width for i in x_values], send_total_data,
        width=width, color='g', alpha=0.5, label='Send Total Size')

# 设置第二个y轴标签 lighter orange
ax2.set_ylabel('Send Size Data and Send Total Data', color='black')
ax2.tick_params('y', colors='black')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')


# 显示图形
plt.title(
    "number of process vs Iteration, Send Size per task, Send Total Size (1 node)")
plt.show()


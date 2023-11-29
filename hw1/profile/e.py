import matplotlib.pyplot as plt

# 假设你有迭代数据和发送大小数据，分别存储在 iteration_data 和 send_size_data 中
iteration_data = [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]
send_unit_data = [5.369e+08, 3.579e+08, 2.684e+08, 2.147e+08,
                  1.79e+08, 1.534e+08, 1.342e+08, 1.193e+08, 1.074e+08, 9.761e+07, 8.948e+07]
send_total_data = [1.611e+09, 1.074e+09, 1.074e+09, 8.59e+08,
                   8.948e+08, 7.67e+08, 8.053e+08, 7.158e+08, 7.516e+08, 6.833e+08, 7.158e+08]

# 创建一个新的图形
fig, ax1 = plt.subplots()

# 设置x轴刻度为1到5
x_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# 绘制迭代折线图，使用淺藍色线条
ax1.plot(x_values, iteration_data, marker='o',
         color='#00808c', label='Iteration')

# 设置第一个y轴标签
ax1.set_xlabel('number of process')
ax1.set_ylabel('Iterations', color='black')
ax1.tick_params('y', colors='black')

# 创建第二个y轴，与第一个y轴共享x轴
ax2 = ax1.twinx()

# 绘制发送大小和发送总大小的条形图，使用红色和绿色条形
width = 0.3  # 条形宽度
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


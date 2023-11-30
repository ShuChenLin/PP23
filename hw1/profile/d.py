import matplotlib.pyplot as plt
import numpy as np

# 进程数量
process_count = list(range(1, 13))

# 每个进程数量对应的运行时间（示例数据，需要替换成实际数据）
runtime_1_node = [55.8, 31.1, 27.9, 20.8, 22.6,
                  20.6, 19.8, 20, 19.5, 20.7, 19.3, 20.3]  # 1 节点
runtime_2_nodes = [0, 31.1, 25.4, 24.3, 22.3, 20.9,
                   19.8, 19.9, 20.0, 19.3, 20.3, 19.5]  # 2 节点
runtime_3_nodes = [0, 0, 26.5, 27.7, 23, 21.2,
                   20.4, 19.7, 20.5, 19.5, 18.8, 18.4]  # 3 节点

# 条形图的宽度
bar_width = 0.2

# X轴偏移量
x1 = np.arange(len(process_count))
x2 = [x + bar_width for x in x1]
x3 = [x + bar_width for x in x2]

# 创建子图
fig, ax = plt.subplots()

# plot speedup

speedup = []
speedup2 = []
speedup3 = []

for i in range(len(runtime_1_node)):
    speedup.append(runtime_1_node[0] / runtime_1_node[i])

for i in range(1, len(runtime_2_nodes)):
    speedup2.append(runtime_1_node[0] / runtime_2_nodes[i])

for i in range(2, len(runtime_3_nodes)):
    speedup3.append(runtime_1_node[0] / runtime_3_nodes[i])

# 繪製折線圖
# 加入點點
# 顏色分開一點也柔合一點

plt.plot(process_count, speedup, color='#1f77b4', marker='o', label='1 Node')
plt.plot(process_count[1:], speedup2,
         color='#ff7f0e', marker='o', label='2 Nodes')
plt.plot(process_count[2:], speedup3,
         color='#2ca02c', marker='o', label='3 Nodes')

plt.title('Speedup factor')

# 设置X轴标签
plt.xlabel('# of Process')

# 设置Y轴标签
plt.ylabel('Speedup')


# 添加图例
plt.legend()

# 显示图
plt.show()


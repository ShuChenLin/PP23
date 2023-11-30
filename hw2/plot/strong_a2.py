import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Data from your runs
num_processes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
cpu_times = [
    [194.168312],                          # -c1
    [97.098506, 97.072520],                # -c2
    [64.743315, 64.722035, 64.702095],     # -c3
    [48.567735, 48.552245, 48.537422, 48.520734],  # -c4
    [38.877218, 38.851834, 38.838971, 38.828455, 38.804310],  # -c5
    [32.392532, 32.383388, 32.383387, 32.376598, 32.351287, 32.341457],  # -c6
    [27.784916, 27.780086, 27.769545, 27.749170, 27.742342, 27.728567, 27.726540],  # -c7
    [24.303141, 24.311560, 24.284231, 24.275564, 24.252606, 24.243571, 24.325067, 24.275564],  # -c8
    [21.611230, 21.615282, 21.600744, 21.580526, 21.573912, 21.558282, 21.566306, 21.629577, 21.600890],  # -c9
    [19.457816, 19.449731, 19.436180, 19.429774, 19.427544, 19.427075, 19.410207, 19.414358, 19.475038, 19.407501],  # -c10
    [17.689270, 17.678714, 17.677217, 17.675141, 17.670170, 17.661899, 17.653463, 17.632706, 17.620347, 17.648375, 17.678714],  # -c11
    [16.219847, 16.234525, 16.213605, 16.208723, 16.198250, 16.195740, 16.188080, 16.181759, 16.180157, 16.171197, 16.171093, 16.161376]  # -c12
]


# another data

cpu_times1 = [40.323318, 21.320104, 10.187260, 9.717784, 8.326441, 7.793288, 6.485945, 5.973920, 5.409384, 5.290850, 4.324753, 4.140656]

cpu_times2 = [55.894079, 29.055430, 18.776218, 15.594615, 13.000980, 11.159160, 9.571601, 8.936366, 8.140131, 7.596275, 7.090059, 6.633217]


# plot the data, every row are independent runs and each row with different colors, make sure the colors are soft
# not finding speedup
# i hope each row has different color

# colormap = plt.cm.get_cmap('viridis', len(cpu_times))

# # Plotting
# plt.figure(figsize=(10, 6))

# for i, times in enumerate(cpu_times):
#     color = colormap(i / len(cpu_times))  # Get a color from the colormap
#     plt.plot(num_processes[:len(times)], times, marker='o', linestyle='-', color=color, label=f'Run {i+1}')


# plt.title('CPU Time')
# plt.xlabel('Processes ID (rank)')
# plt.ylabel('CPU Time')
# plt.grid(True)
# plt.show()

# # Using a colormap to get different colors for each run
# colormap = plt.cm.get_cmap('viridis', len(cpu_times))

# # Plotting as Bar Chart
# plt.figure(figsize=(10, 6))

# for i, times in enumerate(cpu_times):
#     color = colormap(i / len(cpu_times))  # Get a color from the colormap
#     plt.bar(np.array(num_processes[:len(times)]) + i * 0.2, times, width=0.2, color=color, label=f'Run {i+1}')

# plt.title('CPU Time')
# plt.xlabel('Processes ID (rank)')
# plt.ylabel('CPU Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# colors = plt.cm.Paired(np.linspace(0, 1, len(cpu_times)))
# custom_cmap = ListedColormap(colors)

# # Plotting as Bar Chart
# plt.figure(figsize=(10, 6))

# for i, times in enumerate(cpu_times):
#     plt.bar(np.array(num_processes[:len(times)]) + i * 0.2, times, width=0.2, color=custom_cmap(i), label=f'{i+1} threads')

# plt.title('Load Balancing (pthread)')
# plt.xlabel('thread ID')
# plt.ylabel('Execution Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# plot the strong scalability with the largest number in every row

largest_times = [max(times) for times in cpu_times]

# plot two bars for each process num

plt.figure(figsize=(10, 6))

plt.bar(np.array(num_processes) - 0.2, cpu_times2, width=0.4, color='#ff7f0e', label='without vectorization')
plt.bar(np.array(num_processes) + 0.2, cpu_times1, width=0.4, color='#1f77b4', label='with vectorization')

plt.title('Strong Scalability (pthread)')
plt.xlabel('thread number')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting as Bar Chart

# plt.figure(figsize=(10, 6))

# plt.bar(num_processes, largest_times, width=0.5, color='#1f77b4')

# plt.title('Strong Scalability')
# plt.xlabel('Processes number')
# plt.ylabel('Execution Time')
# plt.grid(True)
# plt.show()


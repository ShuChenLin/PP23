import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Data from your runs
num_processes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
cpu_times = [
    [69.845574],
    [34.970887, 34.969388],
    [23.341108, 23.337564, 23.337546],
    [17.529940, 17.525818, 17.524846, 17.524626],
    [14.050098, 14.033371, 14.030341, 14.031748, 14.080243],
    [11.715990, 11.687670, 11.687438, 11.686862, 11.820355, 11.821146],
    [10.055466, 10.020472, 10.019043, 10.019483, 10.189604, 10.193300, 10.189360],
    [8.807127, 8.766316, 8.765613, 8.765511, 8.947718, 8.952905, 8.947603, 8.952528],
    [7.848864, 7.820547, 7.812104, 7.819924, 7.818986, 7.822022, 7.818980, 7.821893, 7.819980],
    [7.067939, 7.020033, 7.020488, 7.019668, 7.070003, 7.075365, 7.069918, 7.075378, 7.092327, 7.092326],
    [6.434356, 6.290777, 6.390594, 6.390337, 6.434125, 6.434113, 6.434122, 6.434113, 6.530814, 6.513809, 6.529893],
    [5.956162, 5.852572, 5.851614, 5.851515, 5.951251, 5.850749, 5.850696, 5.849407, 5.847554, 5.842638, 5.842606, 5.8181252]
]

# another data

cpu_times2 = [104.701702, 52.399600, 34.974635, 26.252863, 21.024298, 17.536132, 15.152035, 13.394071, 11.722589, 10.617756, 9.668921, 8.850054]

cpu_times3 = [147.825532, 73.974682, 49.529703, 37.040387, 29.757575, 24.830132, 21.245421, 18.569052, 16.526800, 14.898523, 13.590691, 12.565321]


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
#     plt.bar(np.array(num_processes[:len(times)]) + i * 0.2, times, width=0.2, color=custom_cmap(i), label=f'{i+1} processes')

# plt.title('Load Balancing (hybrid)')
# plt.xlabel('Processes ID (rank)')
# plt.ylabel('Execution Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# plot the strong scalability with the largest number in every row

largest_times = [max(times) for times in cpu_times]

# # plot two bars for each process num

plt.figure(figsize=(10, 6))

plt.bar(np.array(num_processes) - 0.2, cpu_times3, width=0.4, color='#ff7f0e', label='without vectorization')
plt.bar(np.array(num_processes) + 0.2, largest_times, width=0.4, color='#1f77b4', label='with vectorization')

plt.title('Strong Scalability')
plt.xlabel('Processes number')
plt.ylabel('Execution Time')
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


import matplotlib.pyplot as plt

# Data from your runs

num_processes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

gather = [0.0754, 0.0272, 0.0290, 0.0304, 0.0217, 0.0213, 0.0219, 0.0215, 0.0219, 0.0215, 0.0219, 0.0245]

# plot the bars

plt.figure(figsize=(10, 6))

plt.bar(num_processes, gather, width=0.5, color='#1f77b4')

plt.title('MPI_Gather Time')
plt.xlabel('Processes ID (rank)')
plt.ylabel('Execution Time (s)')
plt.grid(True)
plt.show()
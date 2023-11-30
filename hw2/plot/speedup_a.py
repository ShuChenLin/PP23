import matplotlib.pyplot as plt

# Data from your runs
num_processes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
total_time = [196.746141, 99.730393, 67.352201, 51.140611, 41.544086, 34.970187, 30.428692, 26.326411, 24.326411, 22.122273, 20.305331, 18.902516]



# Calculate speedup
for i in range(1, len(total_time)):
    total_time[i] = total_time[0] / total_time[i]

total_time[0] = 1

ideal_speedup = num_processes

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_processes, total_time, marker='o', linestyle='-', alpha=0.5, color='#1f77b4', label='1 process')

plt.plot(num_processes, ideal_speedup, marker='o', linestyle='-', alpha=1, color='#d62728', label='Ideal Speedup')


# plt.plot(num_processes, ideal_speedup, marker='o', linestyle='-', alpha=1, color='#d62728', label='Ideal Speedup')


plt.title('Speedup Factor')
plt.xlabel('Number of threads')
plt.ylabel('Speedup')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Data from your runs
num_processes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
total_time = [72.423049, 37.536633, 25.916324, 20.094225, 16.790881, 14.289587, 12.618107, 11.367447, 10.455602, 9.690616, 8.993552,8.487447]

cpu_times2 = [107.300872, 50.511627, 37.307072, 28.826871, 23.598278, 20.115822, 17.636118, 15.773798, 14.288694, 13.125150, 12.193178, 11.395369]

cross_time = []

cross_time.append(cpu_times2[0] / total_time[0])

# Calculate speedup
for i in range(1, len(total_time)):
    cross_time.append(cpu_times2[i] / total_time[i])
    total_time[i] = total_time[0] / total_time[i]
    cpu_times2[i] = cpu_times2[0] / cpu_times2[i]

total_time[0] = 1
cpu_times2[0] = 1

ideal_speedup = num_processes

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_processes, total_time, marker='o', linestyle='-', alpha=0.5, color='#1f77b4', label='3 CPU per process')

# plot cpu_times2

plt.plot(num_processes, cpu_times2, marker='o', linestyle='-', alpha=0.5, color='#ff7f0e', label='2 CPU per process')

plt.plot(num_processes, cross_time, marker='o', linestyle='-', alpha=1, color='#2ca02c', label='Speedup between 2 and 3 CPU per process')

ideal_speedup = num_processes

plt.plot(num_processes, ideal_speedup, marker='o', linestyle='-', alpha=1, color='#d62728', label='Ideal Speedup')

# plt.plot(num_processes, ideal_speedup, marker='o', linestyle='-', alpha=1, color='#d62728', label='Ideal Speedup')


plt.title('Speedup Factor')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.legend()
plt.grid(True)
plt.show()

import re

# Define regular expressions to match Callsite Time statistics lines
callsite_stats_start = re.compile(
    r"@--- Callsite Time statistics \(all, milliseconds\): \d+ ---")
callsite_stats_end = re.compile(
    r"@--- Callsite Message Sent statistics \(all, sent bytes\) ---")

# Initialize a flag to indicate when to start and stop capturing data
capture_data = False

# Initialize a list to store the Callsite Time statistics
callsite_time_stats = []

# size = 12 in float type
size = 12.0 * 1000.0
COUNT = 0.0 * 0.0

# Read the MPIP file
with open('PARTION_PROFILE/test33_10.mpiP', 'r') as f:
    current_callsite = None
    first_row = True
    for line in f:
        # Check if the line matches the start pattern
        if callsite_stats_start.match(line):
            capture_data = True
            continue

        # Check if the line matches the end pattern
        if callsite_stats_end.match(line):
            capture_data = False
            break

        if capture_data:
            # Process the lines containing Callsite Time statistics
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 9:
                    name, site, rank, count, max_time, mean_time, min_time, app_percentage, mpi_percentage = parts
                    if (name == 'Name' or rank == "*"):
                        continue
                    elif (name == "Allreduce"):
                        COUNT = float(count)
                    callsite_time_stats.append({
                        "Name": name,
                        "Site": site,
                        "Rank": rank,
                        "Count": count,
                        "Max Time": max_time,
                        "Mean Time": float(mean_time),
                        "Min Time": min_time,
                        "App Percentage": app_percentage,
                        "MPI Percentage": mpi_percentage
                    })

# delete callsite_time_stats[0]


# Define the callsites you want to sum the Mean Time for
Allreduce_to_sum = ["Allreduce"]
Sendrecv_to_sum = ["Sendrecv"]
IO_to_sum = ["File_read_at", "File_write_at", "File_open", "File_close"]

# Initialize a dictionary to store the total Mean Time for each callsite
total_mean_time_Allreduce = {}
total_mean_time_Sendrecv = {}
total_mean_time_IO = {}

# Sum up the Mean Time for the specified callsites
for callsite in Allreduce_to_sum:
    total_mean_time_Allreduce[callsite] = sum(
        entry["Mean Time"] for entry in callsite_time_stats if entry["Name"] == "Allreduce")

for callsite in Sendrecv_to_sum:
    total_mean_time_Sendrecv[callsite] = sum(
        entry["Mean Time"] for entry in callsite_time_stats if entry["Name"] == "Sendrecv")

for callsite in IO_to_sum:
    total_mean_time_IO["IO"] = sum(
        entry["Mean Time"] for entry in callsite_time_stats if entry["Name"] == "File_read_at" or entry["Name"] == "File_write_at" or entry["Name"] == "File_open" or entry["Name"] == "File_close")

# Read the existing arrays from the file
with open('output2.txt', 'r') as file:
    exec(file.read())

print(COUNT, "is count")

# multiple count to mean time
# size /= COUNT

# Append your answer to the respective arrays
Allreduce.append(total_mean_time_Allreduce["Allreduce"] * COUNT / size)
Sendrecv.append(total_mean_time_Sendrecv["Sendrecv"] * COUNT / size)
IO_.append(total_mean_time_IO["IO"] / size)

# Write the updated arrays back to the file
with open('output2.txt', 'w') as file:
    file.write(f'Allreduce = {Allreduce}\n')
    file.write(f'Sendrecv = {Sendrecv}\n')
    file.write(f'IO_ = {IO_}\n')

import subprocess
import time
import os

# Define executable paths
BASE_EXECUTABLE = "./apm_base"
PARALLEL_EXECUTABLE = "./apm_parallel"

# Define test parameters
APPROX_FACTOR = "25"  # Example approximation factor
TEXT_FILE = "../dna/big_chrY.fa"  # Change this to your test file
PATTERNS = ["CCAGTTCCCTTCTGGAATTTAGGGGCCCTGGGACAGCCCTGTACATGAGC", "CATACCGATAACAACCACGAGCTAGTAAGCGCCGTCGCGCCAATAAATCT", "TATGCCACATGCCCGGAATTAGGTCTGTTACTCGTAGCAAACGTATGCGG"]  # Example patterns
NUM_PROCESSES = 6  # Number of MPI processes for the optimized version
NUM_THREADS = 6  # Number of OpenMP threads for the optimized version

def run_command(command):
    """Runs a shell command and returns the output and execution time."""
    start_time = time.time()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    end_time = time.time()
    
    return result.stdout, result.stderr, end_time - start_time

def parse_results(output):
    """Parses the number of matches from program output."""
    results = {}
    for line in output.split("\n"):
        if "Number of matches for pattern" in line:
            parts = line.split("<")
            pattern = parts[1].split(">")[0]
            count = int(parts[1].split(":")[1].strip())
            results[pattern] = count
    return results

# Build command for base implementation
base_command = f"{BASE_EXECUTABLE} {APPROX_FACTOR} {TEXT_FILE} " + " ".join(PATTERNS)

# Build command for optimized implementation (MPI + OpenMP)
optimized_command = f"OMP_NUM_THREADS={NUM_THREADS} salloc -N 1 -n {NUM_PROCESSES} mpirun {PARALLEL_EXECUTABLE} {APPROX_FACTOR} {TEXT_FILE} " + " ".join(PATTERNS)

print("\nRunning Base Implementation...")
base_output, base_error, base_time = run_command(base_command)

print("\nRunning Optimized Implementation...")
optimized_output, optimized_error, optimized_time = run_command(optimized_command)

# Parse results
base_results = parse_results(base_output)
optimized_results = parse_results(optimized_output)

# Compare results
match = base_results == optimized_results

# Print comparison summary
print("\n===================================")
print("          COMPARISON RESULTS       ")
print("===================================\n")

print(f"Base Implementation Time: {base_time:.4f} seconds")
print(f"Optimized Implementation Time: {optimized_time:.4f} seconds\n")

if match:
    print("✅ Results Match! The optimization is correct.")
else:
    print("❌ Mismatch Detected! There might be a bug in the optimized implementation.")

print("\n-----------------------------------")
print("| Pattern | Base Matches | Optimized Matches |")
print("-----------------------------------")
for pattern in PATTERNS:
    base_match = base_results.get(pattern, "N/A")
    opt_match = optimized_results.get(pattern, "N/A")
    print(f"| {pattern} | {base_match:<12} | {opt_match:<17} |")
print("-----------------------------------")

print("\nSpeedup Factor:", round(base_time / optimized_time, 2) if optimized_time else "N/A")

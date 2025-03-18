import random

def generate_mock_dna(length, rng):
    """Generates a random DNA sequence of the given length using a controlled random generator."""
    bases = ['A', 'T', 'C', 'G']
    return ''.join(rng.choices(bases, k=length))

def save_mock_data(filename, num_lines, line_length, seed=None):
    """Generates and saves mock DNA sequences to a file with controlled randomness."""
    rng = random.Random(seed)  # Create a random generator instance with an optional seed

    with open(filename, 'w') as f:
        for _ in range(num_lines):
            f.write(generate_mock_dna(line_length, rng) + '\n')

if __name__ == "__main__":
    filename = "big_chrY.fa"
    num_lines = 10_000_000  # Number of lines in the file
    line_length = 50  # Length of each DNA sequence per line
    seed = 42  # Change or set to None for true randomness

    save_mock_data(filename, num_lines, line_length, seed)
    print(f"Mock DNA data saved to {filename} with seed {seed}")


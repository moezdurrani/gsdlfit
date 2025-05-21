import numpy as np
import random
import argparse
import os
import csv

def kronecker_power(matrix, n):
    result = np.array(matrix)
    for _ in range(1, n):
        result = np.kron(result, matrix)
    return result

def mask_generated_matrix(matrix):
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        for j in range(matrix.shape[1]):
            random_value = random.uniform(0, 1)
            matrix[i, j] = 1 if matrix[i, j] > random_value else 0
    return matrix

def matrix_edge_list(matrix, filename):
    with open(filename, "w") as f:
        f.write("# Synthetic Kronecker Graph \n")
        f.write("# Edges are directed \n")
        for src in range(len(matrix)):
            for dst in range(len(matrix[src])):
                if matrix[src][dst] == 1:
                    f.write(f"{src} {dst}\n")

def create_graphs(init_matrix, n, samples_per_k, output_dir):
    # create a csv file to store metadata
    with open(os.path.join(output_dir, "metadata.csv"), "w", newline="") as meta_file:
        writer = csv.writer(meta_file)
        writer.writerow(["filename", "k_power", "sample_number"])

        for k in range(1, n + 1):
            for s in range(samples_per_k):
                result = kronecker_power(init_matrix, k)
                result = mask_generated_matrix(result)
                filename = f"graph_kron_power{k}_sample{s}.txt"
                filepath = os.path.join(output_dir, filename)
                matrix_edge_list(result, filepath)
                writer.writerow([filename, k, s])
                print(f"saved : {filename}")
    print(f"All Edges and Metadata saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("init_matrix", nargs=4, type=float, help="Four parameters for the initiator matrix separated with space")
    parser.add_argument("num_of_graphs", type=int, help="(range of k values) Number of iterations for fitting the model")
    parser.add_argument("samples_per_k", type=int, help="Number of graphs being generated per k value")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the edge list files")
    args = parser.parse_args()

    init_matrix = np.array(args.init_matrix).reshape(2, 2)
    create_graphs(init_matrix, args.num_of_graphs, args.samples_per_k, args.output_dir)

if __name__ == "__main__":
    main()

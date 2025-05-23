import argparse
import networkx as nx
import numpy as np
import random
import math
import pandas as pd
import os
import json


def kron_product(theta, k):
    result = theta.copy()
    for _ in range(1, k):
        result = np.kron(result, theta)
    return result


def generate_kronecker_graph(theta, k):
    N1 = theta.shape[0]
    mu = np.sum(theta)**k
    sigma2 = mu - np.sum(theta**2)**k
    num_edges = int(np.random.normal(mu, math.sqrt(sigma2)))
    kron_size = N1 ** k

    G = nx.DiGraph()
    G.add_nodes_from(range(kron_size))

    for _ in range(num_edges):
        u = v = 0
        for _ in range(k):
            idx = np.random.choice(N1**2, p=theta.flatten()/theta.sum())
            row, col = divmod(idx, N1)
            u = u * N1 + row
            v = v * N1 + col
        G.add_edge(u, v)
    return G


def metropolis_sample_permutation(G, theta_kron, num_iter):
    nodes = list(G.nodes())
    N = len(nodes)
    sigma = list(range(N))
    for _ in range(num_iter):
        j, k = random.sample(range(N), 2)
        sigma[j], sigma[k] = sigma[k], sigma[j]
        p_old = edge_likelihood(G, theta_kron, sigma)
        sigma[j], sigma[k] = sigma[k], sigma[j]
        p_new = edge_likelihood(G, theta_kron, sigma)
        acceptance_ratio = min(1, p_new / p_old)
        if random.random() < acceptance_ratio:
            sigma[j], sigma[k] = sigma[k], sigma[j]
    return sigma


def edge_likelihood(G, theta_kron, sigma):
    score = 1.0
    for (u, v) in G.edges():
        su, sv = sigma[u], sigma[v]
        score *= theta_kron[su, sv]
    for u in G.nodes():
        for v in G.nodes():
            if (u, v) not in G.edges():
                su, sv = sigma[u], sigma[v]
                score *= (1 - theta_kron[su, sv])
    return score


def compute_gradient(G, theta, k, num_samples):
    theta_kron = kron_product(theta, k)
    grad = np.zeros_like(theta)
    for _ in range(num_samples):
        sigma = metropolis_sample_permutation(G, theta_kron, num_iter=100)
        permuted_edges = [(sigma[u], sigma[v]) for u, v in G.edges()]
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                partial_sum = 0.0
                for (u, v) in permuted_edges:
                    if (u % theta.shape[0] == i) and (v % theta.shape[1] == j):
                        partial_sum += 1.0 / theta[i, j]
                grad[i, j] += partial_sum / num_samples
    return grad


def kronfit(G, theta_init, k, learning_rate, iterations):
    theta = theta_init.copy()
    for _ in range(iterations):
        grad = compute_gradient(G, theta, k, num_samples=5)
        theta += learning_rate * grad
        theta = np.clip(theta, 1e-5, 1 - 1e-5)
    return theta


def main():
    parser = argparse.ArgumentParser(description="Run Kronecker graph fitting.")
    parser.add_argument("file_path", type=str, help="Path to the edge list file")
    parser.add_argument("init_matrix", nargs=4, type=float, help="Four parameters for the initiator matrix")
    parser.add_argument("iterations", type=int, help="Number of iterations for fitting")
    parser.add_argument("learning_rate", type=float, help="Learning rate")

    args = parser.parse_args()

    G = nx.read_edgelist(args.file_path, nodetype=int, create_using=nx.DiGraph())
    theta_init = np.array([[args.init_matrix[0], args.init_matrix[1]],
                           [args.init_matrix[2], args.init_matrix[3]]])
    k = int(round(math.log(len(G), len(theta_init))))

    theta_fit = kronfit(G, theta_init, k, args.learning_rate, args.iterations)
    
    # Create output filename from input edge list
    base = os.path.splitext(os.path.basename(args.file_path))[0]
    output_path = os.path.join("gsdl_predictions", f"{base}_gsdl_fit.json")

    os.makedirs("predictions", exist_ok=True)

    # Save predicted theta
    with open(output_path, "w") as f:
        json.dump(theta_fit.tolist(), f)
    
    
    print("Fitted Theta:")
    print(theta_fit)


if __name__ == "__main__":
    main()


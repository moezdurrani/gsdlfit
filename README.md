# GSDLFit

## Kronecker Graph Generator
The script (*create_graphs*) generates synthetic graphs using **Kronecker Graph Expansion**. It starts from a small 2x2 initiator matrix and recursively expands it using Kronecker powers to simulate realistic large-scale networks. The final graphs are saved as edge list files.

### How to Run
```bash
python create_graphs.py --init_matrix a00 a01 a10 a11 --num_of_graphs max_K --sample_per_k S --output_dir path_to_directory/
```
Example usage is shown below
```bash
python create_graphs.py 0.9 0.5 0.6 0.3 3 10 ../data/generated_graphs/
```

For each Kronecker power i, an edge list file is generated.
```bash
output_folder/graph_kron_power{i}_sample{s}.txt
```

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Kronecker graph fitting.")
    parser.add_argument("file_path", type=str, help="Path to the edge list file")
    parser.add_argument("init_matrix", nargs=4, type=float, help="Four parameters for the initiator matrix")
    parser.add_argument("iterations", type=int, help="Number of iterations for fitting")
    parser.add_argument("learning_rate", type=float, help="Learning rate")
    args = parser.parse_args()
    
    
def main():
    args = parse_args()
    
if __name__ == "__main__:
    main()

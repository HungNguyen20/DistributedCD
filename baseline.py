import argparse
from baselines.notears_admm.notears_admm import utils
from baselines.notears_admm.notears_admm.postprocess import postprocess
from baselines.notears_admm.notears_admm.linear_admm import notears_linear_admm
import torch
from pathlib import Path
import os
import numpy as np
import time

def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="m3_d1_n10")
    parser.add_argument("--output", type=str, default="res.csv")
    
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "nonlinear"])
    parser.add_argument("--gt", type=str, default="ER", choices=["ER", "BP", "SF"])
    parser.add_argument("--st", type=str, default="gauss", choices=["gauss, exp, gumbel, uniform, logistic, poisson"])
    
    parser.add_argument("--K", type=int, default=10, help="Number of clients")
    parser.add_argument("--n", type=int, default=100, help="Data volume at each client")
    parser.add_argument("--d", type=int, default=20, help="Number of variables")
    parser.add_argument("--s", type=int, default=20, help="Number of edges")
    
    parser.add_argument("--repeat", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--baseline", type=str, choices=["CDNOD", "notears-admm", "FedCDH", "FedDAG"], default="notears-admm")
    
    options = vars(parser.parse_args())
    return options


def notears_admm_main(Xs):
    # Run NOTEARS-MLP-ADMM
    utils.set_random_seed(1)
    B_est = notears_linear_admm(Xs, lambda1=0.01, verbose=False)
    B_processed = postprocess(B_est, threshold=0.3)
    return B_processed


def load_data(options):
    K, n, d, s = options['K'], options['n'], options['d'], options['s']
    graph_type, sem_type, seed = options['gt'], options['st'], options['seed']
    folder = f"K{K}_{n}_d{d}_s{s}_gt{graph_type}_st{sem_type}_seed{seed}"
    
    if not Path(f"./res/{options['output']}").exists():
        f = open(f"./res/{options['output']}", "w")
        f.write("baseline,etrue,espur,emiss,efals,runtime\n")
        f.close()
    
    if not Path(f"./data/{folder}/data.csv").exists():
        utils.set_random_seed(seed)
        groundtruth = utils.simulate_dag(d, s, graph_type)
        B_true = utils.simulate_parameter(groundtruth)
        X = utils.simulate_linear_sem(B_true, K * n, sem_type)
        os.mkdir(f"./data/{folder}")
        np.savetxt(f"./data/{folder}/data.csv", X, fmt='%.6f', delimiter=",")
        np.savetxt(f"./data/{folder}/graph.csv", groundtruth, fmt='%.1f', delimiter=",")
    
    else:
        X = np.loadtxt(f"./data/{folder}/data.csv", delimiter=",")
        groundtruth = np.loadtxt(f"./data/{folder}/graph.csv", delimiter=",")
        
    Xs = X.reshape(K, n, d)
    return Xs, groundtruth
        
    
if __name__ == "__main__":
    options = read_opts()
    
    # Configuration of torch
    torch.set_default_dtype(torch.double)

    # Load data
    Xs, groundtruth = load_data(options)
    
    # Run algorithms
    if options['baseline'] == "notears-admm":
        st = time.time()
        B_processed = notears_admm_main(Xs)
        
    elif options['baseline'] == "FedDAG":
        st = time.time()
        pass
    
    elif options['baseline'] == "FedCDH":
        st = time.time()
        pass
    
    runtime = time.time() - st
    
    
    # Processed output
    etrue, espur, emiss, efals = utils.count_accuracy(B_processed, groundtruth)
    print(etrue, espur, emiss, efals)
    
    # Write results
    f = open(f"./res/{options['output']}", "a")
    f.write(f"{options['baseline']},{etrue},{espur},{emiss},{efals},{runtime}\n")
    f.close()
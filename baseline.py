import argparse
import os
import torch
from pathlib import Path
import numpy as np
import time

# Imports for notears-admm
from baselines.notears_admm.notears_admm import utils
from baselines.notears_admm.notears_admm.postprocess import postprocess
from baselines.notears_admm.notears_admm.linear_admm import notears_linear_admm

# Imports for FedDAG
from baselines.FedDAG.models import GS_FedDAG
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation # type: ignore
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Imports for FedCDH
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci
from causallearn.utils.data_utils import get_cpdag_from_cdnod, get_dag_from_pdag


def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="m3_d1_n10")
    parser.add_argument("--output", type=str, default="res.csv")
    
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "nonlinear"])
    parser.add_argument("--dt", type=str, default="noniid", choices=["iid", "noniid"])
    parser.add_argument("--gt", type=str, default="ER", choices=["ER", "BP", "SF"], help="Type of graph: Erdos-Renyi, Bipartile, Scale-free")
    parser.add_argument("--st", type=str, default="gauss", choices=["gauss", 'exp', 'gumbel', 'uniform', 'logistic', 'poisson'])
    
    parser.add_argument("--K", type=int, default=10, help="Number of clients")
    parser.add_argument("--n", type=int, default=100, help="Data volume at each client")
    parser.add_argument("--d", type=int, default=10, help="Number of variables")
    parser.add_argument("--s", type=int, default=10, help="Number of edges")
    parser.add_argument("--dz", type=int, default=2, help="Number of missing variables")
    
    parser.add_argument("--num_rounds", type=int, default=100, help="Number of maximum comm rounds - global problem solver")
    parser.add_argument("--repeat", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--data_seed", type=int, default=1, help="Random seed for data")
    parser.add_argument("--baseline", type=str, choices=["CDNOD", "notears-admm", "FedCDH", "FedDAG"], default="notears-admm")
    
    options = vars(parser.parse_args())
    return options


def notears_admm_main(Xs, options, seed):
    # Run NOTEARS-MLP-ADMM
    utils.set_random_seed(seed)
    B_est = notears_linear_admm(Xs, lambda1=0.01, verbose=False, max_iter=options['num_rounds'])
    B_processed = postprocess(B_est, threshold=0.3)
    return B_processed


def feddag_main(Xs, options, seed):
    model = GS_FedDAG(d=options['d'], 
                    num_client=options['K'],
                    use_gpu=False, 
                    seed=seed,
                    max_iter=options['num_rounds'], 
                    num_shared_client=options['K'])
        
    model.learn([Xs[i] for i in range(Xs.shape[0])])
    return model.causal_matrix


def fedcdh_main(Xs, options):
    c_indx = np.asarray(list(range(options["K"])))
    c_indx = np.repeat(c_indx, options['n']) 
    c_indx = np.reshape(c_indx, (options['n'] * options['K'],1)) 
    
    cg = cdnod(Xs, c_indx, options['K'], 0.05, kci, True, 0, -1)

    est_graph = np.zeros((options['d'], options['d']))
    est_graph = cg.G.graph[0:options['d'], 0:options['d']]
    est_cpdag = get_cpdag_from_cdnod(est_graph) # est_graph[i,j]=-1 & est_graph[j,i]=1  ->  est_graph_cpdag[i,j]=1
    est_dag_from_pdag = get_dag_from_pdag(est_cpdag) # return a DAG from a PDAG in causaldag.
    return est_dag_from_pdag


def load_data(options):
    K, n, d, s = options['K'], options['n'], options['d'], options['s']
    dist_type, graph_type, sem_type, seed = options['dt'], options['gt'], options['st'], options['data_seed']
    folder = f"K{K}_{n}_d{d}_s{s}_dt{dist_type}_gt{graph_type}_st{sem_type}_seed{seed}"
    
    if not Path(f"./res/{options['output']}").exists():
        f = open(f"./res/{options['output']}", "w")
        f.write("d,e,dz,model,dt,gt,st,K,n,seed,baseline,etrue,espur,emiss,efals,runtime\n")
        f.close()
    
    if not Path(f"./data/{folder}/data.csv").exists():
        utils.set_random_seed(seed)
        groundtruth = utils.simulate_dag(d, s, graph_type)
        B_true = utils.simulate_parameter(groundtruth)
        noise_scale = np.random.uniform(0,5, size=(K,d))
        
        if dist_type == "noniid":
            X = []
            for k in range(K):
                X.append(utils.simulate_linear_sem(B_true, n, sem_type, noise_scale=noise_scale[k]))
            X = np.vstack(tuple(X))
            
        else:
            X = utils.simulate_linear_sem(B_true, K * n, sem_type, noise_scale[0])
        
        try:
            os.mkdir(f"./data/{folder}")
        except:
            pass
        
        np.savetxt(f"./data/{folder}/data.csv", X, fmt='%.6f', delimiter=",")
        np.savetxt(f"./data/{folder}/graph.csv", groundtruth, fmt='%d', delimiter=",")
    
    else:
        X = np.loadtxt(f"./data/{folder}/data.csv", delimiter=",")
        groundtruth = np.loadtxt(f"./data/{folder}/graph.csv", delimiter=",")
        
    Xs = X.reshape(K, n, d)
    return Xs, groundtruth


def augment_data(Xs: np.ndarray, m=1):
    aug_X = Xs.copy()
    d = aug_X.shape[0]
    for k in range(d):
        for i in range(m):
            aug_X[k][:,(k+i)%d] *= 0
    return aug_X

    
if __name__ == "__main__":
    options = read_opts()
    
    # Configuration of torch
    torch.set_default_dtype(torch.double)

    # Load data
    Xs, groundtruth = load_data(options)
    Xs = augment_data(Xs, options['dz'])
    
    # Run algorithms
    for r in range(options['repeat']):
        print(f"Run {r+1}/{options['repeat']}... ", end="")
        
        if options['baseline'] == "notears-admm":
            st = time.time()
            B_est = notears_admm_main(Xs, options, seed = r**2 + 2*r + 2)
            B_processed = postprocess(B_est, threshold=0.3)
            B_processed[B_processed != 0] = 1
            
        elif options['baseline'] == "FedDAG":
            st = time.time()
            B_processed = feddag_main(Xs, options, seed = r**2 + 2*r + 2)
        
        elif options['baseline'] == "FedCDH":
            st = time.time()
            B_processed = fedcdh_main(Xs, options)
            
        runtime = time.time() - st
        
        # Processed output
        etrue, espur, emiss, efals = utils.count_accuracy(groundtruth, B_processed)
        
        # Write results
        print("Writting results...", end="")
        f = open(f"./res/{options['output']}", "a")
        f.write(f"{options['d']},{options['d']},{options['dz']},{options['model']},{options['dt']},{options['gt']},{options['st']},{options['K']},{options['n']},{options['data_seed']},{options['baseline']},{etrue},{espur},{emiss},{efals},{runtime}\n")
        f.close()
        print("Done!")
    
import argparse
from baselines.notears_admm.notears_admm import utils
from baselines.notears_admm.notears_admm.postprocess import postprocess
from baselines.notears_admm.notears_admm.linear_admm import notears_linear_admm
import torch

def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="asia")
    parser.add_argument("--folder", type=str, default="m3_d1_n10")
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--ntype", type=str, default="linear", choices=["linear", "nonlinear", 
                                                                        "sf_linear", "sf_nonlinear",
                                                                        "bp_linear", "bp_nonlinear"])
    
    parser.add_argument("--K", type=int, default=10, help="Number of clients")
    parser.add_argument("--n", type=int, default=100, help="Data volume at each client")
    parser.add_argument("--d", type=int, default=20, help="Number of variables")
    parser.add_argument("--s", type=int, default=None, help="Number of edges")
    parser.add_argument("--repeat", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--baseline", type=str, 
                        choices=["CDNOD", "notears-admm", "FedCDH", "FedDAG"], 
                        default="notears-admm")
    
    options = vars(parser.parse_args())
    return options


def notears_admm_main(Xs):
    # Run NOTEARS-MLP-ADMM
    utils.set_random_seed(1)
    B_est = notears_linear_admm(Xs, lambda1=0.01, verbose=True)
    B_processed = postprocess(B_est, threshold=0.3)
    return B_processed
    
    
if __name__ == "__main__":
    options = read_opts()
    # Configuration of torch
    torch.set_default_dtype(torch.double)

    # Generate data
    utils.set_random_seed(1)
    K = 8
    n, d, s0, graph_type, sem_type = 32, 20, 20, 'ER', 'gauss'
    B_bin_true = utils.simulate_dag(d, s0, graph_type)
    B_true = utils.simulate_parameter(B_bin_true)
    X = utils.simulate_linear_sem(B_true, K * n, sem_type)
    Xs = X.reshape(K, n, d)
    
    if options['baseline'] == "notears-admm":
        B_processed = notears_admm_main(Xs)
    
    acc = utils.count_accuracy(B_processed, B_bin_true)
    print(acc)
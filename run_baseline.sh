# rm -r data/*
# python baseline.py --baseline "notears-admm" --n 100 --K 10 --d 10 --s 10 --dt "noniid" --data_seed 1 --st "gauss" --dz 1 --output "res2.csv"

# conda activate <abc>

python baseline.py --baseline "notears-admm" --n 100 --K 10 --d 10 --s 10 --gt "ER" --dt "iid" --data_seed 1 --st "gauss" --dz 1 --output "res2.csv" --repeat 2

# python baseline.py --baseline "notears-admm" --n 100 --K 10 --d 10 --s 10 --gt "ER" --dt "iid" --data_seed 1 --st "gauss" --dz 1 --output "res2.csv"

# python baseline.py --baseline "FedDAG" --d 5 --s 5 --K 5 --repeat 5 --dt "iid" --dz 0 --output "res.csv"



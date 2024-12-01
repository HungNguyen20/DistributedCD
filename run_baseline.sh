# rm -r data/*
# python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "noniid" --data_seed 1 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "iid" --data_seed 1 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "noniid" --data_seed 2 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "iid" --data_seed 2 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "noniid" --data_seed 3 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "iid" --data_seed 3 --st "gauss" --dz 1 --output "res2.csv"

python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "noniid" --data_seed 4 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "iid" --data_seed 4 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "noniid" --data_seed 5 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "iid" --data_seed 5 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "noniid" --data_seed 6 --st "gauss" --dz 1 --output "res2.csv"
python baseline.py --baseline "notears-admm" --n 100 --d 10 --s 10 --K 10 --dt "iid" --data_seed 6 --st "gauss" --dz 1 --output "res2.csv"

# python baseline.py --baseline "FedDAG" --d 5 --s 5 --K 5 --repeat 5 --dt "iid" --dz 0 --output "res.csv"



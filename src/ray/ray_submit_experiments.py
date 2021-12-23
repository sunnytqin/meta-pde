from collections import Counter
import socket
import time
import os
from subprocess import Popen
import ray

ray.init(address="auto")

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
    {} GPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU'], ray.cluster_resources()['GPU']))

expt_cmds = [
    f"python -m src.nn_pde --pde 'td_burgers' --max_holes 0 --xmin -1.0 --ymin -1.0 --xmax 1.0 --ymax 1.0 "
    f"--viz_every 2_000 --log_every 1_000 --num_layers 12 --siren_omega 30.0 "    
    f"--siren_omega0 30.0 --outer_steps 50_000 --outer_lr 1.0e-06 --optimizer adam --bc_weight 10 "
    f"--td_burger_impose_symmetry=False --tmax 0.5 --num_tsteps 6 --validation_points=4096 "
    f"--outer_points=4096 --ground_truth_resolution 32 --vary_source=False "
#    f"--load_model_from_expt=td_burgers_nn_results/tmax_2_bc_weight_2.0 "
    f"--expt_name tmax_0.5_trivial_2_bc_weight"
    ]

#expt_cmds = [
#    f"python -m src.nn_pde --pde 'td_burgers' --max_holes 0 --xmin -1.0 --ymin -1.0 --xmax 1.0 --ymax 1.0 "
#    f"--viz_every 1_000 --log_every 500 --num_layers 12 --siren_omega 30.0 "
#    f"--siren_omega0 30.0 --outer_steps 50_000 --outer_lr 1.0e-05 --optimizer adam --bc_weight 1.0e-3 "
#    f"--td_burger_impose_symmetry=True --tmax 2.0 --num_tsteps 40 --validation_points=163840 "
#    f"--outer_points=10240 --ground_truth_resolution 32 --vary_source=False "
#    f"--load_model_from_expt=td_burgers_leap_results/tmax_2_2 "
#    f"--expt_name tmax_2_pretrain_d"
#    ]

#expt_cmds = [
#    f"python -m src.nn_pde --pde 'td_burgers' --max_holes 0 --xmin -1.0 --ymin -1.0 --xmax 1.0 --ymax 1.0 "
#    f"--viz_every 1_000 --log_every 100 --num_layers 12 --siren_omega 5.0 "
#    f"--siren_omega0 5.0 --outer_steps 5_000 --outer_lr 1.0e-05 --optimizer adam --bc_weight 1.0 "
#    f"--td_burger_impose_symmetry=True --tmax 2.0 --num_tsteps 40 --validation_points=81920 "
#    f"--outer_points=4096 --ground_truth_resolution 32 --vary_source=False "
#    f"--load_model_from_expt=td_burgers_leap_results/inner_expt_125 "
#    f"--expt_name tmax_2_pretrain_d"
#    ]
    
#expt_cmds = [
#    f"python -m src.leap_pde --pde 'td_burgers' --max_holes 0 --xmin -1.0 --ymin -1.0 --xmax 1.0 --ymax 1.0 "
#    f"--viz_every 1_000 --log_every 1_00 --num_layers 12 --siren_omega 30.0 "
#    f"--siren_omega0 30.0 --outer_steps 50_000 --outer_lr 3.0e-04 --inner_lr 1.0e-05 --optimizer adam --bc_weight 1.0e-3 "
#    f"--td_burger_impose_symmetry=True --tmax 2.0 --num_tsteps 40 --validation_points=1024 --inner_points=4096 "
#    f"--outer_points=4096 --inner_steps 20 --ground_truth_resolution 32 --vary_source=False "
#    f"--expt_name tmax_2_2"
#    ]

@ray.remote(num_gpus=1)
def run_experiments(i):
    # Return IP address.
    if "meta-pde" not in os.getcwd():
        os.chdir("meta-pde")
    print(expt_cmds[i])
    os.system(expt_cmds[i])
    return ray._private.services.get_node_ip_address()


remaining_ids = [run_experiments.remote(i) for i in range(len(expt_cmds))]
total_jobs = len(remaining_ids)
while remaining_ids:
    time.sleep(3)
    ready_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
    finished_worker_ip = ray.get(ready_ids[0])

    #SYNC_DIRECTORY = '/home/ubuntu/meta-pde/td_burgers_nn_results/'
    #LOCAL_SYNC_DIRECTORY = '/home/ubuntu/meta-pde/'
    #AWS_KEY_LOC='~/ray-autoscaler_7_us-east-2.pem'
    #RSYNC_COMMAND = 'rsync -savz -e \"ssh -i {aws_key} -o ConnectTimeout=120s -o StrictHostKeyChecking=no\" {node_ip}:{sync_dir} {loc_sync_dir}'
    #command = RSYNC_COMMAND.format(aws_key=AWS_KEY_LOC, node_ip=finished_worker_ip, sync_dir=SYNC_DIRECTORY, loc_sync_dir=LOCAL_SYNC_DIRECTORY)
    #print(command)
    #os.system(command)

#ip_addresses = ray.get(object_ids)
print('Tasks executed')
#for ip_address, num_tasks in Counter(ip_addresses).items():
#    print('    {} tasks on {}'.format(num_tasks, ip_address))
#    finished_worker_ip = ray.get(ready_ids[0])
#    if finished_worker_ip == 0:
#        print('Exception occured in subtask. Continuing.')
#        continue

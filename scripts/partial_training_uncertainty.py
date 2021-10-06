

import subprocess
import os
from subprocess import Popen, PIPE
import glob


def select_data_via_entropy(checkpoint_path,idx,n):
    selected = "uncertainty_splits/accumulating_{}.txt".format(n)
    if idx > 0:
        unselected = "uncertainty_splits/remaining_{}.txt".format(n)
    else:    
        unselected = "uncertainty_splits/remaining_{}.txt".format(n)
    path = "uncertainty_splits/remaining_{}.txt".format(n)
    command = "python scripts/eval_with_uncertainty.py --dataset scannet --selected_path {} --unselected_path {} --checkpoint_path {} --num_point 40000 --num_samples 5  --cluster_sampling   seed_fps --batch_size  8   --use_3d_nms --use_cls_nms  --num_batch -1 --conf_thresh 0.5 --custom_path {}".format(
        selected,
        path,
        checkpoint_path,
        unselected
    )

    print(command)
    subprocess.call(command,
                    stdout=subprocess.PIPE,
                    shell=True,
                    )


splits = [ 0.2, 0.3, 0.4, 0.5, 0.6]
trial_start = 3
trial_end = 5
exp_name = "uncertainty_accumulation"
for n in range(trial_start, trial_end + 1):
    for idx, s in enumerate(splits):
        
        custom_path = os.path.join(
            "uncertainty_splits", "accumulating_{}.txt".format(n))
        log_dir = os.path.join(
            "logs", "{}_{}_{}".format(exp_name, idx, n))
        if idx > 0:
            prev_log_dir = os.path.join(
                "logs", "{}_{}_{}".format(exp_name, idx-1, n))
            checkpoints = glob.glob("{}/*.tar".format(prev_log_dir))
            print(prev_log_dir)
            checkpoints.sort(key=os.path.getmtime, reverse=True)

            start_iter = checkpoints[0]
            dash = start_iter.rfind("_")
            dot = start_iter.rfind(".")
            start_iter = start_iter[dash+1:dot]

            checkpoint_path = os.path.join("logs", "{}_{}_{}".format(
                exp_name, idx-1, n), "best_checkpoint_{}.tar".format(start_iter))
        else:
            # continue
            checkpoint_path = "logs/random_accumulation_0_{}".format(n)
            checkpoints = glob.glob("{}/*.tar".format(checkpoint_path))
            
            checkpoints.sort(key=os.path.getmtime, reverse=True)

            start_iter = checkpoints[0]
            dash = start_iter.rfind("_")
            dot = start_iter.rfind(".")
            start_iter = start_iter[dash+1:dot]

            checkpoint_path = os.path.join("logs", "{}_{}_{}".format(
                "random_accumulation", 0, n), "best_checkpoint_{}.tar".format(start_iter))
            start_iter = 0
        
        print("Checkpoint : ", checkpoint_path)
        print("Log dir : ", log_dir)
        print("Custom path: ",custom_path)
        # if idx is 0:
        #     continue
        select_data_via_entropy(checkpoint_path,idx,n)
        
        print("Sorted things")
        print("Start iter : ",start_iter)
        p = subprocess.call(
            "python scripts/train.py  --dataset scannet --start_iter {} --log_dir {}  --custom_path {} --checkpoint_path {} --num_point 40000 --max_epoch 200 --batch_size 8 --ratio 0.23 ".format  
            (
            start_iter,    
            log_dir,
            custom_path,
            checkpoint_path
            ),
            stdout=subprocess.PIPE,
            shell=True,
        )
        # print("Start iter : s

# output, err = p.communicate(b"input data that is passed to subprocess' stdin")
# rc = p.returncode
# print(output.decode())

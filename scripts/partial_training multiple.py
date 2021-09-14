

import subprocess
import os
from subprocess import Popen, PIPE
import glob

splits = os.listdir("/home/yildirir/workspace/votenet/splits")
trial_start = 3
trial_end = 6
for n in range(trial_start,trial_end + 1):

    for idx,s in enumerate(splits):
        custom_path = os.path.join("splits",s)
        log_dir =  os.path.join("logs","random_accumulation_{}_{}".format(idx,n))
        if idx > 0:
            prev_log_dir =  os.path.join("logs","random_accumulation_{}_{}".format(idx-1,n))
            checkpoints = glob.glob("{}/*.tar".format(prev_log_dir))
            print(prev_log_dir)
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            
            start_iter = checkpoints[0]
            dash = start_iter.rfind("_")
            dot = start_iter.rfind(".")
            start_iter = start_iter[dash+1:dot]


            checkpoint_path =  os.path.join("logs","random_accumulation_{}_{}".format(idx-1,n),"best_checkpoint_{}.tar".format(start_iter))
        else:
            # continue
            checkpoint_path =  "NONEXISTING/PATH"
            start_iter = 0
        print("Checkpoint : ",checkpoint_path)
        print("Log dir : ",log_dir)
        print("Start iter : ",start_iter)
        p = subprocess.call(
            "python scripts/train.py  --dataset scannet --start_iter {} --log_dir {}  --custom_path {} --checkpoint_path {} --num_point 40000 --max_epoch 200 --batch_size 8 --ratio 0.23 > tmp.txt".format  
            (
            start_iter,    
            log_dir,
            custom_path,
            checkpoint_path
            ),
            stdout=subprocess.PIPE,
            shell=True,
        )

# output, err = p.communicate(b"input data that is passed to subprocess' stdin")
# rc = p.returncode
# print(output.decode())

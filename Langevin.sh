#!/bin/sh 
### General options 
#BSUB -q gpuv100 ### -- specify queue -- 
#BSUB -J Langevin ### -- set the job Name -- 
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]" ### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -W 4:00 ### -- set walltime limit: hh:mm -- 
#BSUB -u s223517@dtu.dk ### -- set the email address -- 
###BSUB -B ### -- send notification at start -- 
###BSUB -N ### -- send notification at completion -- 
#BSUB -o hpc_outputs/Langevin/out.out  ### -- Specify the output and error file. %J is the job-id -- 
#BSUB -e hpc_outputs/Langevin/err.err  ### -- -o and -e mean append, -oo and -eo mean overwrite -- 

JOB_DIR="hpc_outputs/Langevin/${LSB_JOBID}"  # LSB_JOBID is an environment variable holding the job ID
mkdir -p "$JOB_DIR"


nvidia-smi
module load cuda/11.6 # Load the cuda module
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source ../venv_2/bin/activate

python -u Langevin.py 100 1 0.1 100 1 0.1 > hpc_outputs/Langevin/${LSB_JOBID}/log.log

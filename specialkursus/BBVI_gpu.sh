#!/bin/sh 
### General options 
#BSUB -q gpuv100 ### -- specify queue -- 
#BSUB -J BBVI ### -- set the job Name -- 
#BSUB -gpu "num=1:mode=exclusive_process" ### if GPU is needed uncomment this
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]" ### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -M 5GB ### -- specify that we want the job to get killed if it exceeds 11 GB per core/slot -- 
#BSUB -W 01:30 ### -- set walltime limit: hh:mm -- 
#BSUB -u s230368@dtu.dk ### -- set the email address -- 
#BSUB -B ### -- send notification at start -- 
#BSUB -N ### -- send notification at completion -- 
#BSUB -o hpc_outputs/out_%J.out  ### -- Specify the output and error file. %J is the job-id -- 
#BSUB -e hpc_outputs/rr_%J.err  ### -- -o and -e mean append, -oo and -eo mean overwrite -- 

nvidia-smi
module load cuda/11.6 # Load the cuda module
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source ../../../ML/bin/activate

python -u BBVI.py > hpc_outputs/log_${LSB_JOBID}.log
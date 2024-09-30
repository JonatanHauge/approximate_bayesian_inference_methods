#!/bin/sh 
### General options 
#BSUB -q gpuv100 ### -- specify queue -- 
#BSUB -J VGG16 ### -- set the job Name -- 
#BSUB -gpu "num=1:mode=exclusive_process" ### if GPU is needed uncomment this
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]" ### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -M 11GB ### -- specify that we want the job to get killed if it exceeds 11 GB per core/slot -- 
#BSUB -W 02:30 ### -- set walltime limit: hh:mm -- 
#BSUB -u s230368@dtu.dk ### -- set the email address -- 
###BSUB -B ### -- send notification at start -- 
###BSUB -N ### -- send notification at completion -- 
#BSUB -o hpc_outputs/VGG/%J/out.out  ### -- Specify the output and error file. %J is the job-id -- 
#BSUB -e hpc_outputs/VGG/%J/err.err  ### -- -o and -e mean append, -oo and -eo mean overwrite -- 
JOB_DIR="hpc_outputs/VGG/${LSB_JOBID}"  # LSB_JOBID is an environment variable holding the job ID
mkdir -p "$JOB_DIR"

nvidia-smi
module load cuda/11.6 # Load the cuda module
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source ../../../ML/bin/activate

python -u BBVI_VGG.py > hpc_outputs/VGG/${LSB_JOBID}/log.log
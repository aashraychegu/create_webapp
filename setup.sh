#!/bin/bash
slurm_jobid_prpt () { echo ""; }
export -f slurm_jobid_prpt
module restore data_product
alias queue="squeue -u aashrayc --format='%.18i %.9P %.75j %.8u %.10T %.10M %.10l %.6D %R'"

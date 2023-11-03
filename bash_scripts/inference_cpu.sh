#!/bin/bash
# change the directory path of model run-time output and error messages to your own
#SBATCH --output=/scratch/gilbreth/gupt1075/inference_cpu.out
#SBATCH --error=/scratch/gilbreth/gupt1075/inference_cpu.err
# The file name of this submission file, so it's easier to track jobs
# filename: submit_run_model_example.sub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:0
#SBATCH --time=2:00:00
# partner queue has a 24-hour limit
#SBATCH -A gdsp-k|standby
# Job name, it will show up when you track this job
#SBATCH -J fourcastnet_inference_cpu
# Use your email address so that you will receive email notifications about the job begin, end, or fail status
# To submit the job via command line:$  sbatch submit_run_model_example.sub 
# To check status of the submitted job:$  squeue -u yourUserID

module --force purge
unset PYTHONPATH
module load anaconda/5.3.1-py37
module use /depot/gdsp/etc/modules
module load utilities monitor
module load rcac

module list
export PRECXX11ABI=1


echo $PYTHONPATH

echo "$now"
echo "Current date completed loading modules: $now"






# Loading anaconda environment
source  /apps/spack/gilbreth/apps/anaconda/5.3.1-py37-gcc-4.8.5-7vvmykn/etc/profile.d/conda.sh
conda activate pytorch


# Change this directory to where you save the model-related files such as run_model.py
# cd /scratch/gilbreth/wwtung/FourCastNet/

python /scratch/gilbreth/gupt1075/FourCastNet/inference/inference.py \
       --config='afno_backbone' \
       --run_num='02' \
       --weights="/scratch/gilbreth/gupt1075/model_weights/FCN_weights_v0/backbone.ckpt"  \
       --override_dir="/scratch/gilbreth/gupt1075/ERA5_expts_22_feb_2018_v10_trial_1/" \
       --fld="v10"







# 1.1 Basics of GPU and CUDA Training 

qsub -q gpu -l select=1:ncpus=10:ngpus=1,walltime=01:00:00 -I
module load nv-sdk-tool/nvhpc/23.5

nvidia-smi
nvcc --version

nvcc exe-1.cu -o exe-1
./exe-1

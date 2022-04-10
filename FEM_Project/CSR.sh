#!/bin/bash -l

#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -J CSR # name of the job
#SBATCH --output=CSR_gpu.log

./app.cuda -N 32 -repeat 200 -number float > CSR_float_32.log 2>&1
./app.cuda -N 64 -repeat 200 -number float > CSR_float_64.log 2>&1
./app.cuda -N 128 -repeat 200 -number float > CSR_float_128.log 2>&1
./app.cuda -N 256 -repeat 200 -number float > CSR_float_256.log 2>&1
./app.cuda -N 32 -repeat 200 -number double > CSR_double_32.log 2>&1
./app.cuda -N 64 -repeat 200 -number double > CSR_double_64.log 2>&1
./app.cuda -N 128 -repeat 200 -number double > CSR_double_128.log 2>&1
./app.cuda -N 256 -repeat 200 -number double > CSR_double_256.log 2>&1 

#!/bin/bash -l

#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH -J stream_cuda      # name of the job
#SBATCH --output=double_cpu_16.log

./app.host -N 32 -repeat 200 -number float > float_32.log 2>&1
./app.host -N 64 -repeat 200 -number float > float_64.log 2>&1
./app.host -N 128 -repeat 200 -number float > float_128.log 2>&1
./app.host -N 256 -repeat 200 -number float > float_256.log 2>&1
./app.host -N 32 -repeat 200 -number double > double_32.log 2>&1
./app.host -N 64 -repeat 200 -number double > double_64.log 2>&1
./app.host -N 128 -repeat 200 -number double > double_128.log 2>&1
./app.host -N 256 -repeat 200 -number double > double_256.log 2>&1 

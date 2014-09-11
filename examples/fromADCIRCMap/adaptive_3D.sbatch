#!/bin/bash
#SBATCH -p normal 
#SBATCH -J adaptive
#SBATCH -n 320                  # The job acquires enough nodes to execute total_tasks tasks (launching 16 tasks/node). Use the -N option with the -n option when fewer than 16 tasks/node are required (e.g. for hybrid codes).
#SBATCH -o adapt3D.o.%j
#SBATCH -e adapt3D.e.%j
#SBATCH --mail-user=youremail@place.com
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -t 05:00:00

python adaptive_3D.py

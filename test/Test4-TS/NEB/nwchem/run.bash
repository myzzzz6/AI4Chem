#!/bin/bash

traj="traj.xyz"
n_atoms=$(head -n1 "$traj")
lines_per_frame=$((n_atoms+2))
frames=$(wc -l < "$traj")
total_frames=$((frames / lines_per_frame))

#for ((i=2; i<total_frames; i+=2)); do
#for ((i=20; i<total_frames; i+=1))
#for i in 2 10 18 19 20 21 28 36 44 48; do
for i in  19 21 ; do
  start_line=$((i * lines_per_frame + 1))
  end_line=$((start_line + lines_per_frame -1))

  sed -n "${start_line},${end_line}p" "$traj" > calc.xyz
  cp calc.xyz $i.xyz
  echo "Running frame $i ..."
  nwchem-docker.bash nwchem.in 5 20GB  > calc$i-wb97.out

done


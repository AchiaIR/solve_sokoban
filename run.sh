#!/bin/bash

if [ "$#" -eq 0 ]; then
    python solve_sokoban.py -config_file config.yaml
else
    python solve_sokoban.py -config_file config.yaml SOKOBAN.NUM_BOXES $1 
fi
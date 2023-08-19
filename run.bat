@echo off
IF "%~1"=="" (
    python solve_sokoban.py -config_file config.yaml
) ELSE (
    python solve_sokoban.py -config_file config.yaml SOKOBAN.NUM_BOXES %1 
)

This directory is supposed to run on SLURM cluster of UKP!! 


1) Setup a conda environment using the environment file in this directory.

2) execute run.sh

_______________________________________________

In case you try to execute it somehow else:

execute main.py will do basically everything. ensure you use an evironemnt that meets the requirements listed in requirements.txt
1) train all models and save them at each at models/model_name

2) calculate a performance comparison for the given test data at model_performance_comparison.json

3) place an output.txt file in this directory to allow to proofread if the output does not only have a good score but is actually good. Convince yourself.
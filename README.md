# CIL_PROJECT

You can use winSCP to upload datasets to leonhard cluster.

1. Log in the Leonhard cluster

2. Clone the github repository: 
```
git clone link_for_repository.git
```
*If the CIL_PROJECT has been updated, you need to clone the repository again. You can remove the old one with 'rm -rf CIL_PROJECT'*

3. Set up the virtual environment and modules
```
source init_leonhard.sh
```
*You need to create a virtual environment first by running 'python -m venv venv'. Afterwards, a virtual environment called venv is created in your home directory*

4. Submit the jobs, parameters specify the number of the gpus and the memory we request, and the time limit we set for our task
```
bsub -R "rusage[ngpus_excl_p=1,mem=64000]" -W 1:00 baseline1.ipynb
```

5. 'bjobs' - command to see if the job is running or pending, 'bpeek job_id' - command to monitor the job


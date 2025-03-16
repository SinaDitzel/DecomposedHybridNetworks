# Uncertainty-Aware Decomposed Hybrid Networks

This repository contains the implementation of the paper "Uncertainty-Aware Decomposed Hybrid Networks". 

## Usage

### Install

To install the necessary dependencies, run:

```
pip install -r requirements.txt
```

Install jupyter notebook if you want to run the notebooks.

### File Structure

- `notebooks/`: Jupyter Notebooks visualizing the pipeline steps. To make it easier for reviewers to follow the process, the notebooks have been saved with all outputs included. 
- `main.py`: Main script to launch experiments, run `python3 main.py -h` to see all the options 
- `config_runs_paper/`: Configuration files for each run performed in the paper.
- `models/`: Contains the method implementation: rg, LBP, noise estimation, confidence calculation, decomposed network...

### How to Run the Experiments

1. Set up your environment with the necessary dependencies using the `requirements.txt`.
2. Download the GTSRB dataset, we expect the data to be in the structure:
```
    ├── GTSRB/
    │   ├── GTSRB_Final_Training/
    │   │   └── Images_train
    │   │   └── Images_val
    │   ├── GTSRB_Final_Test/ 
```           
2. Adapt the datapath in the config-files and run the bash script
 

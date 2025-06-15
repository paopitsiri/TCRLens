# TCRLens: Structure-Aware Equivariant Graph Learning for TCR-pMHC-I Recognition and Immunogenic Epitope Discovery
We introduce TCRLens, a structure-aware deep learning framework that models residue-level interactions across five critical interface zones using multi-scale graph representations and an equivariant graph neural network (EGNN).

# Requirements
To set up the conda environment, use the provided `environment.yml` file:

conda env create -f environment.yml
conda activate TCRLens

# Quick start
The raw structural data in PDB format and corresponding binding affinity information are stored in the `data_raw` directory.  
To generate interface graphs in HDF5 format, use the preprocessing scripts located in the `generated_script` folder.



#!/usr/bin/env python
# coding: utf-8

# # Data preparation for protein-protein interfaces
# 

# ## Introduction
# 
# <img style="margin-left: 1.5rem" align="right" src="images/data_generation_ppi.png" width="400">
# 
# This tutorial will demonstrate the use of DeepRank2 for generating protein-protein interface (PPI) graphs and saving them as [HDF5 files](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files, using [PBD files](<https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)>) of protein-protein complexes as input.
# 
# In this data processing phase, for each protein-protein complex an interface is selected according to a distance threshold that the user can customize, and it is mapped to a graph. Nodes either represent residues or atoms, and edges are the interactions between them. Each node and edge can have several different features, which are generated and added during the processing phase as well. Optionally, the graphs can be mapped to volumetric grids (i.e., 3D image-like representations), together with their features. The mapped data are finally saved into HDF5 files, and can be used for later models' training (for details go to [training_ppi.ipynb](https://github.com/DeepRank/deeprank2/blob/main/tutorials/training_ppi.ipynb) tutorial). In particular, graphs can be used for the training of Graph Neural Networks (GNNs), and grids can be used for the training of Convolutional Neural Networks (CNNs).
# 

# ### Input Data
# 
# The example data used in this tutorial are available on Zenodo at [this record address](https://zenodo.org/record/7997585). To download the raw data used in this tutorial, please visit the link and download `data_raw.zip`. Unzip it, and save the `data_raw/` folder in the same directory as this notebook. The name and the location of the folder are optional but recommended, as they are the name and the location we will use to refer to the folder throughout the tutorial.
# 
# Note that the dataset contains only 100 data points, which is not enough to develop an impactful predictive model, and the scope of its use is indeed only demonstrative and informative for the users.
# 

# ## Utilities
# 

# ### Libraries
# 

# The libraries needed for this tutorial:
# 

# In[ ]:


import contextlib
import glob
import os
from pathlib import Path

import h5py
import matplotlib.image as img
import matplotlib.pyplot as plt
import pandas as pd

from deeprank2.dataset import GraphDataset
from deeprank2.features import components, contact, exposure, surfacearea, secondary_structure
from deeprank2.query import ProteinProteinInterfaceQuery, QueryCollection
from deeprank2.utils.grid import GridSettings, MapMethod


# ### Raw files and paths
# 

# The paths for reading raw data and saving the processed ones:
# 

# In[ ]:


data_path = os.path.join("data_raw_add2", "ppi")
processed_data_path = os.path.join("data_processed_add2_CE_8A", "ppi")
residue_data_path = os.path.join(processed_data_path, "residue")

for output_path in [residue_data_path]:
    os.makedirs(output_path, exist_ok=True)
    if any(Path(output_path).iterdir()):
        msg = f"Please store any required data from `./{output_path}` and delete the folder.\nThen re-run this cell to continue."
        raise FileExistsError(msg)

# Flag limit_data as True if you are running on a machine with limited memory (e.g., Docker container)
limit_data = False


# - Raw data are PDB files in `data_raw/ppi/pdb/`, which contains atomic coordinates of the protein-protein complexes of interest, so in our case of pMHC complexes.
# - Target data, so in our case the BA values for the pMHC complex, are in `data_raw/ppi/BA_values.csv`.
# - The final PPI processed data will be saved in `data_processed/ppi/` folder, which in turns contains a folder for residue-level data and another one for atomic-level data. More details about such different levels will come a few cells below.
# 

# `get_pdb_files_and_target_data` is an helper function used to retrieve the raw pdb files names in a list and the BA target values from a CSV containing the IDs of the PDB models as well:
# 

# In[ ]:


def get_pdb_files_and_target_data(data_path: str) -> tuple[list[str], list[float]]:
    csv_data = pd.read_csv(os.path.join(data_path, "BA_values.csv"))
    pdb_files = glob.glob(os.path.join(data_path, "pdb", "*.pdb"))
    pdb_files.sort()
    pdb_ids_csv = [pdb_file.split("/")[-1].split(".")[0] for pdb_file in pdb_files]
    with contextlib.suppress(KeyError):
        csv_data_indexed = csv_data.set_index("ID")
    csv_data_indexed = csv_data_indexed.loc[pdb_ids_csv]
    bas = csv_data_indexed.Kd.tolist()

    return pdb_files, bas


pdb_files, bas = get_pdb_files_and_target_data(data_path)

if limit_data:
    pdb_files = pdb_files[:15]


# ## `QueryCollection` and `Query` objects
# 

# For each protein-protein complex, so for each data point, a query can be created and added to the `QueryCollection` object, to be processed later on.
# 
# A query takes as inputs:
# 
# - A `.pdb` file, representing the protein-protein structural complex.
# - The resolution (`"residue"` or `"atom"`), i.e. whether each node should represent an amino acid residue or an atom.
# - The ids of the two chains composing the complex. In our use case, "M" indicates the MHC protein chain and "P" the peptide chain.
# - The interaction radius, which determines the threshold distance (in Ångström) for residues/atoms surrounding the interface that will be included in the graph.
# - The target values associated with the query. For each query/data point, in the use case demonstrated in this tutorial will add two targets: "BA" and "binary". The first represents the actual BA value of the complex in nM, while the second represents its binary mapping, being 0 (BA > 500 nM) a not-binding complex and 1 (BA <= 500 nM) a binding one.
# - The max edge distance, which is the maximum distance between two nodes to generate an edge between them.
# - Optional: The correspondent [Position-Specific Scoring Matrices (PSSMs)](https://en.wikipedia.org/wiki/Position_weight_matrix), in the form of .pssm files. PSSMs are optional and will not be used in this tutorial.
# 

# ## Residue-level PPIs using `ProteinProteinInterfaceQuery`
# 

# In[ ]:


queries = QueryCollection()

influence_radius = 8  # max distance in Å between two interacting residues/atoms of two proteins
max_edge_length = 8
binary_target_value = 100

print(f"Adding {len(pdb_files)} queries to the query collection ...")
for i in range(len(pdb_files)):
    queries.add(
        ProteinProteinInterfaceQuery(
            pdb_path=pdb_files[i],
            resolution="residue",
            chain_ids=["C", "E"],
            influence_radius=influence_radius,
            max_edge_length=max_edge_length,
            targets={
                "binary": int(float(bas[i]) <= binary_target_value),
                "Kd": bas[i],  # continuous target value
            },
        ),
    )
    if i + 1 % 20 == 0:
        print(f"{i+1} queries added to the collection.")

print(f"{i+1} queries ready to be processed.\n")


# ### Notes on `process()` method
# 
# Once all queries have been added to the `QueryCollection` instance, they can be processed. Main parameters of the `process()` method, include:
# 
# - `prefix` sets the output file location.
# - `feature_modules` allows you to choose which feature generating modules you want to use. By default, the basic features contained in `deeprank2.features.components` and `deeprank2.features.contact` are generated. Users can add custom features by creating a new module and placing it in the `deeprank2.feature` subpackage. A complete and detailed list of the pre-implemented features per module and more information about how to add custom features can be found [here](https://deeprank2.readthedocs.io/en/latest/features.html).
#   - Note that all features generated by a module will be added if that module was selected, and there is no way to only generate specific features from that module. However, during the training phase shown in `training_ppi.ipynb`, it is possible to select only a subset of available features.
# - `cpu_count` can be used to specify how many processes to be run simultaneously, and will coincide with the number of HDF5 files generated. By default it takes all available CPU cores and HDF5 files are squashed into a single file using the `combine_output` setting.
# - Optional: If you want to include grids in the HDF5 files, which represent the mapping of the graphs to a volumetric box, you need to define `grid_settings` and `grid_map_method`, as shown in the example below. If they are `None` (default), only graphs are saved.
# 

# In[ ]:


#grid_settings = GridSettings(  # None if you don't want grids
    # the number of points on the x, y, z edges of the cube
#    points_counts=[35, 30, 30],
    # x, y, z sizes of the box in Å
#    sizes=[1.0, 1.0, 1.0],
#)
#grid_map_method = MapMethod.GAUSSIAN  # None if you don't want grids

grid_settings = None
grid_map_method = None

queries.process(
    prefix=os.path.join(processed_data_path, "residue", "proc"),
    feature_modules=[components, contact, exposure, surfacearea],
    cpu_count=1,
    combine_output=False,
    grid_settings=grid_settings,
    grid_map_method=grid_map_method,
)

print(f'The queries processing is done. The generated HDF5 files are in {os.path.join(processed_data_path, "residue")}.')


# ### Exploring data
# 
# As representative example, the following is the HDF5 structure generated by the previous code for `BA-100600.pdb`, so for one single graph, which represents one PPI, for the graph + grid case:
# 
# ```bash
# └── residue-ppi:M-P:BA-100600
#     |
#     ├── edge_features
#     │   ├── _index
#     │   ├── _name
#     │   ├── covalent
#     │   ├── distance
#     │   ├── electrostatic
#     │   ├── same_chain
#     │   └── vanderwaals
#     |
#     ├── node_features
#     │   ├── _chain_id
#     │   ├── _name
#     │   ├── _position
#     │   ├── hb_acceptors
#     │   ├── hb_donors
#     │   ├── polarity
#     │   ├── res_charge
#     │   ├── res_mass
#     |   ├── res_pI
#     |   ├── res_size
#     |   └── res_type
#     |
#     ├── grid_points
#     │   ├── center
#     │   ├── x
#     │   ├── y
#     │   └── z
#     |
#     ├── mapped_features
#     │   ├── _position_000
#     │   ├── _position_001
#     │   ├── _position_002
#     │   ├── covalent
#     │   ├── distance
#     │   ├── electrostatic
#     │   ├── polarity_000
#     │   ├── polarity_001
#     │   ├── polarity_002
#     │   ├── polarity_003
#     |   ├── ...
#     |   └── vanderwaals
#     |
#     └── target_values
#     │   ├── BA
#         └── binary
# ```
# 
# `edge_features`, `node_features`, `mapped_features` are [HDF5 Groups](https://docs.h5py.org/en/stable/high/group.html) which contain [HDF5 Datasets](https://docs.h5py.org/en/stable/high/dataset.html) (e.g., `_index`, `electrostatic`, etc.), which in turn contains features values in the form of arrays. `edge_features` and `node_features` refer specificly to the graph representation, while `grid_points` and `mapped_features` refer to the grid mapped from the graph. Each data point generated by deeprank2 has the above structure, with the features and the target changing according to the user's settings. Features starting with `_` are present for human inspection of the data, but they are not used for training models.
# 
# It is always a good practice to first explore the data, and then make decision about splitting them in training, test and validation sets. There are different possible ways for doing it.
# 

# #### Pandas dataframe
# 
# The edge and node features just generated can be explored by instantiating the `GraphDataset` object, and then using `hdf5_to_pandas` method which converts node and edge features into a [Pandas](https://pandas.pydata.org/) dataframe. Each row represents a ppi in the form of a graph.
# 

# In[ ]:


processed_data = glob.glob(os.path.join(processed_data_path, "residue", "*.hdf5"))
dataset = GraphDataset(processed_data, target="binary")
dataset_df = dataset.hdf5_to_pandas()
dataset_df.head()


# We can also generate histograms for looking at the features distributions. An example:
# 

# In[ ]:


#fname = os.path.join(processed_data_path, "residue", "res_mass_distance_electrostatic")
#dataset.save_hist(features=["res_mass", "distance", "electrostatic"], fname=fname)

#im = img.imread(fname + ".png")
#plt.figure(figsize=(15, 10))
#fig = plt.imshow(im)
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.get_yaxis().set_visible(False)


# #### Other tools
# 
# - [HDFView](https://www.hdfgroup.org/downloads/hdfview/), a visual tool written in Java for browsing and editing HDF5 files.
#   As representative example, the following is the structure for `BA-100600.pdb` seen from HDF5View:
# 
#   <img style="margin-bottom: 1.5rem" align="centrum" src="images/hdfview_ppi.png" width="200">
# 
#   Using this tool you can inspect the values of the features visually, for each data point.
# 
# - Python packages such as [h5py](https://docs.h5py.org/en/stable/index.html). Examples:
# 

# In[ ]:


with h5py.File(processed_data[0], "r") as hdf5:
    # List of all graphs in hdf5, each graph representing a ppi
    ids = list(hdf5.keys())
    print(f"IDs of PPIs in {processed_data[0]}: {ids}")
    node_features = list(hdf5[ids[0]]["node_features"])
    print(f"Node features: {node_features}")
    edge_features = list(hdf5[ids[0]]["edge_features"])
    print(f"Edge features: {edge_features}")
    target_features = list(hdf5[ids[0]]["target_values"])
    print(f"Targets features: {target_features}")
    # Polarity feature for ids[0], numpy.ndarray
    node_feat_polarity = hdf5[ids[0]]["node_features"]["polarity"][:]
    print(f"Polarity feature shape: {node_feat_polarity.shape}")
    # Electrostatic feature for ids[0], numpy.ndarray
    edge_feat_electrostatic = hdf5[ids[0]]["edge_features"]["electrostatic"][:]
    print(f"Electrostatic feature shape: {edge_feat_electrostatic.shape}")

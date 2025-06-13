#!/usr/bin/env python
# coding: utf-8

# # Training Neural Networks
# 

# ## Introduction
# 
# <img style="margin-left: 1.5rem" align="right" src="images/training_ppi.png" width="400">
# 
# This tutorial will demonstrate the use of DeepRank2 for training graph neural networks (GNNs) and convolutional neural networks (CNNs) using protein-protein interface (PPI) or single-residue variant (SRV) data for classification and regression predictive tasks.
# 
# This tutorial assumes that the PPI data of interest have already been generated and saved as [HDF5 files](https://en.wikipedia.org/wiki/Hierarchical_Data_Format), with the data structure that DeepRank2 expects. This data can be generated using the [data_generation_ppi.ipynb](https://github.com/DeepRank/deeprank2/blob/main/tutorials/data_generation_ppi.ipynb) tutorial or downloaded from Zenodo at [this record address](https://zenodo.org/record/7997585). For more details on the data structure, please refer to the other tutorial, which also contains a detailed description of how the data is generated from PDB files.
# 
# This tutorial assumes also a basic knowledge of the [PyTorch](https://pytorch.org/) framework, on top of which the machine learning pipeline of DeepRank2 has been developed, for which many online tutorials exist.
# 

# ### Input data
# 
# If you have previously run `data_generation_ppi.ipynb` or `data_generation_srv.ipynb` notebook, then their output can be directly used as input for this tutorial.
# 
# Alternatively, preprocessed HDF5 files can be downloaded directly from Zenodo at [this record address](https://zenodo.org/record/7997585). To download the data used in this tutorial, please visit the link and download `data_processed.zip`. Unzip it, and save the `data_processed/` folder in the same directory as this notebook. The name and the location of the folder are optional but recommended, as they are the name and the location we will use to refer to the folder throughout the tutorial.
# 
# Note that the datasets contain only ~100 data points each, which is not enough to develop an impactful predictive model, and the scope of their use is indeed only demonstrative and informative for the users.
# 

# ## Utilities
# 

# ### Libraries
# 

# The libraries needed for this tutorial:
# 

# In[ ]:


import glob
import logging
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split

from deeprank2.dataset import GraphDataset
from deeprank2.neuralnets.cnn.model3d import CnnClassification
from deeprank2.neuralnets.gnn.vanilla_gnn import EGNNNetwork
from deeprank2.neuralnets.gnn.ginet_nocluster import GINet
from deeprank2.neuralnets.gnn.foutnet import FoutNet
from deeprank2.trainer import Trainer
from deeprank2.utils.exporters import HDF5OutputExporter

np.seterr(divide="ignore")
np.seterr(invalid="ignore")

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")

# ruff: noqa: PD901


# ### Paths and sets
# 
# The paths for reading the processed data:
# 

# In[ ]:

def check_hdf5_for_nan(file_path):
    with h5py.File(file_path, "r") as hdf5:
        def check_group(group, group_name=""):
            for key, item in group.items():
                full_path = f"{group_name}/{key}" if group_name else key
                if isinstance(item, h5py.Group):
                    check_group(item, full_path)
                elif isinstance(item, h5py.Dataset):
                    data = item[()]
                    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
                        if np.isnan(data).any():
                            print(f"? NaN found in dataset: {full_path} in file {file_path}")

data_type = "ppi"
level = "residue"
processed_data_path = os.path.join("data_processed_add_merge_8A_gen_all", data_type, level)
input_data_path = glob.glob(os.path.join(processed_data_path, "*.hdf5"))
output_path = os.path.join("data_processed_add_CDE_8A", data_type, level)  # for saving predictions results


# The `data_type` can be either "ppi" or "srv", depending on which application the user is most interested in. The `level` can be either "residue" or "atomic", and refers to the structural resolution, where each node either represents a single residue or a single atom from the molecular structure.
# 
# In this tutorial, we will use PPI residue-level data by default, but the same code can be applied to SRV or/and atomic-level data with no changes, apart from setting `data_type` and `level` parameters in the cell above.
# 

# A Pandas DataFrame containing data points' IDs and the binary target values can be defined:
# 

# In[ ]:


df_dict = {}
df_dict["entry"] = []
df_dict["target"] = []
for fname in input_data_path:
    with h5py.File(fname, "r") as hdf5:
        for mol in hdf5:
            target_value = float(hdf5[mol]["target_values"]["binary"][()])
            df_dict["entry"].append(mol)
            df_dict["target"].append(target_value)

df = pd.DataFrame(data=df_dict)
df.head()

#print("Check HDF5")
#for fname in input_data_path:
#    check_hdf5_for_nan(fname)

# As explained in [data_generation_ppi.ipynb](https://github.com/DeepRank/deeprank2/blob/main/tutorials/data_generation_ppi.ipynb), for each data point there are two targets: "BA" and "binary". The first represents the strength of the interaction between two molecules that bind reversibly (interact) in nM, while the second represents its binary mapping, being 0 (BA > 500 nM) a not-binding complex and 1 (BA <= 500 nM) binding one.
# 
# For SRVs, each data point has a single target, "binary", which is 0 if the SRV is considered benign, and 1 if it is pathogenic, as explained in [data_generation_srv.ipynb](https://github.com/DeepRank/deeprank-core/blob/main/tutorials/data_generation_srv.ipynb).
# 
# The pandas DataFrame `df` is used only to split data points into training, validation and test sets according to the "binary" target - using target stratification to keep the proportion of 0s and 1s constant among the different sets. Training and validation sets will be used during the training for updating the network weights, while the test set will be held out as an independent test and will be used later for the model evaluation.
# 

# In[ ]:


df_train, df_test = train_test_split(df, test_size=0.1, stratify=df.target, random_state=42)
df_train, df_valid = train_test_split(df_train, test_size=0.2, stratify=df_train.target, random_state=42)

print("Data statistics:\n")
print(f"Total samples: {len(df)}\n")
print(f"Training set: {len(df_train)} samples, {round(100*len(df_train)/len(df))}%")
print(f"\t- Class 0: {len(df_train[df_train.target == 0])} samples, {round(100*len(df_train[df_train.target == 0])/len(df_train))}%")
print(f"\t- Class 1: {len(df_train[df_train.target == 1])} samples, {round(100*len(df_train[df_train.target == 1])/len(df_train))}%")
print(f"Validation set: {len(df_valid)} samples, {round(100*len(df_valid)/len(df))}%")
print(f"\t- Class 0: {len(df_valid[df_valid.target == 0])} samples, {round(100*len(df_valid[df_valid.target == 0])/len(df_valid))}%")
print(f"\t- Class 1: {len(df_valid[df_valid.target == 1])} samples, {round(100*len(df_valid[df_valid.target == 1])/len(df_valid))}%")
print(f"Testing set: {len(df_test)} samples, {round(100*len(df_test)/len(df))}%")
print(f"\t- Class 0: {len(df_test[df_test.target == 0])} samples, {round(100*len(df_test[df_test.target == 0])/len(df_test))}%")
print(f"\t- Class 1: {len(df_test[df_test.target == 1])} samples, {round(100*len(df_test[df_test.target == 1])/len(df_test))}%")


# ## Classification example
# 
# A GNN and a CNN can be trained for a classification predictive task, which consists in predicting the "binary" target values.
# 

# ### GNN
# 

# #### GraphDataset
# 

# For training GNNs the user can create `GraphDataset` instances. This class inherits from `DeeprankDataset` class, which in turns inherits from `Dataset` [PyTorch geometric class](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html), a base class for creating graph datasets.
# 
# A few notes about `GraphDataset` parameters:
# 
# - By default, all features contained in the HDF5 files are used, but the user can specify `node_features` and `edge_features` in `GraphDataset` if not all of them are needed. See the [docs](https://deeprank2.readthedocs.io/en/latest/features.html) for more details about all the possible pre-implemented features.
# - For regression, `task` should be set to `regress` and the `target` to `BA`, which is a continuous variable and therefore suitable for regression tasks.
# - For the `GraphDataset` class it is possible to define a dictionary to indicate which transformations to apply to the features, being the transformations lambda functions and/or standardization.
#   - If the `standardize` key is `True`, standardization is applied after transformation. Standardization consists in applying the following formula on each feature's value: ${x' = \frac{x - \mu}{\sigma}}$, being ${\mu}$ the mean and ${\sigma}$ the standard deviation. Standardization is a scaling method where the values are centered around mean with a unit standard deviation.
#   - The transformation to apply can be speficied as a lambda function as a value of the key `transform`, which defaults to `None`.
#   - Since in the provided example standardization is applied, the training features' means and standard deviations need to be used for scaling validation and test sets. For doing so, `train_source` parameter is used. When `train_source` parameter is set, it will be used to scale the validation/testing sets. You need to pass `features_transform` to the training dataset only, since in other cases it will be ignored and only the one of `train_source` will be considered.
#   - Note that transformations have not currently been implemented for the `GridDataset` class.
#   - In the example below a logarithmic transformation and then the standardization are applied to all the features. It is also possible to use specific features as keys for indicating that transformation and/or standardization need to be apply to few features only.
# 

# In[ ]:


target = "binary"
task = "classif"
fol = "noNodeS"
node_features = ["res_type", "polarity", "res_size", "res_mass", "res_charge", "res_pI"]
edge_features = ["distance", "same_chain", "covalent", "electrostatic", "vanderwaals"]
features_transform = {"all": {"transform": lambda x: np.cbrt(x), "standardize": True}}

print(node_features)
print(edge_features)

print("Loading training data...")
dataset_train = GraphDataset(
    hdf5_path=input_data_path,
    subset=list(df_train.entry),  # selects only data points with ids in df_train.entry
    node_features=node_features,
    edge_features=edge_features,
    features_transform=features_transform,
    target=target,
    task=task,
)
print("\nLoading validation data...")
dataset_val = GraphDataset(
    hdf5_path=input_data_path,
    subset=list(df_valid.entry),  # selects only data points with ids in df_valid.entry
    train_source=dataset_train,
)
print("\nLoading test data...")
dataset_test = GraphDataset(
    hdf5_path=input_data_path,
    subset=list(df_test.entry),  # selects only data points with ids in df_test.entry
    train_source=dataset_train,
)

#for i, data in enumerate(dataset_train):
#    if torch.isnan(data.x).any():
#        print(f"? NaN found in dataset_train entry {i}!")

# #### Trainer
# 
# The class `Trainer` implements training, validation and testing of PyTorch-based neural networks.
# 

# A few notes about `Trainer` parameters:
# 
# - `neuralnet` can be any neural network class that inherits from `torch.nn.Module`, and it shouldn't be specific to regression or classification in terms of output shape. The `Trainer` class takes care of formatting the output shape according to the task. This tutorial uses a simple network, `VanillaNetwork` (implemented in `deeprank2.neuralnets.gnn.vanilla_gnn`). All GNN architectures already implemented in the pakcage can be found [here](https://github.com/DeepRank/deeprank-core/tree/main/deeprank2/neuralnets/gnn) and can be used for training or as a basis for implementing new ones.
# - `class_weights` is used for classification tasks only and assigns class weights based on the training dataset content to account for any potential inbalance between the classes. In this case the dataset is balanced (50% 0 and 50% 1), so it is not necessary to use it. It defaults to False.
# - `cuda` and `ngpu` are used for indicating whether to use CUDA and how many GPUs. By default, CUDA is not used and `ngpu` is 0.
# - The user can specify a deeprank2 exporter or a custom one in `output_exporters` parameter, together with the path where to save the results. Exporters are used for storing predictions information collected later on during training and testing. Later the results saved by `HDF5OutputExporter` will be read and evaluated.
# 

# ##### Training
# 

# In[ ]:


trainer = Trainer(
    neuralnet=EGNNNetwork,
    dataset_train=dataset_train,
    dataset_val=dataset_val,
    dataset_test=dataset_test,
    output_exporters=[HDF5OutputExporter(os.path.join(output_path, f"egnn_{fol}_{task}"))],
)


# The default optimizer is `torch.optim.Adam`. It is possible to specify optimizer's parameters or to use another PyTorch optimizer object:
# 

# In[ ]:


optimizer = torch.optim.SGD
lr = 1e-3
weight_decay = 0.001

trainer.configure_optimizers(optimizer, lr, weight_decay)


# The default loss function for classification is `torch.nn.CrossEntropyLoss` and for regression it is `torch.nn.MSELoss`. It is also possible to set some other PyTorch loss functions by using `Trainer.set_lossfunction` method, although not all are currently implemented.
# 
# Then the model can be trained using the `train()` method of the `Trainer` class.
# 
# A few notes about `train()` method parameters:
# 
# - `earlystop_patience`, `earlystop_maxgap` and `min_epoch` are used for controlling early stopping logic. `earlystop_patience` indicates the number of epochs after which the training ends if the validation loss does not improve. `earlystop_maxgap` indicated the maximum difference allowed between validation and training loss, and `min_epoch` is the minimum number of epochs to be reached before evaluating `maxgap`.
# - If `validate` is set to `True`, validation is performed on an independent dataset, which has been called `dataset_val` few cells above. If set to `False`, validation is performed on the training dataset itself (not recommended).
# - `num_workers` can be set for indicating how many subprocesses to use for data loading. The default is 0 and it means that the data will be loaded in the main process.
# 

# In[ ]:


epochs = 100
batch_size = 8
earlystop_patience = 5
earlystop_maxgap = 0.1
min_epoch = 10

trainer.train(
    nepoch=epochs,
    batch_size=batch_size,
    earlystop_patience=earlystop_patience,
    earlystop_maxgap=earlystop_maxgap,
    min_epoch=min_epoch,
    validate=True,
    filename=os.path.join(output_path, f"egnn_{fol}_{task}", "model.pth.tar"),
)

epoch = trainer.epoch_saved_model
print(f"Model saved at epoch {epoch}")
pytorch_total_params = sum(p.numel() for p in trainer.model.parameters())
print(f"Total # of parameters: {pytorch_total_params}")
pytorch_trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
print(f"Total # of trainable parameters: {pytorch_trainable_params}")


# ##### Testing
# 
# And the trained model can be tested on `dataset_test`:
# 

# In[ ]:


trainer.test()


# ##### Results visualization
# 
# Finally, the results saved by `HDF5OutputExporter` can be inspected, which can be found in the `data/ppi/egnn_classif/` folder in the form of an HDF5 file, `output_exporter.hdf5`. Note that the folder contains the saved pre-trained model as well.
# 
# `output_exporter.hdf5` contains [HDF5 Groups](https://docs.h5py.org/en/stable/high/group.html) which refer to each phase, e.g. training and testing if both are run, only one of them otherwise. Training phase includes validation results as well. This HDF5 file can be read as a Pandas Dataframe:
# 

# In[ ]:


output_train = pd.read_hdf(os.path.join(output_path, f"egnn_{fol}_{task}", "output_exporter.hdf5"), key="training")
output_test = pd.read_hdf(os.path.join(output_path, f"egnn_{fol}_{task}", "output_exporter.hdf5"), key="testing")
output_train.head()


# The dataframes contain `phase`, `epoch`, `entry`, `output`, `target`, and `loss` columns, and can be easily used to visualize the results.
# 
# For classification tasks, the `output` column contains a list of probabilities that each class occurs, and each list sums to 1 (for more details, please see documentation on the [softmax function](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)). Note that the order of the classes in the list depends on the `classes` attribute of the DeeprankDataset instances. For classification tasks, if `classes` is not specified (as in this example case), it is defaulted to [0, 1].
# 
# The loss across the epochs can be plotted for the training and the validation sets:
# 

# In[ ]:


fig = px.line(output_train, x="epoch", y="loss", color="phase", markers=True)

fig.add_vline(x=trainer.epoch_saved_model, line_width=3, line_dash="dash", line_color="green")

fig.update_layout(
    xaxis_title="Epoch #",
    yaxis_title="Loss",
    title="Loss vs epochs - GNN training",
    width=700,
    height=400,
)


# And now a few metrics of interest for classification tasks can be printed out: the [area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) (AUC), and for a threshold of 0.5 the [precision, recall, accuracy and f1 score](https://en.wikipedia.org/wiki/Precision_and_recall#Definition).
# 

# In[ ]:


threshold = 0.5
df = pd.concat([output_train, output_test])
df_plot = df[(df.epoch == trainer.epoch_saved_model) | ((df.epoch == trainer.epoch_saved_model) & (df.phase == "testing"))]

for dataset in ["training", "validation", "testing"]:
    df_plot_phase = df_plot[(df_plot.phase == dataset)]
    y_true = df_plot_phase.target
    y_score = np.array(df_plot_phase.output.tolist())[:, 1]

    print(f"\nMetrics for {dataset}:")
    fpr_roc, tpr_roc, thr_roc = roc_curve(y_true, y_score)
    auc_score = auc(fpr_roc, tpr_roc)
    print(f"AUC: {round(auc_score, 2)}")
    print(f"Considering a threshold of {threshold}")
    y_pred = (y_score > threshold) * 1
    print(f"- Precision: {round(precision_score(y_true, y_pred), 2)}")
    print(f"- Recall: {round(recall_score(y_true, y_pred), 2)}")
    print(f"- Accuracy: {round(accuracy_score(y_true, y_pred), 2)}")
    print(f"- F1: {round(f1_score(y_true, y_pred), 2)}")


# Note that the poor performance of this network is due to the small number of datapoints used in this tutorial. For a more reliable network we suggest using a number of data points on the order of at least tens of thousands.

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
from sklearn.model_selection import train_test_split, KFold

from deeprank2.dataset import GraphDataset
from deeprank2.neuralnets.cnn.model3d import CnnClassification
from deeprank2.neuralnets.gnn.vanilla_gnn import VanillaNetwork
from deeprank2.trainer import Trainer
from deeprank2.utils.exporters import HDF5OutputExporter

np.seterr(divide="ignore")
np.seterr(invalid="ignore")

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")

data_type = "ppi"
level = "residue"
processed_data_path = os.path.join("data_processed_add2_merge_8A", data_type, level)
input_data_path = glob.glob(os.path.join(processed_data_path, "*.hdf5"))
output_path = os.path.join("data_processed_add2_merge_8A_tmp", data_type, level)

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

target = "binary"
task = "classif"
node_features = ["res_type", "polarity", "res_size", "res_mass", "res_charge", "res_pI", "res_depth", "hse", "sasa", "bsa"]
edge_features = ["distance", "same_chain", "covalent"]
features_transform = {"all": {"transform": lambda x: np.cbrt(x), "standardize": True}}

optimizer = torch.optim.SGD
lr = 1e-3
weight_decay = 0.001

epochs = 100
batch_size = 8
earlystop_patience = 5
earlystop_maxgap = 0.1
min_epoch = 10
threshold = 0.5

cv_results = []

k_folds = 10
df_train_valid, df_test = train_test_split(df, test_size=0.1, stratify=df["target"], random_state=42)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_valid)):
    print(f"\nFold {fold + 1}/{k_folds}")

    df_fold_train = df_train_valid.iloc[train_idx]
    df_fold_val = df_train_valid.iloc[val_idx]

    print("Loading training data...")
    dataset_train = GraphDataset(
        hdf5_path=input_data_path,
        subset=list(df_fold_train.entry),
        node_features=node_features,
        edge_features=edge_features,
        features_transform=features_transform,
        target=target,
        task=task,
    )
    print("\nLoading validation data...")
    dataset_val = GraphDataset(
        hdf5_path=input_data_path,
        subset=list(df_fold_val.entry),
        train_source=dataset_train,
    )
    print("\nLoading test data...")
    dataset_test = GraphDataset(
        hdf5_path=input_data_path,
        subset=list(df_test.entry),
        train_source=dataset_train,
    )
    
    trainer = Trainer(
      neuralnet=VanillaNetwork,
      dataset_train=dataset_train,
      dataset_val=dataset_val,
      dataset_test=dataset_test,
      output_exporters=[HDF5OutputExporter(os.path.join(output_path, f"gnn_{task}"))],
    )

    trainer.configure_optimizers(optimizer, lr, weight_decay)

    trainer.train(
        nepoch=epochs,
        batch_size=batch_size,
        earlystop_patience=earlystop_patience,
        earlystop_maxgap=earlystop_maxgap,
        min_epoch=min_epoch,
        validate=True,
        filename=os.path.join(output_path, f"gnn_{task}", "model.pth.tar"),
    )

    trainer.test()

    output_val = pd.read_hdf(os.path.join(output_path, f"gnn_classif", "output_exporter.hdf5"), key="testing")

    y_true = output_val["target"].values
    y_score = np.array(output_val["output"].tolist())[:, 1]
    y_pred = (y_score > threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)

    metrics = {
        "AUC": auc_score,
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }
    cv_results.append(metrics)

print("Cross-Validation Results:")
avg_metrics = {key: np.mean([m[key] for m in cv_results]) for key in cv_results[0]}
for metric, value in avg_metrics.items():
    print(f"{metric}: {round(value, 2)}")
    

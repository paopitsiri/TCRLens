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
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from deeprank2.dataset import GraphDataset
from deeprank2.neuralnets.cnn.model3d import CnnClassification
from deeprank2.neuralnets.gnn.vanilla_gnn import EGNNNetwork
from deeprank2.trainer import Trainer
from deeprank2.utils.exporters import HDF5OutputExporter

np.seterr(divide="ignore")
np.seterr(invalid="ignore")

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")

data_type = "ppi"
level = "residue"
processed_data_path = os.path.join("data_processed_add_CDE_8A", data_type, level)
input_data_path = glob.glob(os.path.join(processed_data_path, "*.hdf5"))
output_path = os.path.join("data_processed_add_merge_8A_gen_all", data_type, level)

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
fol = "rkf_noTypeS"
node_features = ["res_type", "polarity", "res_size", "res_mass", "res_charge", "res_pI"]
edge_features = ["distance", "same_chain", "covalent", "electrostatic", "vanderwaals"]
features_transform = {"all": {"transform": lambda x: np.cbrt(x), "standardize": True}}

print(node_features)
print(edge_features)
print(fol)

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

num_splits = 10
num_repeats = 2

kf_test = KFold(n_splits=num_repeats, shuffle=True, random_state=42)
for repeat, (train_valid_idx, test_idx) in enumerate(kf_test.split(df)):
    
    print(f"\nRepeated K-Fold Round {repeat + 1}/{num_repeats}")
    
    df_train_valid = df.iloc[train_valid_idx]
    df_test = df.iloc[test_idx]

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=repeat)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_valid)):
        print(f"\nFold {fold + 1}/{num_splits} in Round {repeat + 1}")

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
          neuralnet=EGNNNetwork,
          dataset_train=dataset_train,
          dataset_val=dataset_val,
          dataset_test=dataset_test,
          output_exporters=[HDF5OutputExporter(os.path.join(output_path, f"egnn_{fol}_{task}"))],
        )
        
        trainer.configure_optimizers(optimizer, lr, weight_decay)

        trainer.train(
            nepoch=epochs,
            batch_size=batch_size,
            earlystop_patience=earlystop_patience,
            earlystop_maxgap=earlystop_maxgap,
            min_epoch=min_epoch,
            validate=True,
            filename=os.path.join(output_path, f"egnn_{fol}_{task}", f"model_repeat{repeat}.pth.tar"),
        )

        trainer.test()
        
        export_dir = os.path.join(output_path, f"egnn_{fol}_{task}")
        default_export_file = os.path.join(export_dir, "output_exporter.hdf5")
        renamed_export_file = os.path.join(export_dir, f"output_exporter_repeat{repeat}.hdf5")
        if os.path.exists(default_export_file):
            shutil.move(default_export_file, renamed_export_file)
            print(f"? Renamed output to: {renamed_export_file}")
        else:
            print(f"?? Could not find expected output: {default_export_file}")

        output_val = pd.read_hdf(os.path.join(output_path, f"egnn_{fol}_{task}", f"output_exporter_repeat{repeat}.hdf5"), key="testing")

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
    
df_cv = pd.DataFrame(cv_results)
df_cv.to_csv(os.path.join(output_path, f"egnn_{fol}_{task}", "cv_results_summary.csv"), index=False)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_cv[["AUC", "Precision", "Recall", "Accuracy", "F1"]])
plt.title("Distribution of CV Metrics")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f"egnn_{fol}_{task}", "cv_metric_boxplot.png"))
plt.close()
print("Saved boxplot: cv_metric_boxplot.png")

all_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

kf_test = KFold(n_splits=num_repeats, shuffle=True, random_state=42)
for repeat, (_, test_idx) in enumerate(kf_test.split(df)):
    df_test = df.iloc[test_idx]
    output_val = pd.read_hdf(os.path.join(output_path, f"egnn_{fol}_{task}", f"output_exporter_repeat{repeat}.hdf5"), key="testing")
    
    y_true = output_val["target"].values
    y_score = np.array(output_val["output"].tolist())[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    tpr_interp = np.interp(all_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)
    aucs.append(auc(fpr, tpr))

mean_tpr = np.mean(tprs, axis=0)
std_tpr = np.std(tprs, axis=0)
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

plt.figure(figsize=(7, 6))
plt.plot(all_fpr, mean_tpr, color="blue", label=f"Mean ROC (AUC = {mean_auc:.2f} ? {std_auc:.2f})", lw=2)
plt.fill_between(all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.3, color="blue", label="?1 std dev")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve from Repeated K-Fold")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f"egnn_{fol}_{task}", "roc_curve_avg.png"))
plt.close()
print("Saved ROC curve: roc_curve_avg.png")

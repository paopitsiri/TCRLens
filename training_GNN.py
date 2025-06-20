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
from deeprank2.neuralnets.gnn.vanilla_gnn import VanillaNetwork
from deeprank2.neuralnets.gnn.ginet_nocluster import GINet
from deeprank2.neuralnets.gnn.foutnet import FoutNet
from deeprank2.trainer import Trainer
from deeprank2.utils.exporters import HDF5OutputExporter

np.seterr(divide="ignore")
np.seterr(invalid="ignore")

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")


data_type = "ppi"
level = "residue"
processed_data_path = os.path.join("data_processed_add_merge_8A_gen_all", data_type, level)
input_data_path = glob.glob(os.path.join(processed_data_path, "*.hdf5"))
output_path = os.path.join("data_processed_add_merge_8A_gen_all", data_type, level)  # for saving predictions results



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


target = "binary"
task = "classif"
fol = "noNodeS"
node_features = ["res_type", "polarity", "res_size", "res_mass", "res_charge", "res_pI"]
edge_features = ["distance", "same_chain", "covalent", "electrostatic", "vanderwaals"]
features_transform = {"all": {"transform": lambda x: np.cbrt(x), "standardize": True}}

print(fol)
print(node_features)
print(edge_features)

print("Loading training data...")
dataset_train = GraphDataset(
    hdf5_path=input_data_path,
    subset=list(df_train.entry),
    node_features=node_features,
    edge_features=edge_features,
    features_transform=features_transform,
    target=target,
    task=task,
)
print("\nLoading validation data...")
dataset_val = GraphDataset(
    hdf5_path=input_data_path,
    subset=list(df_valid.entry),
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
    output_exporters=[HDF5OutputExporter(os.path.join(output_path, f"gnn_{fol}_{task}"))],
)


optimizer = torch.optim.SGD
lr = 1e-3
weight_decay = 0.001

trainer.configure_optimizers(optimizer, lr, weight_decay)



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
    filename=os.path.join(output_path, f"gnn_{fol}_{task}", "model.pth.tar"),
)

epoch = trainer.epoch_saved_model
print(f"Model saved at epoch {epoch}")
pytorch_total_params = sum(p.numel() for p in trainer.model.parameters())
print(f"Total # of parameters: {pytorch_total_params}")
pytorch_trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
print(f"Total # of trainable parameters: {pytorch_trainable_params}")


trainer.test()


output_train = pd.read_hdf(os.path.join(output_path, f"gnn_{fol}_{task}", "output_exporter.hdf5"), key="training")
output_test = pd.read_hdf(os.path.join(output_path, f"gnn_{fol}_{task}", "output_exporter.hdf5"), key="testing")
output_train.head()


fig = px.line(output_train, x="epoch", y="loss", color="phase", markers=True)

fig.add_vline(x=trainer.epoch_saved_model, line_width=3, line_dash="dash", line_color="green")

fig.update_layout(
    xaxis_title="Epoch #",
    yaxis_title="Loss",
    title="Loss vs epochs - GNN training",
    width=700,
    height=400,
)


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

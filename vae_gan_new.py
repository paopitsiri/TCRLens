# train.py

import pandas as pd
import warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import h5py
import random
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import (
    load_graphs_from_hdf5,
    normalize_features_in_graphs,
    inverse_generated_features,
    save_generated_graphs_to_hdf5,
    compute_feature_statistics
)

# === Metric Functions ===
def compare_degree_distribution(real_graphs, gen_graphs):
    real_degrees = []
    gen_degrees = []
    for g in real_graphs:
        real_degrees.extend(g.in_degrees().tolist())
    for g in gen_graphs:
        gen_degrees.extend(g.in_degrees().tolist())
    plt.hist(real_degrees, bins=20, alpha=0.5, label='Real')
    plt.hist(gen_degrees, bins=20, alpha=0.5, label='Generated')
    plt.legend()
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig("degree_distribution.png")
    plt.close()

def kl_divergence(p, q, eps=1e-10):
    p = np.array(p) + eps
    q = np.array(q) + eps
    return np.sum(p * np.log(p / q))

def visualize_latent_space(encoder, graphs):
    latent = []
    sizes = []
    for g in graphs:
        x = g.ndata['features']
        e = g.edata['features']
        mu, _ = encoder(g, x, e)
        latent.append(mu.detach().numpy())
        sizes.append(g.num_nodes())
    latent = np.vstack(latent)
    z = TSNE(n_components=2).fit_transform(latent)
    sizes = np.array(sizes)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=sizes, cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Nodes')
    plt.title("Latent Space Visualization (colored by number of nodes)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.savefig("latent_space_colored.png")
    plt.close()


class GraphVAEEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, latent_dim):
        super(GraphVAEEncoder, self).__init__()
        self.gc1 = dglnn.GraphConv(in_dim, hidden_dim, activation=F.relu, allow_zero_in_degree=True)
        self.gc2 = dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu, allow_zero_in_degree=True)
        self.edge_fc = nn.Linear(edge_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, g, x, edge_feat):
    
        h = self.gc1(g, x)
        h = self.gc2(g, h)

        if edge_feat.dim() == 1:
            edge_feat = edge_feat.unsqueeze(1)

        edge_h = self.edge_fc(edge_feat)
        g.edata['features'] = edge_h
        g.ndata['h'] = h
        h_graph = dgl.mean_nodes(g, 'h')
        
        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        return mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class GraphVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_nodes, feature_dim, edge_dim):
        super(GraphVAEDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_adj = nn.Linear(hidden_dim, num_nodes * num_nodes)
        self.fc_feat = nn.Linear(hidden_dim, num_nodes * feature_dim)
        self.fc_edge_feat = nn.Linear(hidden_dim, num_nodes * num_nodes * edge_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))

        adj_flat = torch.sigmoid(self.fc_adj(h))
        node_feat_flat = self.fc_feat(h)
        edge_feat_flat = self.fc_edge_feat(h)

        adj_matrix = adj_flat.view(-1, self.num_nodes, self.num_nodes)
        node_features = node_feat_flat.view(-1, self.num_nodes, self.feature_dim)
        edge_features = edge_feat_flat.view(-1, self.num_nodes, self.num_nodes, self.edge_dim)

        return adj_matrix.squeeze(0), node_features.squeeze(0), edge_features.squeeze(0)


# ===== Discriminator =====
class GraphDiscriminator(nn.Module):
    def __init__(self, num_nodes, feature_dim, edge_dim):
        super().__init__()
        input_dim = num_nodes * num_nodes + num_nodes * feature_dim + num_nodes * num_nodes * edge_dim
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def extract_features(self, adj, feat, edge_feat):
        x = torch.cat([
            adj.view(-1),
            feat.view(-1),
            edge_feat.view(-1)
        ], dim=0)
        x = x + 0.1 * torch.randn_like(x)
        x = self.dropout(x)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return h2

    def forward(self, adj, feat, edge_feat):
        h2 = self.extract_features(adj, feat, edge_feat)
        return self.fc3(h2)  # Raw score (no Sigmoid)
                
        
def vae_loss(recon_adj, real_adj, recon_feat, real_feat, recon_edge, real_edge, mu, logvar):
    
    real_adj = real_adj.squeeze(0)
    real_feat = real_feat.squeeze(0)
    real_edge = real_edge.squeeze(0)
    
    loss_adj = F.mse_loss(recon_adj, real_adj)
    loss_feat = F.mse_loss(recon_feat, real_feat)
    loss_edge = F.mse_loss(recon_edge, real_edge)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss_adj + loss_feat + loss_edge + kl
 
def add_noise(x, scale=0.01):
    return x + scale * torch.randn_like(x) 
 
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score - self.delta:  # we assume lower is better
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

  
selected_node_keys = ["res_type", "polarity", "res_size", "res_mass", "res_charge", "res_pI", "res_depth", "hse", "sasa", "bsa", "hb_acceptors", "hb_donors", "_position"]
selected_edge_keys = ["distance", "same_chain", "covalent", "electrostatic", "vanderwaals", "source_file"]
#selected_node_keys = ["res_type", "polarity", "res_size", "res_mass", "res_charge"]
#selected_edge_keys = ["distance", "same_chain"]

print("Load Data")
graphs, feature_dim, edge_dim, node_dims, edge_dims = load_graphs_from_hdf5(
    "Gaga.hdf5",
    node_feature_keys=selected_node_keys,
    edge_feature_keys=selected_edge_keys,
    target_binary_value=0,
    verbose=True
)
print("Load Data Finish")

num_nodes_list = [g.num_nodes() for g in graphs]

node_scalers, edge_scalers = normalize_features_in_graphs(
    graphs,
    node_dims=node_dims,
    edge_dims=edge_dims,
)

latent_dim = 16
hidden_dim = 64
lambda_gan = 1.0
lambda_fm = 0.5
clip_grad = 5.0
clip_value = 0.01
encoder = GraphVAEEncoder(feature_dim, edge_dim, hidden_dim, latent_dim)
#decoder = GraphVAEDecoder(latent_dim, hidden_dim, num_nodes, feature_dim, edge_dim)

decoders, discriminators, optimizers, disc_optimizers = {}, {}, {}, {}
for n in set(num_nodes_list):
    decoders[n] = GraphVAEDecoder(latent_dim, hidden_dim, n, feature_dim, edge_dim)
    discriminators[n] = GraphDiscriminator(n, feature_dim, edge_dim)
    optimizers[n] = torch.optim.Adam(list(encoder.parameters()) + list(decoders[n].parameters()), lr=1e-3)
    disc_optimizers[n] = torch.optim.Adam(discriminators[n].parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=0.0)

#bce_loss = nn.BCELoss()

# Training Loop
epochs = 100
log_interval=1
step = 0
n_critic = 7

early_stopper = EarlyStopping(patience=10)
best_loss = float("inf")
save_path = "checkpoints/best_model_0_Gaga.pt"
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    
    total_loss = 0.0
    total_vae = 0.0
    total_gan = 0.0
    total_disc = 0.0
    for g in graphs:
    
        g_clone = g.clone()
    
        decoder = decoders[g.num_nodes()]
        discriminator = discriminators[g.num_nodes()]
        optimizer = optimizers[g.num_nodes()]
        disc_optimizer = disc_optimizers[g.num_nodes()]
    
        node_feat = g_clone.ndata['features']
        edge_feat = g_clone.edata['features']              

        mu, logvar = encoder(g_clone, node_feat, edge_feat)
        z = reparameterize(mu, logvar)
        recon_adj, recon_feat, recon_edge = decoder(z)

        # === Prepare real ===
        real_adj = g.adjacency_matrix().to_dense().unsqueeze(0)
        real_feat = node_feat.unsqueeze(0)
        edge_tensor = torch.zeros(g.num_nodes(), g.num_nodes(), edge_dim)
        src, dst = g.edges()
        edge_tensor[src, dst] = edge_feat
        real_edge = edge_tensor.unsqueeze(0)

        # === 1. Train Discriminator (WGAN)
        #real_score = discriminator(real_adj.squeeze(0), real_feat.squeeze(0), real_edge.squeeze(0))
        real_score = discriminator(add_noise(real_adj.squeeze(0)),
                           add_noise(real_feat.squeeze(0)),
                           add_noise(real_edge.squeeze(0)))
        fake_score = discriminator(recon_adj.detach(), recon_feat.detach(), recon_edge.detach())
        loss_D = fake_score.mean() - real_score.mean()

        if step % n_critic == 0:
            disc_optimizer.zero_grad()
            loss_D.backward()
            disc_optimizer.step()

            # Weight clipping
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # === 2. Train Generator (VAE)
        loss_vae = vae_loss(recon_adj, real_adj, recon_feat, real_feat, recon_edge, real_edge, mu, logvar)
        fake_score_for_G = discriminator(recon_adj, recon_feat, recon_edge)
        gan_loss = -fake_score_for_G.mean()
        
        # === Feature Matching Loss (optional)
        if lambda_fm > 0.0:
            with torch.no_grad():
                real_feat_D = discriminator.extract_features(real_adj.squeeze(0), real_feat.squeeze(0), real_edge.squeeze(0))
            fake_feat_D = discriminator.extract_features(recon_adj, recon_feat, recon_edge)
            fm_loss = F.mse_loss(fake_feat_D, real_feat_D)

            gan_loss += lambda_fm * fm_loss

        total_loss_batch = loss_vae + lambda_gan * gan_loss
        optimizer.zero_grad()
        total_loss_batch.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_grad)
        optimizer.step()

        total_loss += total_loss_batch.item()
        total_vae += loss_vae.item()
        total_gan += gan_loss.item()
        total_disc += loss_D.item()
        step += 1


    if epoch % log_interval == 0:
        print(f"Epoch {epoch:3d} | Total: {total_loss/len(graphs):.4f} | "
            f"VAE: {total_vae/len(graphs):.4f} | GAN: {total_gan/len(graphs):.4f} | "
            f"D: {total_disc/len(graphs):.4f}")
            
    avg_epoch_loss = total_loss / len(graphs)

    # Save model if it's the best
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        checkpoint = {
            'encoder': encoder.state_dict(),
            'decoder': {k: d.state_dict() for k, d in decoders.items()},
            'discriminator': {k: d.state_dict() for k, d in discriminators.items()},
            'optimizer': {k: o.state_dict() for k, o in optimizers.items()},
            'disc_optimizer': {k: o.state_dict() for k, o in disc_optimizers.items()},
            'epoch': epoch
        }
        torch.save(checkpoint, save_path)
        print(f"?? Saved new best model at epoch {epoch:3d} | Loss: {avg_epoch_loss:.4f}")

    # Check early stopping
    early_stopper(avg_epoch_loss)
    if early_stopper.early_stop:
        print("?? Early stopping triggered.")
        break


#checkpoint = torch.load("checkpoints/best_model.pt")
#encoder.load_state_dict(checkpoint['encoder'])
#for k in decoders:
#    decoders[k].load_state_dict(checkpoint['decoder'][k])
#for k in discriminators:
#    discriminators[k].load_state_dict(checkpoint['discriminator'][k])

num_generate = 200
generated_graphs = []

for _ in range(num_generate):

    n = random.choice(num_nodes_list)
    z_sample = torch.randn(1, latent_dim)

    decoder = decoders[n]
    gen_adj, gen_feat, gen_edge = decoder(z_sample)

    adj = gen_adj
    feat = gen_feat
    edge = gen_edge

    adj_thresh = (adj > 0.5)
    src, dst = torch.nonzero(adj_thresh, as_tuple=True)

    gen_graph = dgl.graph((src, dst), num_nodes=n)
    gen_graph.ndata['features'] = feat
    gen_graph.edata['features'] = edge[src, dst]

    generated_graphs.append(gen_graph)

    print(f"? Generated graph with {n} nodes and {gen_graph.num_edges()} edges")

for g in generated_graphs:
    inverse_generated_features(
        g,
        node_scalers=node_scalers,
        edge_scalers=edge_scalers,
        node_dims=node_dims,
        edge_dims=edge_dims,
    )

save_generated_graphs_to_hdf5(
    generated_graphs,
    "generated_Gaga_0.hdf5",
    node_dims=node_dims,
    edge_dims=edge_dims,
    binary_value=0
)

#compute_feature_statistics(graphs, node_dims, edge_dims, is_generated=False)
#compute_feature_statistics(generated_graphs, node_dims, edge_dims, is_generated=True)

# === Evaluate Generated Graphs ===
#print("\n=== Evaluation ===")
#compare_degree_distribution(graphs, generated_graphs)
#visualize_latent_space(encoder, graphs)
#print("Saved degree_distribution.png and latent_space.png")

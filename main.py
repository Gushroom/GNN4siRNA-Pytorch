import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv
from sklearn.model_selection import KFold
import pandas as pd
from torchviz import make_dot

# Load params from the params.py file
import params

###############################################
# Data preparation
###############################################

# Load data into dataframes
siRNA_df = pd.read_csv(params.sirna_kmer_file, header=None).set_index(0)
mRNA_df = pd.read_csv(params.mrna_kmer_file, header=None).set_index(0)
thermo_feats_df = pd.read_csv(params.sirna_target_thermo_file, header=None)
siRNA_efficacy_df = pd.read_csv(params.sirna_efficacy_file)

# print the first 5 rows of each dataframe
# print(siRNA_df.head())
# print(mRNA_df.head())
# print(thermo_feats_df.head())
# print(siRNA_efficacy_df.head())

# drop the first two columns of the thermo_feats_df dataframe
interaction_feats_df = thermo_feats_df.drop([0, 1], axis=1)
# print the first 5 rows of the interaction_df dataframe
# print(interaction_feats_df.head())
interaction_connections_df = thermo_feats_df[[0, 1]]

# PyTorch expects the index to be a numerical
# make a mapping from name of rna to integers
siRNA_name_to_int = {name: i for i, name in enumerate(siRNA_df.index)}
mRNA_name_to_int = {name: i for i, name in enumerate(mRNA_df.index)}
# check the mapping
# print(siRNA_name_to_int)
# print(mRNA_name_to_int)

# create a new column for each unique interaction
interaction_connections_df['interaction'] = interaction_connections_df[0].astype(str) + '_' + interaction_connections_df[1].astype(str)
# print the first 5 rows of the interaction_connections_df dataframe
# print(interaction_connections_df.head())

# create a new dataframe with the integer index
# rename every entry in the first column of the siRNA_df dataframe with the integer index
# rename every entry in the second column of the mRNA_df dataframe with the integer index
mapped_siRNA_efficacy_df = siRNA_efficacy_df.copy()
mapped_siRNA_efficacy_df['siRNA'] = siRNA_efficacy_df['siRNA'].map(siRNA_name_to_int)
mapped_siRNA_efficacy_df['mRNA'] = siRNA_efficacy_df['mRNA'].map(mRNA_name_to_int)
# print the first 5 rows of the mapped_siRNA_efficacy_df dataframe
# print(mapped_siRNA_efficacy_df.head())
mapped_interaction_connections_df = interaction_connections_df.copy()
mapped_interaction_connections_df[0] = interaction_connections_df[0].map(siRNA_name_to_int)
mapped_interaction_connections_df[1] = interaction_connections_df[1].map(mRNA_name_to_int)
# print the first 5 rows of the mapped_interaction_connections_df dataframe
# print(mapped_interaction_connections_df.head())

# create a map for each unique interaction
interaction_name_to_int = {name: i for i, name in enumerate(mapped_interaction_connections_df['interaction'].unique())}
mapped_interaction_connections_df['interaction'] = mapped_interaction_connections_df['interaction'].map(interaction_name_to_int)
# print the first 5 rows of the mapped_interaction_connections_df dataframe
# rename the columns of the mapped_interaction_connections_df dataframe
mapped_interaction_connections_df.columns = ['siRNA', 'mRNA', 'interaction']
# print(mapped_interaction_connections_df.head())

data = HeteroData()

data['siRNA'].x = torch.tensor(siRNA_df.values, dtype=torch.float)
# show the shape of the siRNA node features
# print(data['siRNA'].x.shape) # (2816, 64)
data['mRNA'].x = torch.tensor(mRNA_df.values, dtype=torch.float)
# show the shape of the mRNA node features
# print(data['mRNA'].x.shape) # (44, 256)
data['interaction'].x = torch.tensor(interaction_feats_df.values, dtype=torch.float)
# show the shape of the interaction node features
# print(data['interaction'].x.shape) # (2816, 22)
data['interaction'].y = torch.tensor(mapped_siRNA_efficacy_df['efficacy'].values, dtype=torch.float)
# show the shape of the interaction node labels
# print(data['interaction'].y.shape) # (2816, )
# Edge: siRNA -> interaction
siRNA_to_interaction_edges = torch.tensor(
    mapped_interaction_connections_df[['siRNA', 'interaction']].values, dtype=torch.long
).t().contiguous()
data['siRNA', 'to', 'interaction'].edge_index = siRNA_to_interaction_edges
# show the shape of the interaction node edges
# print(data['siRNA', 'to', 'interaction'].edge_index.shape) # (2, 2816)
# Edge: mRNA -> interaction
mRNA_to_interaction_edges = torch.tensor(
    mapped_interaction_connections_df[['mRNA', 'interaction']].values, dtype=torch.long
).t().contiguous() 
data['mRNA', 'to', 'interaction'].edge_index = mRNA_to_interaction_edges
# show the shape of the interaction node edges
# print(data['mRNA', 'to', 'interaction'].edge_index.shape) # (2, 2816)
# Correctly add self-loops for siRNA nodes (shape [2, num_nodes])
siRNA_self_loop_edge_index = torch.stack([torch.arange(data['siRNA'].x.size(0)),
                                          torch.arange(data['siRNA'].x.size(0))], dim=0)

# Add self-loops for mRNA nodes (shape [2, num_nodes])
mRNA_self_loop_edge_index = torch.stack([torch.arange(data['mRNA'].x.size(0)),
                                         torch.arange(data['mRNA'].x.size(0))], dim=0)

# Add self-loops to HeteroData
data['siRNA', 'self_loop', 'siRNA'].edge_index = siRNA_self_loop_edge_index
data['mRNA', 'self_loop', 'mRNA'].edge_index = mRNA_self_loop_edge_index


print(data)

###############################################
# Model definition
###############################################

class HeteroGraphSAGE(nn.Module):
    def __init__(self, hinsage_layer_sizes, dropout):
        super().__init__()
        
        # First layer: input to hidden size of 32
        self.conv1 = HeteroConv({
            ('siRNA', 'to', 'interaction'): SAGEConv((-1, -1), hinsage_layer_sizes[0]),
            ('mRNA', 'to', 'interaction'): SAGEConv((-1, -1), hinsage_layer_sizes[0]),
            ('siRNA', 'self_loop', 'siRNA'): SAGEConv((-1, -1), hinsage_layer_sizes[0]),
            ('mRNA', 'self_loop', 'mRNA'): SAGEConv((-1, -1), hinsage_layer_sizes[0]),
        }, aggr='mean')

        # Second layer: hidden size of 32 to 16
        self.conv2 = HeteroConv({
            ('siRNA', 'to', 'interaction'): SAGEConv((-1, -1), hinsage_layer_sizes[1]),
            ('mRNA', 'to', 'interaction'): SAGEConv((-1, -1), hinsage_layer_sizes[1]),
            ('siRNA', 'self_loop', 'siRNA'): SAGEConv((-1, -1), hinsage_layer_sizes[1]),
            ('mRNA', 'self_loop', 'mRNA'): SAGEConv((-1, -1), hinsage_layer_sizes[1]),
        }, aggr='mean')
        
        # Final linear layer to reduce the output to 1 value per interaction
        self.linear = nn.Linear(hinsage_layer_sizes[1], 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # First layer message passing with ReLU
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Dropout after the first layer
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Second layer message passing
        x_dict = self.conv2(x_dict, edge_index_dict)

        # Reduce to a single scalar for interaction nodes
        interaction_emb = x_dict['interaction']
        interaction_pred = self.linear(interaction_emb).squeeze(1)
        
        return interaction_pred


###############################################
# Model training
###############################################

from torch_geometric.loader import DataLoader

# Define parameters for minibatch size, epochs, dropout, etc.
batch_size = 60
epochs = 10
hinsage_layer_sizes = [32, 16]  # Two hidden layers with sizes [32, 16]
hop_samples = [8, 4]  # Neighbor sampling sizes for each layer
dropout = 0.15  # Dropout rate
lr = 0.005  # Learning rate for Adamax
loss_fn = F.mse_loss  # MSE loss function

model = HeteroGraphSAGE(hinsage_layer_sizes, dropout)

output = model(data.x_dict, data.edge_index_dict)
graph = make_dot(output, params=dict(model.named_parameters()))
graph.render("hetero_graph_sage_model", format="png")

# Initialize KFold cross-validator
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Prepare interaction data for cross-validation
interaction_features = data['interaction'].x
interaction_labels = data['interaction'].y

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(interaction_features)):
    print(f"Fold {fold + 1}/10")

    # Initialize a fresh model for each fold
    model = HeteroGraphSAGE(hinsage_layer_sizes, dropout)
    optimizer = optim.Adamax(model.parameters(), lr=lr)

    # Convert train_idx to a PyTorch tensor
    train_loader = DataLoader(torch.tensor(train_idx, dtype=torch.long), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Minibatch training loop
        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass for interaction node predictions
            pred = model(data.x_dict, data.edge_index_dict)[batch]
            
            # Compute the MSE loss
            target = interaction_labels[batch]
            loss = loss_fn(pred, target)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
    
    # Convert val_idx to a PyTorch tensor for validation
    val_idx = torch.tensor(val_idx, dtype=torch.long)

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_pred = model(data.x_dict, data.edge_index_dict)[val_idx]
        val_loss = loss_fn(val_pred, interaction_labels[val_idx])
    
    print(f"Fold {fold + 1}, Validation Loss: {val_loss.item()}")

from scipy.stats import pearsonr

def calculate_pcc(predictions, targets):
    # Convert tensors to numpy arrays if necessary
    predictions = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    targets = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    # Calculate the Pearson correlation coefficient
    pcc, _ = pearsonr(predictions, targets)
    return pcc

pcc = calculate_pcc(val_pred, interaction_labels[val_idx])
print(f'Pearson Correlation Coefficient (PCC): {pcc}')

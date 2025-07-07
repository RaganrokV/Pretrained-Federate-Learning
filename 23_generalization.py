# -*- coding: utf-8 -*-
#%%
import numpy as np
import pandas as pd
import warnings
from My_utils.evaluation_scheme import evaluation
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import time
import torch.utils.data as Data
import random
from functions import normalize
from models import TransformerEncoder, EnergyPredictionHead, GeneralPredictionModel
#%%

test_ev = pd.read_pickle('/home/ps/haichao/13_Fedaral_learning/test_ev/final_data.pkl')
#%%
test_ev = normalize(test_ev)
# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Split the data into three datasets based on car_id prefixes
sh_data = test_ev[test_ev['car_id'].str.contains('SH')].copy()
sz_data = test_ev[test_ev['car_id'].str.contains('SZ')].copy()
bit_data = test_ev[test_ev['car_id'].str.contains('BIT')].copy()

# Function to prepare data in your preferred format
def prepare_data(df):
    # Assuming:
    # - The first column (iloc[:, 0]) is the target
    # - The remaining columns (iloc[:, 1:]) are features
    # - 'car_id' column has been dropped already
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare training data
    trainX = torch.Tensor(train_df.iloc[:, 1:].values).float()
    trainY = torch.Tensor(train_df.iloc[:, 0].values).float()
    train_dataset = Data.TensorDataset(trainX, trainY)
    train_loader = Data.DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )
    
    # Prepare test data
    testX = torch.Tensor(test_df.iloc[:, 1:].values).float()
    testY = torch.Tensor(test_df.iloc[:, 0].values).float()
    test_dataset = Data.TensorDataset(testX, testY)
    test_loader = Data.DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,  # No need to shuffle test data
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_loader, test_loader, testX, testY

# Prepare datasets (drop 'car_id' column first)
sh_train, sh_test, sh_X_test, sh_y_test = prepare_data(sh_data.drop('car_id', axis=1))
sz_train, sz_test, sz_X_test, sz_y_test = prepare_data(sz_data.drop('car_id', axis=1))
bit_train, bit_test, bit_X_test, bit_y_test = prepare_data(bit_data.drop('car_id', axis=1))
#%%

criterion = nn.MSELoss()


# sh_model.load_state_dict(ckpt['model_state_dict'])

# sz_model.load_state_dict(ckpt['model_state_dict'])

# bit_model.load_state_dict(ckpt['model_state_dict'])
#%%
# models = {
#     'SH': sh_model,
#     'SZ': sz_model,
#     'BIT': bit_model
# }

# Training function modified for DataLoader input
def train_model(model, train_loader, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), 
    lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Avg Loss: {epoch_loss/len(train_loader):.4f}')


    return model

# Evaluation function using your metric function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    # Calculate metrics using your function
    mae, rmse, mape, smape, r2 = evaluation(y_true.reshape(-1, 1)* 60, 
                                            y_pred.reshape(-1, 1)* 60)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape,
        'R2': r2
    }

# Train and evaluate models
#%%


# === shanghai model ===
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

d_model = 256
feat_dim = 55
max_len = 100
num_layers = 6
nhead = 4
dim_feedforward = 1023
dropout = 0.1
# dropout = 0.5
# === Model Initialization ===
# Initialize GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
Attribution_ids = torch.tensor([5] * 9 + [4] * 6 + [3] * 16 + [2] * 8 + [1] * 6 + [0] * 10)
min_val = Attribution_ids.min()
max_val = Attribution_ids.max()
normalized_attribution_ids = (Attribution_ids - min_val) / (max_val - min_val)

encoder_pretrained = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model, 
                                        max_len=max_len, num_layers=num_layers, nhead=nhead, 
                                        dim_feedforward=dim_feedforward, dropout=dropout)
# Decoder head
energy_head = EnergyPredictionHead(d_model=d_model, feat_dim=feat_dim, 
                                   dropout=dropout).to(device)

# Train each model on its own dataset
sh_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
print("Training SH model...")
SH_model=train_model(sh_model, sh_train, epochs=10,lr=1e-4)

for loader in [sh_test, sz_test, bit_test]:
    metrics=evaluate_model(SH_model, loader)
    print(metrics)
#%%
# torch.save(SH_model, '/home/ps/haichao/13_Fedaral_learning/trained_models/SH_model.pt')

#%%



# === shenzhen model ===

# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# d_model = 256
# feat_dim = 55
# max_len = 100
# num_layers = 6
# nhead = 4
# dim_feedforward = 1023
# dropout = 0.1
# # dropout = 0.5
# # === Model Initialization ===
# # Initialize GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cpu"
# Attribution_ids = torch.tensor([5] * 9 + [4] * 6 + [3] * 16 + [2] * 8 + [1] * 6 + [0] * 10)
# min_val = Attribution_ids.min()
# max_val = Attribution_ids.max()
# normalized_attribution_ids = (Attribution_ids - min_val) / (max_val - min_val)

# encoder_pretrained = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model, 
#                                         max_len=max_len, num_layers=num_layers, nhead=nhead, 
#                                         dim_feedforward=dim_feedforward, dropout=dropout)
# # Decoder head
# energy_head = EnergyPredictionHead(d_model=d_model, feat_dim=feat_dim, 
#                                    dropout=dropout).to(device)
# sz_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
# print("\nTraining SZ model...")
# SZ_model=train_model(sz_model , sz_train, epochs=7,lr=1e-4)


# for loader in [sh_test, sz_test, bit_test]:
#     metrics=evaluate_model(SZ_model, loader)
#     print(metrics)

# torch.save(SZ_model, '/home/ps/haichao/13_Fedaral_learning/trained_models/SZ_model.pt')
#%%


# === other model ===

# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# d_model = 256
# feat_dim = 55
# max_len = 100
# num_layers = 6
# nhead = 4
# dim_feedforward = 1023
# dropout = 0.1
# # dropout = 0.5
# # === Model Initialization ===
# # Initialize GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cpu"
# Attribution_ids = torch.tensor([5] * 9 + [4] * 6 + [3] * 16 + [2] * 8 + [1] * 6 + [0] * 10)
# min_val = Attribution_ids.min()
# max_val = Attribution_ids.max()
# normalized_attribution_ids = (Attribution_ids - min_val) / (max_val - min_val)

# encoder_pretrained = TransformerEncoder(normalized_attribution_ids.to(device), d_model=d_model, 
#                                         max_len=max_len, num_layers=num_layers, nhead=nhead, 
#                                         dim_feedforward=dim_feedforward, dropout=dropout)
# # Decoder head
# energy_head = EnergyPredictionHead(d_model=d_model, feat_dim=feat_dim, 
#                                    dropout=dropout).to(device)

# bit_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
# print("\nTraining BIT model...")
# BIT_model=train_model(bit_model, bit_train, epochs=10,lr=1e-4)


# for loader in [sh_test, sz_test, bit_test]:
#     metrics=evaluate_model(BIT_model, loader)
#     print(metrics)
#%%
# torch.save(BIT_model, '/home/ps/haichao/13_Fedaral_learning/trained_models/BIT_model.pt')
#%%




pfl = torch.load(
    '/home/ps/haichao/13_Fedaral_learning/trained_models/fedavg_model.pth',
    map_location=device  # 自动处理设备映射
)


for loader in [sh_test, sz_test, bit_test]:
    metrics=evaluate_model(pfl, loader)
    print(metrics)


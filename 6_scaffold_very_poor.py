# -*- coding: utf-8 -*-
#%%
import numpy as np
import pandas as pd
import warnings
from My_utils.evaluation_scheme import evaluation
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import time
import torch.utils.data as Data
import random
from functions import normalize
from models import TransformerEncoder, EnergyPredictionHead, GeneralPredictionModel

#%%
'''Karimireddy, Sai Praneeth et al. “SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning.” 
ArXiv abs/1910.06378 (2019): n. pag.
@inproceedings{10.5555/3524938.3525414,
author = {Karimireddy, Sai Praneeth and Kale, Satyen and Mohri, Mehryar and Reddi, Sashank J. and Stich, Sebastian U. and Suresh, Ananda Theertha},
title = {SCAFFOLD: stochastic controlled averaging for federated learning},
year = {2020},
publisher = {JMLR.org},
booktitle = {Proceedings of the 37th International Conference on Machine Learning},
articleno = {476},
numpages = {12},
series = {ICML'20}
}'''
#%%
test_ev = pd.read_pickle('/home/ps/haichao/13_Fedaral_learning/test_ev/final_data.pkl')
#%%
test_ev = normalize(test_ev)
EV_dict = {}
for car_id, group in test_ev.groupby('car_id'):
    # Remove the 'car_id' column from each sub DataFrame
    group = group.drop(columns=['car_id'])
    EV_dict[car_id] = group
train_dict = {}  # Store the training set
test_dict = {}   # Store the test set
for car_id, data in EV_dict.items():
    # Calculate the split point
    split_point = int(len(data) * 0.8)
    
    # The first 80% as the training set
    train_data = data.iloc[:split_point]
    train_dict[car_id] = train_data
    
    # The last 20% as the test set
    test_data = data.iloc[split_point:]
    test_dict[car_id] = test_data
#%%
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === Hyperparameter Settings ===
# chunk_size = 64
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
# Specify to use the second GPU (index 1)
device = "cuda:1" if torch.cuda.is_available() else "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
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
# ===== Create Energy Consumption Prediction Model =======
ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')

#%%
clients_percentage = []
time_complexity = []
# ==== Federated Learning Main Loop ====
for rounds in [2]:
    # break
# for p in [0.2, 0.4, 0.6, 0.8, 1]:
    start_time = time.time()  # Start timing

    # Initialize global model parameters
    global_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
    global_model.load_state_dict(ckpt['model_state_dict'])
    global_model.to(device)
    criterion = nn.MSELoss()
    global_weights = global_model.state_dict()

    # ==== SCAFFOLD Initialization ====
    global_control = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
    client_controls = {}  # {car_id: control_dict}
    # ==== Federated Learning Main Loop ====
    for r in range(rounds):
        local_updates = []  # Store each client's update (parameters and control variables)
        all_clients = list(train_dict.keys())
        num_clients = max(1, int(0.1 * len(all_clients)))
        selected_ids = random.sample(all_clients, num_clients)
        selected_clients = [(car_id, train_dict[car_id]) for car_id in selected_ids]
        for car_id, ev_data in selected_clients:
            trainX = torch.Tensor(ev_data.iloc[:, 1:].values).float()
            trainY = torch.Tensor(ev_data.iloc[:, 0].values).float()
            dataset = Data.TensorDataset(trainX, trainY)
            dataloader = Data.DataLoader(dataset, batch_size=64, shuffle=True)
            # Create local model (load global parameters)
            local_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
            local_model.load_state_dict(global_weights)
            # Initialize or get client's control variables
            if car_id not in client_controls:
                client_controls[car_id] = {
                    name: torch.zeros_like(param, device=device) 
                    for name, param in local_model.named_parameters()
                }
            local_control = client_controls[car_id]
            # Use SGD optimizer (more aligned with SCAFFOLD paper)
            # optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-5)
            # Optimizer settings (unchanged)
            optimizer = torch.optim.AdamW(local_model.parameters(), 
                                        lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                 mode='min', patience=2, factor=0.5, verbose=True)
            num_epochs = 5
            local_model.train()
            for epoch in range(num_epochs):
                total_loss = 0.0
                for step, (x, y) in enumerate(dataloader):
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    pred_y = local_model(x)
                    loss = criterion(pred_y, y.unsqueeze(1))
                    loss.backward()
                    # Modify gradient: grad = grad - local_control + global_control
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            if param.grad is not None:
                                param.grad = param.grad - local_control[name].to(device) + global_control[name].to(device)
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                scheduler.step(avg_loss)
                # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
            # Calculate client control variable update delta_control
            delta_control = {}
            with torch.no_grad():
                for name, param in local_model.named_parameters():
                    delta_control[name] = (
                        (global_weights[name].to(device) - param) 
                        / (len(dataloader) * optimizer.param_groups[0]['lr'])
                    )
            # Save local update
            local_updates.append({
                'weights': local_model.state_dict(),
                'delta_control': delta_control
            })
            # Update client control variable (temporary storage)
            client_controls[car_id] = {
                name: local_control[name] - global_control[name] + delta_control[name]
                for name in local_control
            }
        # ==== Server Aggregation ====
        # Update global model parameters
        total_samples = sum(len(ev_data) for _, ev_data in selected_clients)
        for key in global_weights.keys():
            weighted_params = []
            for (_, ev_data), update in zip(selected_clients, local_updates):
                weight = len(ev_data) / total_samples
                weighted_params.append(update['weights'][key] * weight)
            global_weights[key] = torch.sum(torch.stack(weighted_params), dim=0)
        # Update global control variable
        for key in global_control.keys():
            global_control[key] = torch.mean(
                torch.stack([update['delta_control'][key].to(device) for update in local_updates]),
                dim=0
            )
        # Update global model
        global_model.load_state_dict(global_weights)
        print(f"Round {r + 1}/{rounds} complete")
    end_time = time.time()  # End timing
    training_time = (end_time - start_time) 
    final_model = global_model
    #%% Testing
    start_time = time.time()  # Start timing
    all_predictions = []
    all_true_values = []
    results = []
    # Test each vehicle
    for car_id, ev_data in test_dict.items():
        # Get the test data for this vehicle
        testX = torch.Tensor(ev_data.iloc[:, 1:].values).float()
        testY = torch.Tensor(ev_data.iloc[:, 0].values).float()
        # Move test data to device
        testX, testY = testX.to(device), testY.to(device)
        # Make predictions on the global model
        final_model.to(device).eval()  # Set model to evaluation mode
        with torch.no_grad():
            predictions = final_model(testX)  # Get predictions

        all_predictions.append(predictions)
        all_true_values.append(testY)

        EC_True = testY.cpu().numpy() * 60
        EC_Pred = np.abs(predictions.cpu().numpy()) * 60
        # Compute metrics
        DirectMeasure = np.array(evaluation(EC_True.reshape(-1, 1), 
                                            EC_Pred.reshape(-1, 1)))
        row = [car_id] + DirectMeasure.tolist()
        results.append(row)
    end_time = time.time()  # End timing
    infer_time = (end_time - start_time) 
    complexity = pd.DataFrame({
        'training_time': [training_time],
        'infer_time': [infer_time]
    })
    # Aggregate predictions and true values for further analysis
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    all_true_values = torch.cat(all_true_values, dim=0).cpu().numpy()
    EC_True = all_true_values * 60
    EC_Pred = np.abs(all_predictions) * 60

    Metric1 = np.array(evaluation(EC_True.reshape(-1, 1), EC_Pred.reshape(-1, 1)))
    # print(f"Metric: {Metric1:.4f}")
    columns = ['car_id', 'MAE', 'RMSE', 'MAPE', 'sMAPE', 'R2']
    results_df = pd.DataFrame(results, columns=columns)
    print(results_df.describe())
    print(complexity.describe())
    clients_percentage.append(results_df)
    time_complexity.append(complexity)

# %%
# import pickle
# # Combine data into a dictionary
# data_to_save = {
#     'clients_percentage': clients_percentage,
#     'time_complexity': time_complexity
# }
# # Save as a pickle file
# with open('/home/ps/haichao/13_Fedaral_learning/results/scaffold_round.pkl', 'wb') as f:
#     pickle.dump(data_to_save, f)

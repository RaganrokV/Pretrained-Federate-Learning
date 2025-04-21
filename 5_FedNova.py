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
'''Wang, Jianyu et al. “Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization.” 
ArXiv abs/2007.07481 (2020): n. pag..'''
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
# ===== Create Energy Consumption Prediction Model =======
ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')
#%%
Times_5_m = []
Times_5_t = []
for tt in range(1):
    clients_percentage = []
    time_complexity = []
    for rounds in [16]:  # Ensure correct indentation in the loop body
        start_time = time.time()  # Start timing
        # Initialize global model parameters
        global_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
        global_model.load_state_dict(ckpt['model_state_dict'])
        global_model.to(device)
        criterion = nn.MSELoss()
        global_weights = global_model.state_dict()
        for r in range(rounds):
            # Save initial global parameters
            initial_global_weights = global_model.state_dict()
            # Initialize collection containers
            local_deltas = []   # Parameter changes for each client
            local_taus = []     # Local iterations for each client
            data_sizes = []     # Data sizes for each client
            # Client selection (keep the same as FedAvg)
            all_clients = list(train_dict.keys())
            selected_ids = random.sample(all_clients, int(0.1 * len(all_clients)))
            selected_clients = [(car_id, train_dict[car_id]) for car_id in selected_ids]
            # ---- Local Training ----
            for car_id, ev_data in selected_clients:
                # Data preprocessing (unchanged)
                trainX = torch.Tensor(ev_data.iloc[:, 1:].values).float()
                trainY = torch.Tensor(ev_data.iloc[:, 0].values).float()
                dataset = Data.TensorDataset(trainX, trainY)
                dataloader = Data.DataLoader(dataset, batch_size=64, shuffle=True,
                                             generator=torch.Generator().manual_seed(42))
                # Initialize local model (key modification: load from initial_global_weights)
                local_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
                local_model.load_state_dict(initial_global_weights)  # Use initial global parameters for this round
                # Optimizer settings (unchanged)
                optimizer = torch.optim.AdamW(local_model.parameters(), 
                                            lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                     mode='min', patience=2, factor=0.5, verbose=True)
                # Local training loop (unchanged)
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
                        optimizer.step()
                        total_loss += loss.item()
                    avg_loss = total_loss / len(dataloader)
                    scheduler.step(avg_loss)
                    # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
                # ---- FedNova Key Calculations ----
                # Calculate local iterations (batch count × epoch count)
                tau_i = len(dataloader) * num_epochs
                # Calculate parameter changes delta = θ_local - θ_initial
                local_weights = local_model.state_dict()
                delta_i = {key: local_weights[key] - initial_global_weights[key] 
                         for key in local_weights}
                # Record client information
                local_deltas.append(delta_i)
                local_taus.append(tau_i)
                data_sizes.append(len(ev_data))  # Data size for weighted averaging
            # ---- FedNova Aggregation Logic ----
            total_data = sum(data_sizes)
            # Calculate normalization factor (formula: τ_eff = sum(n_i/n * τ_i))
            tau_eff = sum((data_size/total_data)*tau_i for data_size, tau_i in zip(data_sizes, local_taus))
            # Aggregate parameter changes (formula: Δ = sum(n_i/n * Δ_i/τ_i) * τ_eff)
            fednova_update = {}
            for key in initial_global_weights.keys():
                # Ensure all tensors are converted to Float type (key fix)
                weighted_delta = torch.zeros_like(initial_global_weights[key], dtype=torch.float32)  # Explicitly specify float type
                for delta_i, tau_i, data_size in zip(local_deltas, local_taus, data_sizes):
                    # Ensure data type compatibility
                    normalized_delta = delta_i[key].float() / float(tau_i)        # Explicitly convert to float
                    weight = data_size / total_data                              
                    weighted_delta += normalized_delta * weight
                # Handle potential data type issues
                weighted_delta = weighted_delta.to(initial_global_weights[key].dtype)  # Restore original data type
                fednova_update[key] = weighted_delta * tau_eff
            # Update global parameters: θ_new = θ_initial + Δ
            new_global_weights = {key: initial_global_weights[key] + fednova_update[key].to(initial_global_weights[key].device) 
                                 for key in initial_global_weights}
            global_model.load_state_dict(new_global_weights)
            print(f"Round {r + 1}/{rounds} complete.")
        end_time = time.time()  # End timing
        training_time = (end_time - start_time) 
        final_model = global_model
        #%% Testing

        start_time2 = time.time()  # Start timing
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
        end_time2 = time.time()  # End timing
        infer_time = (end_time2 - start_time2) 
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
    Times_5_m.append(clients_percentage)
    Times_5_t.append(time_complexity)
torch.save(final_model, '/home/ps/haichao/13_Fedaral_learning/trained_models/fedpnova_model.pth')
# %%
# import pickle
# # Combine data into a dictionary
# data_to_save = {
#     'clients_percentage': Times_5_m,
#     'time_complexity': Times_5_t
# }

# # Save as a pickle file
# with open('/home/ps/haichao/13_Fedaral_learning/results/fedpnova_round.pkl', 'wb') as f:
#     pickle.dump(data_to_save, f)
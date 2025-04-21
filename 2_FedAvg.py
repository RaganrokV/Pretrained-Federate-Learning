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
'''McMahan, H. B. et al. “Communication-Efficient Learning of Deep Networks from Decentralized Data.” 
International Conference on Artificial Intelligence and Statistics (2016).
In our experiments, we fix C = 0.1, which strikes a good balance between computational efficiency and convergence rate.
'''
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
device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
criterion = nn.MSELoss()
#%%
from thop import profile
# Your existing code
global_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
global_model.load_state_dict(ckpt['model_state_dict'])
global_model.to(device)
# 1. Calculate trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_params = count_parameters(global_model)
print(f"Trainable Parameters: {trainable_params} M")
for car_id, ev_data in test_dict.items():
    testX = torch.Tensor(ev_data.iloc[:, 1:].values).float()

# # 2. Calculate FLOPs (assuming input size)
# input_shape = (1, 3, 224, 224)  # Adjust size based on actual input
# dummy_input = torch.randn(input_shape).to(device)
flops, params = profile(global_model, inputs=(testX.to(device),))
print(f"FLOPs: {flops / 1e6:.2f} M")
#%%
Times_5_m = []
Times_5_t = []
for tt in range(1):
    clients_percentage = []
    time_complexity = []
    for P in [0.1]:  # Ensure correct indentation in the loop body
        start_time = time.time()  # Start timing
        # Initialize global model parameters
        global_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
        global_model.load_state_dict(ckpt['model_state_dict'])
        global_model.to(device)
        criterion = nn.MSELoss()
        global_weights = global_model.state_dict()
        rounds = 16
        for r in range(rounds):
            
            local_weights = []
            # Each client performs local training
            # Clients selected in each round
            all_clients = list(train_dict.keys())
            selected_ids = random.sample(all_clients, int(P * len(all_clients)))
            selected_clients = [(car_id, train_dict[car_id]) for car_id in selected_ids]
            for car_id, ev_data in selected_clients:
                # Data preprocessing
                trainX = torch.Tensor(ev_data.iloc[:, 1:].values).float()
                trainY = torch.Tensor(ev_data.iloc[:, 0].values).float()
                dataset = Data.TensorDataset(trainX, trainY)
                dataloader = Data.DataLoader(dataset, batch_size=64, shuffle=True,
                                             generator=torch.Generator().manual_seed(42))
                # Create local model
                local_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
                local_model.load_state_dict(global_weights)
                optimizer = torch.optim.AdamW(local_model.parameters(), 
                                              lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                       mode='min', patience=2, factor=0.5, verbose=True)
                num_epochs = 5
                local_model.train()
                for epoch in range(num_epochs):  # Update 
                    total_loss = 0.0
                    for step, (x, y) in enumerate(dataloader):
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        pred_y = local_model(x)
                        loss = criterion(pred_y, y.unsqueeze(1))
                        loss.backward()  # Compute gradients for the current batch's loss
                        optimizer.step()
                        total_loss += loss.item()
                    avg_loss = total_loss / len(dataloader)
                    scheduler.step(avg_loss)
                    # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
                # Collect local model weights
                local_weights.append(local_model.state_dict())
            # Aggregate model weights
            total_samples = sum(len(ev_data) for _, ev_data in selected_clients)
            for key in global_weights.keys():
                # Weighted average (based on client data size)
                weighted_params = []
                for (car_id, ev_data), local_weight in zip(selected_clients, local_weights):
                    weight = len(ev_data) / total_samples
                    weighted_params.append(local_weight[key] * weight)
                global_weights[key] = torch.sum(torch.stack(weighted_params), dim=0)  
            # global_weights = {key: torch.stack([local_weights[i][key] for i in range(len(local_weights))], dim=0).mean(dim=0) for key in global_weights.keys()}
            global_model.load_state_dict(global_weights)
            print(f"Round {r + 1}/{rounds} complete.")
        
        print(f"P: {P} complete.")
        end_time = time.time()  # End timing
        training_time = (end_time - start_time) 

        final_model = global_model
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
    
#%%
# Add to the training loop
# def calculate_comm_cost(model, P, num_clients, bits=32):
#     # Calculate the number of model parameters
#     total_params = sum(p.numel() for p in model.parameters())
    
#     # Calculate communication per client (upload + download)
#     bytes_per_client = total_params * (bits / 8) * 2  # Upload and download
    
#     # Total communication = communication per client × number of participating clients
#     total_comm = bytes_per_client * (P * num_clients)
    
#     # Convert to common units
#     comm_cost_MB = total_comm / (1024 ** 2)  # Convert to MB
    
#     return comm_cost_MB
# # Call after each training round
# num_clients = len(train_dict.keys())  # Total number of clients
# comm_cost = calculate_comm_cost(global_model, P=0.1, num_clients=num_clients)
# print(f"Single-round communication cost: {comm_cost:.2f} MB")
#%%
# import pickle
# # Combine data into a dictionary
# data_to_save = {
#     'clients_percentage': Times_5_m,
#     'time_complexity': Times_5_t
# }
# # Save as a pickle file
# with open('/home/ps/haichao/13_Fedaral_learning/results/fedavg_round.pkl', 'wb') as f:
#     pickle.dump(data_to_save, f)
# %% Save a model for demonstration purposes
torch.save(final_model, '/home/ps/haichao/13_Fedaral_learning/trained_models/fedavg_model.pth')
# %%
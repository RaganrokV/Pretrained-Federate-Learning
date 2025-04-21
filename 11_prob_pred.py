# -*- coding: utf-8 -*-
#%%
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import torch
from functions import normalize
from models import monte_carlo_predict, compute_prediction_intervals, compute_picp_mpiw
#%%
test_ev = pd.read_pickle('/home/ps/haichao/13_Fedaral_learning/test_ev/final_data.pkl')
#%%
test_ev = pd.read_pickle('/home/ps/haichao/13_Fedaral_learning/test_ev/final_data.pkl')
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
# %%
# Define test parameter combinations
p_values = [0.1, 0.35, 0.5, 0.65]
mc_samples_list = [10, 50, 200]
# Initialize result storage list
results = []
# Load the base model (load outside the loop to ensure independence)
base_model_path = '/home/ps/haichao/13_Fedaral_learning/trained_models/fedprox_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Iterate through all parameter combinations
for p in p_values:
    for mc_samples in mc_samples_list:
        # Reload the model (load the original model for each new parameter set)
        fedmodel = torch.load(base_model_path)
        
        # Set Dropout parameter
        for module in fedmodel.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = p
        fedmodel.to(device)
        
        # Initialize metrics
        cwc_count = 0
        picp_list, mpiw_list = [], []
        
        # Start test timing
        start_time = time.time()
        
        # Iterate through test data
        for car_id, ev_data in test_dict.items():
            # Data preparation
            testX = torch.Tensor(ev_data.iloc[:, 1:].values).float()
            testY = torch.Tensor(ev_data.iloc[:, 0].values).float()
            testX, testY = testX.to(device), testY.to(device)
            
            # Monte Carlo prediction
            mean_pred, std_pred = monte_carlo_predict(fedmodel, testX, mc_samples=mc_samples)
            
            # Compute prediction intervals
            lower_bound, upper_bound = compute_prediction_intervals(mean_pred, std_pred, z=1.96)
            
            # Compute metrics
            picp, mpiw = compute_picp_mpiw(lower_bound.cpu().numpy(),
                                         upper_bound.cpu().numpy(),
                                         testY.cpu().numpy())
            cwc = picp / mpiw
            
            # Record data meeting the condition
            if picp >= 0:  # Note: Modify to >0.15 if needed
                picp_list.append(picp)
                mpiw_list.append(mpiw)
                cwc_count += 1
                # print(f"[p={p}][mc={mc_samples}] PICP: {picp:.2f}, MPIW: {mpiw:.2f}, CWC: {cwc:.2f}")
        # Compute elapsed time
        infer_time = time.time() - start_time
        
        # Compute averages
        avg_picp = np.mean(picp_list) if picp_list else 0
        avg_mpiw = np.mean(mpiw_list) if mpiw_list else 0
        
        # Store results
        results.append({
            'Dropout p': p,
            'MC Samples': mc_samples,
            'Average PICP': avg_picp,
            'Average MPIW': avg_mpiw,
            # 'CWC Count': cwc_count,
            # 'Infer Time (s)': infer_time
        })
# Convert to DataFrame
results_df = pd.DataFrame(results)
print("\n===== Final Results Summary =====")
print(results_df)
#%%
# fedmodel = torch.load(f'/home/ps/haichao/13_Fedaral_learning/trained_models/fedprox_model.pth')
# for module in fedmodel.modules():
#     if isinstance(module, torch.nn.Dropout):
#         module.p = 0.65  
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fedmodel.to(device)
# cwc_count = 0 
# picp_list, mpiw_list = [], []
# # Test each vehicle
# start_time = time.time()  # Start timing
# for car_id, ev_data in test_dict.items():
#     # Get the test data for this vehicle
#     testX = torch.Tensor(ev_data.iloc[:, 1:].values).float()
#     testY = torch.Tensor(ev_data.iloc[:, 0].values).float()
    
#     # Move test data to device
#     testX, testY = testX.to(device), testY.to(device)

#     mean_pred, std_pred = monte_carlo_predict(fedmodel, testX, mc_samples=10)
#     lower_bound, upper_bound = compute_prediction_intervals(mean_pred, std_pred, z=1.96)
#     picp, mpiw = compute_picp_mpiw(lower_bound.cpu().numpy(), upper_bound.cpu().numpy(), testY.cpu().numpy())
#     cwc = picp / mpiw
#     # Only record data if CWC > 0.15
#     if picp >= 0.0:
#         picp_list.append(picp)
#         mpiw_list.append(mpiw)
#         cwc_count += 1  # Increment counter if condition is met
#         print(f"PICP: {picp:.2f}, MPIW: {mpiw:.2f}, CWC: {cwc:.2f}")
# end_time = time.time()  # End timing
# infer_time = (end_time - start_time)
# # Compute and output statistical results
# avg_picp = np.mean(picp_list)
# avg_mpiw = np.mean(mpiw_list)

# print("\n===== Final Statistics =====")
# print(f"Total Inference Time: {infer_time:.2f} seconds")
# print(f"Average PICP (CWC > 0.15): {avg_picp:.2f}")
# print(f"Average MPIW (CWC > 0.15): {avg_mpiw:.2f} minutes")
# print(f"Number of times CWC > 0.15: {cwc_count}")
# print("====================")
# # Create a dictionary to store the results you want to record
# data = {
#     'Average PICP (CWC > 0.15)': [avg_picp],
#     'Average MPIW (CWC > 0.15)': [avg_mpiw],
#     'Number of times CWC > 0.15': [cwc_count],
#     'Total Inference Time (seconds)': [infer_time]
# }
# # Save the results to a DataFrame
# results_df = pd.DataFrame(data)
# results_df
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
# === Hyperparameters ===
d_model = 256
feat_dim = 55
max_len = 100
num_layers = 6
nhead = 4
dim_feedforward = 1023
dropout = 0.1
# === Model Initialization ===
device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
# Load the pre-trained model
ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')
criterion = nn.MSELoss()
#%%
# ====================== Parameter Configuration ======================
rounds = 16
num_epochs = 5
client_ratio = 0.1
criterion = nn.MSELoss()
# ====================== Standard Federated Learning ======================
# Initialize the global model
# global_model_non_dp = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
# global_model_non_dp.load_state_dict(ckpt['model_state_dict'])
# global_weights_non_dp = global_model_non_dp.state_dict()
# # Store results
# non_dp_results = []
# non_dp_time = {'train': 0.0, 'infer': 0.0}
# start_train = time.time()
# # Training loop
# for r in range(rounds):
#     # Select clients
#     all_clients = list(train_dict.keys())
#     selected_ids = random.sample(all_clients, int(client_ratio * len(all_clients)))
#     selected_clients = [(car_id, train_dict[car_id]) for car_id in selected_ids]
    
#     local_weights = []
#     member_test_set = pd.DataFrame()
#     # Client local training
#     for car_id, ev_data in selected_clients:
#         trainX = torch.Tensor(ev_data.iloc[:, 1:].values).float()
#         trainY = torch.Tensor(ev_data.iloc[:, 0].values).float()
#         dataset = Data.TensorDataset(trainX, trainY)
#         dataloader = Data.DataLoader(dataset, batch_size=64, shuffle=True)
        
#         # Local model
#         local_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
#         local_model.load_state_dict(global_weights_non_dp)
#         optimizer = torch.optim.AdamW(local_model.parameters(), lr=1e-5)
        
#         # Training loop
#         local_model.train()
#         for epoch in range(num_epochs):
#             total_loss = 0.0
#             for x, y in dataloader:
#                 x, y = x.to(device), y.to(device)
#                 optimizer.zero_grad()
#                 pred_y = local_model(x)
#                 loss = criterion(pred_y, y.unsqueeze(1))
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
        
#         local_weights.append(local_model.state_dict())
#         member_test_set = pd.concat([member_test_set, ev_data.sample(25, replace=True)], ignore_index=True)
#     # Model aggregation (without noise)
#     total_samples = sum(len(ev_data) for _, ev_data in selected_clients)
#     for key in global_weights_non_dp.keys():
#         weighted_sum = torch.sum(torch.stack([
#             local_weights[i][key] * (len(ev_data)/total_samples)
#             for i, (_, ev_data) in enumerate(selected_clients)
#         ]), dim=0)
#         global_weights_non_dp[key] = weighted_sum
    
#     global_model_non_dp.load_state_dict(global_weights_non_dp)
#     print(f"Standard Training Round {r+1}/{rounds} Completed")
# non_dp_time['train'] = time.time() - start_train
# # Evaluate the standard model
# start_infer = time.time()
# global_model_non_dp.eval()
# results_non_dp = []
# for car_id, ev_data in test_dict.items():
#     testX = torch.Tensor(ev_data.iloc[:, 1:].values).float().to(device)
#     testY = torch.Tensor(ev_data.iloc[:, 0].values).float().to(device)
    
#     with torch.no_grad():
#         preds = global_model_non_dp(testX)
    
#     EC_True = testY.cpu().numpy() * 60
#     EC_Pred = preds.cpu().numpy() * 60
#     metrics = np.array(evaluation(EC_True.reshape(-1,1), 
#     EC_Pred.reshape(-1,1)))
#     results_non_dp.append([car_id] + metrics.tolist())
# non_dp_time['infer'] = time.time() - start_infer
#%% ====================== Differential Privacy Federated Learning ======================
# ====================== Parameter Configuration ======================
rounds = 16
num_epochs = 5
client_ratio = 0.1
criterion = nn.MSELoss()
# Reinitialize the global model
global_model_dp = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
global_model_dp.load_state_dict(ckpt['model_state_dict'])
global_weights_dp = global_model_dp.state_dict()
# DP parameters
C = 0.2    # Clipping threshold
epsilon = 1.0  # Privacy budget
delta = 1e-5   # Failure probability
# Store results
dp_results = []
dp_time = {'train': 0.0, 'infer': 0.0}
criterion = nn.MSELoss()
start_train = time.time()
# Training loop
member_test_set = pd.DataFrame()
for r in range(rounds):
    # Select clients
    all_clients = list(train_dict.keys())
    selected_ids = random.sample(all_clients, int(client_ratio * len(all_clients)))
    selected_clients = [(car_id, train_dict[car_id]) for car_id in selected_ids]
    
    local_weights = []
    
    # Client local training
    for car_id, ev_data in selected_clients:
        member_test_set = pd.concat([member_test_set, ev_data.sample(25, replace=True)], ignore_index=True)
        trainX = torch.Tensor(ev_data.iloc[:, 1:].values).float()
        trainY = torch.Tensor(ev_data.iloc[:, 0].values).float()
        dataset = Data.TensorDataset(trainX, trainY)
        dataloader = Data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Local model
        local_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)
        local_model.load_state_dict(global_weights_dp)
        optimizer = torch.optim.AdamW(local_model.parameters(), lr=1e-5)
        
        # Training loop (with gradient clipping)
        local_model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred_y = local_model(x)
                loss = criterion(pred_y, y.unsqueeze(1))
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), C)
                
                optimizer.step()
                total_loss += loss.item()
        
        local_weights.append(local_model.state_dict())
        
    # Model aggregation (with noise)
    total_samples = sum(len(ev_data) for _, ev_data in selected_clients)
    max_weight = max(len(ev_data)/total_samples for (_, ev_data) in selected_clients)
    sensitivity = C * max_weight
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    for key in global_weights_dp.keys():
        weighted_sum = torch.sum(torch.stack([
            local_weights[i][key] * (len(ev_data)/total_samples)
            for i, (_, ev_data) in enumerate(selected_clients)
        ]), dim=0)
        
        # Add Gaussian noise
        param_norm = torch.norm(weighted_sum, p=2).item()  # L2 norm
        param_scale = np.sqrt(np.prod(weighted_sum.shape))  # Square root of parameter dimension
        # Calculate base magnitude (avoid division by zero)
        base_magnitude = param_norm / (param_scale + 1e-10)
        # Dynamically adjust noise magnitude (keep noise as 5%-10% of parameter magnitude)
        adaptive_sigma = sigma * base_magnitude * 0.1 
        
        noise = torch.randn_like(weighted_sum) * adaptive_sigma
        global_weights_dp[key] = weighted_sum + adaptive_sigma 
    
    global_model_dp.load_state_dict(global_weights_dp)
    nan_check = any(torch.isnan(p).any() for p in global_model_dp.parameters())
    print(f"Round {r+1}, NaN exists: {nan_check}, Max weight value: {max(p.max() for p in global_model_dp.parameters())}")
    print(f"DP Training Round {r+1}/{rounds} Completed | Noise Scale: {adaptive_sigma:.9f}")
dp_time['train'] = time.time() - start_train
# Evaluate DP model
start_infer = time.time()
global_model_dp.eval()
results_dp = []
for car_id, ev_data in test_dict.items():
    testX = torch.Tensor(ev_data.iloc[:, 1:].values).float().to(device)
    testY = torch.Tensor(ev_data.iloc[:, 0].values).float().to(device)
    
    with torch.no_grad():
        preds = global_model_dp(testX)
    
    EC_True = testY.cpu().numpy() * 60
    EC_Pred = preds.cpu().numpy() * 60
    metrics = np.array(evaluation(EC_True.reshape(-1,1), 
    EC_Pred.reshape(-1,1)))
    results_dp.append([car_id] + metrics.tolist())
dp_time['infer'] = time.time() - start_infer
# ====================== Result Output ======================
def print_metrics(name, results, time_info):
    df = pd.DataFrame(results, columns=['car_id', 'MAE', 'RMSE', 'MAPE', 'sMAPE', 'R2'])
    print(f"\n=== {name} Results ===")
    print("Evaluation Metrics Statistics:")
    print(df.describe())
    print(f"\nTraining Time: {time_info['train']:.2f}s")
    print(f"Inference Time: {time_info['infer']:.2f}s")
# print_metrics("Standard Federated Learning", results_non_dp, non_dp_time)
print_metrics("Differential Privacy Federated Learning", results_dp, dp_time)
#%%
non_member_test_set = pd.DataFrame()  # Store non-member test set
for car_id, data in test_dict.items():
    # Randomly sample 40 data points from each car's test data
    sampled_data = data.sample(n=40, replace=True)  # Set random_state for reproducibility
    
    # Add the sampled data to non_member_test_set
    non_member_test_set = pd.concat([non_member_test_set, sampled_data], ignore_index=True)
non_member_dataset = Data.TensorDataset(
    torch.Tensor(non_member_test_set.iloc[:, 1:].values).float(),
    torch.Tensor(non_member_test_set.iloc[:, 0].values).float()
)
member_dataset = Data.TensorDataset(
    torch.Tensor(member_test_set.iloc[:, 1:].values).float(),
    torch.Tensor(member_test_set.iloc[:, 0].values).float()
)
# === Membership Inference Attack Implementation ===
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
# Prepare feature containers
member_features = []
non_member_features = []
labels = []
# Set batch size
batch_size = 64
criterion = nn.MSELoss(reduction='none') 
# Process member data
member_loader = Data.DataLoader(member_dataset, batch_size=batch_size, shuffle=False)
for x, y in member_loader:
    x = x.to(device)
    with torch.no_grad():
        pred = global_model_dp(x)
        loss = criterion(pred, y.unsqueeze(1).to(device))
    
    # Feature engineering: prediction value + loss value + statistics
    batch_features = torch.cat([
        pred.cpu().flatten(),                # Raw prediction value
        loss.cpu().detach().flatten(),  # Loss value for each sample
        torch.std(pred, dim=0).cpu().repeat(pred.shape[0]),        # Prediction standard deviation
    ]).view(-1, 3).numpy()
    
    member_features.append(batch_features)
    labels.extend([1]*len(batch_features))
# Process non-member data
non_member_loader = Data.DataLoader(non_member_dataset, batch_size=batch_size, shuffle=False)
for x, y in non_member_loader:
    x = x.to(device)
    with torch.no_grad():
        pred = global_model_dp(x)
        loss = criterion(pred, y.unsqueeze(1).to(device))
    
    batch_features = torch.cat([
        pred.cpu().flatten(),                # Raw prediction value
        loss.cpu().detach().flatten(),  # Loss value for each sample
        torch.std(pred, dim=0).cpu().repeat(pred.shape[0]),        # Prediction standard deviation
    ]).view(-1, 3).numpy()
    
    non_member_features.append(batch_features)
    labels.extend([0]*len(batch_features))
# Merge feature data
X = np.concatenate(member_features + non_member_features, axis=0)
y = np.array(labels)
# Shuffle the data
indices = np.random.permutation(len(X))  # Get shuffled indices
X_shuffled = X[indices]  # Reorder X based on shuffled indices
y_shuffled = y[indices]  # Reorder y based on shuffled indices
# Split into training and test sets (80% training, 20% test)
split_idx = int(0.8 * len(X))
X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
# Train the attack model
attack_model = LogisticRegression(max_iter=1000, class_weight='balanced')
attack_model.fit(X_train, y_train)
# Evaluate attack performance
y_pred = attack_model.predict(X_test)
y_proba = attack_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
# Print attack results
print("\n=== Membership Inference Attack Results ===")
print(f"Test Samples Count: {len(X_test)} (Members: {sum(y_test)}, Non-members: {len(y_test)-sum(y_test)})")
print(f"Attack Accuracy: {acc:.4f}")
print(f"AUC Score: {auc:.4f}")
# Visualize ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(3,4))
plt.rcParams.update({'font.size': 14})
RocCurveDisplay.from_estimator(attack_model, X_test, y_test,
                               name='Attack model', color='blue')
# plt.title('ROC Curve for Membership Inference Attack')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', 
         label='Random guess')
text_str = '\n'.join([  
    f"Samples: 3200",       # Member sample count
    f"Privacy Budget: {epsilon:.1f}",       # Member sample count
    f"Noise Scale: {sigma:.1%}",       # Member sample count  
    # f"Non-members: {len(y_test)-sum(y_test)}",  # Non-member sample count
    f"MAE: {pd.DataFrame(results_dp).iloc[:,1].mean():.3f}",
    f"R$^2$: {pd.DataFrame(results_dp).iloc[:,5].mean():.1%}",
    f"Accuracy: {acc:.1%}",
])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Adjust text box position to bottom right (y-coordinate adjusted from 0.25 to 0.05)
plt.text(0.95, 0.05, text_str, 
         transform=plt.gca().transAxes,
         ha='right', 
         va='bottom',fontsize=14,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
plt.legend()
# plt.savefig(r"/home/ps/haichao/13_Fedaral_learning/figures/mia_1.svg", dpi=600)
plt.show()
# %%
# import matplotlib.pyplot as plt
# from sklearn.metrics import RocCurveDisplay
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.figure(figsize=(3,4))
# plt.rcParams.update({'font.size': 14})
# RocCurveDisplay.from_estimator(attack_model, X_test, y_test,
#                                name='Attack model', color='blue')
# # plt.title('ROC Curve for Membership Inference Attack')
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', 
#          label='Random guess')
# text_str = '\n'.join([  
#     f"Samples: 3200",       # Member sample count
#     f"Privacy Budget: 0.0",       # Member sample count
#     f"Noise Scale: 0.0",       # Member sample count  
#     # f"Non-members: {len(y_test)-sum(y_test)}",  # Non-member sample count
#     f"MAE: {pd.DataFrame(results_non_dp).iloc[:,1].mean():.3f}",
#     f"R$^2$: {pd.DataFrame(results_non_dp).iloc[:,5].mean():.1%}",
#     f"Accuracy: {acc:.1%}",
# ])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # Adjust text box position to bottom right (y-coordinate adjusted from 0.25 to 0.05)
# plt.text(0.95, 0.05, text_str, 
#          transform=plt.gca().transAxes,
#          ha='right', 
#          va='bottom',fontsize=15,
#          bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
# plt.legend()
# # plt.savefig(r"/home/ps/haichao/13_Fedaral_learning/figures/mia_0.svg", dpi=600)
# plt.show()
# %%
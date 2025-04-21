# Pretrained Federated Learning for Privacy-Aware Energy Consumption Prediction in Electric Vehicles
#### ECT is a large-scale open-source pre-trained model for predicting trip energy consumption. 

#### To maximize the utility of ECT for the research community, we provide open-access resources including the pre-trained model, and all codes. We encourage other researchers to explore the feasibility of FL in cross-national contexts or extend the current framework to related domains such as energy-minimization routing problems, prognostic battery health management, and smart charging infrastructure allocation.
# Framework

<img width="356" alt="image" src="https://github.com/user-attachments/assets/00e1dbe4-aefa-4138-8b34-1a52b99f2ed4" />

# How to start?

#### 1. You can download our pre-training model on [OneDrive]([https://1drv.ms/u/c/284956e407934917/Ed6g9DN4KRFJh5Zbyo50MowByxbMMutr_ExWMJwA2qzWEA?e=IP2TJq](https://1drv.ms/u/c/284956e407934917/EW_79LiVimRHvlc6Ne1Zi1EBV_90rNBWObv05X33l7ZJTw?e=3NxvrC))
#### 2. Then, loading a pre-trained model with the following code:
1.ckpt = torch.load('/home/ps/haichao/1-lifelong_learning/Model/best_ECM.pt')

2.global_model = GeneralPredictionModel(encoder_pretrained, energy_head).to(device)

3.global_model.load_state_dict(ckpt['model_state_dict'])

#### 3. Start your own federated learning！



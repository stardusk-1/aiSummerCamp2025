# %%
import numpy as np
import pandas as pd

# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_features = train.drop(['datetime'], axis=1).astype('float32')
test_features = test.drop(['datetime'], axis=1).astype('float32')
scaler.fit(train_features)
train_scaled = scaler.transform(train_features)
test_scaled = scaler.transform(test_features)
train['scaled'] = list(train_scaled)
test['scaled'] = list(test_scaled)

# %%
# split X and y
def split_x_and_y(array, days_used_to_train=7):
    features = []
    labels = []
    for i in range(days_used_to_train, len(array)):
        features.append(array[i-days_used_to_train:i, :])
        labels.append(array[i, 0])  # 预测Global_active_power
    return np.array(features), np.array(labels)

train_array = np.stack(train['scaled'].values)
test_array = np.stack(test['scaled'].values)
train_X, train_y = split_x_and_y(train_array)
test_X, test_y = split_x_and_y(test_array)
print('Train X shape:', train_X.shape, 'Train y shape:', train_y.shape)
print('Test X shape:', test_X.shape, 'Test y shape:', test_y.shape)

# %%
# creat dataloaders
import torch
from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_X_tensor = torch.FloatTensor(train_X).to(device)
train_y_tensor = torch.FloatTensor(train_y).to(device)
test_X_tensor = torch.FloatTensor(test_X).to(device)
test_y_tensor = torch.FloatTensor(test_y).to(device)
train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# build a LSTM model
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.dense(lstm_out[:, -1, :])
        return output

input_size = train_X.shape[2]
model = LSTMModel(input_size=input_size, hidden_size=64, output_size=1).to(device)

# %%
# train the model
import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
epochs = 30
train_losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        batch_size = batch_x.size(0)
        train_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}')

# %%
# evaluate the model on the test set
model.eval()
with torch.no_grad():
    pred_y = model(test_X_tensor).cpu().numpy().squeeze()

# %%
# plotting the predictions against the ground truth
import matplotlib.pyplot as plt
test_y_np = test_y_tensor.cpu().numpy()
plt.figure(figsize=(12, 6))
plt.plot(range(len(pred_y)), pred_y, label='Prediction', alpha=0.8)
plt.plot(range(len(test_y_np)), test_y_np, label='Ground Truth', alpha=0.8)
plt.xlabel('Sample Index')
plt.ylabel('Global_active_power (normalized)')
plt.title('Prediction vs Ground Truth')
plt.legend()
plt.grid(True)
plt.show()

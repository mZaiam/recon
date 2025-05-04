import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import timeit
import numpy as np

from automap import AUTOMAP

version = 'v1'
net = 'automap'

n = 128

filters_cv = 64
l1 = 1e-4

noise = 'missing_wedges'
noise_ps = 1e-1
max_angles = 32
wedge_angles = 32

SEED = 42

torch.manual_seed = SEED
torch.cuda.manual_seed = SEED
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device: ', device)
print()

# Load data

x_train, y_train = np.load('train_sin_brain.npy'), np.load('train_im_brain.npy')
x_test, y_test = np.load('test_sin_brain.npy'), np.load('test_im_brain.npy')

x_train = torch.tensor(x_train, dtype=torch.float32).reshape(-1, n, n)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, n, n)
x_test = torch.tensor(x_test, dtype=torch.float32).reshape(-1, n, n)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, n, n)

if version == 'v1':
    y_train = (y_train - y_train.mean(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)) > 0
    y_test = (y_test - y_test.mean(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)) > 0
    y_train = y_train.to(torch.float32)
    y_test = y_test.to(torch.float32)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

generator = torch.Generator(device='cpu')
generator.manual_seed(SEED)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, generator=generator)

# Instantiate and train model

if net == 'automap':
    automap = AUTOMAP(
            n, 
            filters_cv=filters_cv,
    )
    automap.to(device)
     
criterion = nn.MSELoss()
optimizer = optim.Adam(automap.parameters(), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

epochs = 100

train_losses, test_losses = [], []

for epoch in range(epochs):
    start_epoch = timeit.default_timer()
    automap.train()
    train_loss = 0.0
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        if noise == 'pixel_signature':
            gaussian = noise_ps * torch.kron(
                    torch.randn(x_batch.shape[0], n, 1, device=device),
                    torch.ones(1, 1, n, device=device)
                    )
            
            x_batch += gaussian     
        
        if noise == 'missing_angles':
            n_angles = torch.randint(0, max_angles, (1,)).item()
            angles_idx = torch.randint(0, n, (n_angles,))
            missing_angles = torch.ones(x_batch.shape[0], n, n)
            missing_angles[:, :, angles_idx] = 0
            missing_angles = missing_angles.to(device)

            x_batch *= missing_angles

        if noise == 'missing_wedges':
            mw = torch.randint(0, wedge_angles, (1,)).item()
            x_batch[:, :, : int(mw / 2)] = 0
            x_batch[:, :, - int(mw / 2) :] = 0

        optimizer.zero_grad()
        y_pred = automap(x_batch)
        loss = criterion(y_pred, y_batch) + l1 * torch.sum(torch.abs(list(automap.parameters())[-2]))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    automap.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)          
            y_pred = automap(x_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    scheduler.step(test_loss)

    end_epoch = timeit.default_timer()

    print(f'Epoch {epoch+1}/{epochs}: train_loss: {train_loss:.5f} | test_loss: {test_loss:.5f} | epoch_time: {(end_epoch - start_epoch):.2f} sec')

print()
    
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
    
np.save('train_loss', train_losses)
np.save('test_loss', test_losses)
   
# Evaluate performance

automap.eval()  
test_loss = 0.0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = automap(x_batch)
        mse = torch.mean((y_pred - y_batch) ** 2)
        test_loss += mse.item()

rmse = torch.sqrt(torch.tensor(test_loss / len(test_loader)))

print(f"RMSE: {rmse.item():.5f}")

# Save model

torch.save(automap.state_dict(), 'n128_b64_lr1e-5_L11e-4_brain_mw32_reducelrplateau_v1.pth') # nNUMBER_bBATCHSIZE_lrLEARNINGRATE_L1REGULARIZATION_IMAGES_NOISE_VERSION

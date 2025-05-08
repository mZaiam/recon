import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision

import timeit
import argparse
import numpy as np

from ae import AE

# Instatiating model

parser = argparse.ArgumentParser()
parser.add_argument("--ld", type=int, default=64)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = args.ld
print(f'LD={latent_dim}')
print(f'Using: {device}')

ae = AE(
    device=device,
    latent_dim=latent_dim,
)

# Loading and transforming data

n, batch_size = 128, 64

x_train, y_train = np.load('train_sin_brain.npy'), np.load('train_im_brain.npy')
x_test, y_test = np.load('test_sin_brain.npy'), np.load('test_im_brain.npy')

x_train = torch.tensor(x_train, dtype=torch.float32).reshape(-1, 1, n, n)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1, n, n)
x_test = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1, n, n)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1, n, n)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training

epochs, lr, patience = 500, 1e-4, 20

optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
criterion = nn.MSELoss()

start = timeit.default_timer()

ae.fit(
    train_loader,
    test_loader,
    optimizer, 
    criterion, 
    epochs=epochs, 
    patience=patience,
    model_path=f'best_models/autoencoder_ld{latent_dim}.pth',
)

end = timeit.default_timer()

print(f'{int(end-start)} seconds elapsed.')
print()

train_losses = np.array(ae.losses_train)
test_losses = np.array(ae.losses_val)
    
np.save(f'train_loss_ld{latent_dim}.npy', train_losses)
np.save(f'test_loss_ld{latent_dim}.npy', test_losses)

# RMSE

ae.eval()  
test_loss = 0.0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = ae(x_batch)
        mse = torch.mean((y_pred - y_batch) ** 2)
        test_loss += mse.item()

rmse = torch.sqrt(torch.tensor(test_loss / len(test_loader)))

print(f"RMSE: {rmse.item():.5f}")

import numpy as np
import torch
import timeit
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from skimage.transform import radon

SEED = 42
n = 128
ntheta = 128

# Transforms and load data

transform = transforms.Compose([
    transforms.Resize((n, n)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Load dataset

dataset = datasets.ImageFolder(root='data', transform=transform)

# Convert to Numpy

images = [im.numpy() for im, _ in dataset]
images = np.stack(images, axis=0)
images = images.reshape(-1, n, n)

# Train-test split

train_im, test_im = train_test_split(images, test_size=0.1, random_state=SEED)

# Normalization

train_im /= train_im.max()
test_im /= test_im.max() 

# Image augmentation

train_im_aug, test_im_aug = [], []

for i in range(4):
    train_im_aug.append(np.rot90(train_im, k=i, axes=(1, 2)))
    test_im_aug.append(np.rot90(test_im, k=i, axes=(1, 2)))

train_im = np.concatenate(train_im_aug, axis=0)
test_im = np.concatenate(test_im_aug, axis=0)    

# Creating sinograms

train_sinogram, test_sinogram = [], []

theta = np.linspace(0.0, 180.0, ntheta, endpoint=False)

start = timeit.default_timer()

for im in train_im:
    im_sin = radon(im.squeeze(), theta=theta)
    train_sinogram.append(im_sin)

for im in test_im:
    im_sin = radon(im.squeeze(), theta=theta)
    test_sinogram.append(im_sin)

end = timeit.default_timer()

print(f'{(end - start):.1f} seconds.')

train_sin = np.stack(train_sinogram, axis=0)
test_sin = np.stack(test_sinogram, axis=0)

# Sinogram normalization

train_sin = train_sin / train_sin.max()
test_sin = test_sin / test_sin.max()

# Save data

np.savez_compressed('brain_128', train_im=train_im, test_im=test_im, train_sin=train_sin, test_sin=test_sin)

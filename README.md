# GAN for CIFAR-10

A Deep Convolutional Generative Adversarial Network (DCGAN) for generating CIFAR-10 images using PyTorch.

## Overview
This project implements a DCGAN to generate realistic 32x32 images from the CIFAR-10 dataset (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks).

## Features
- DCGAN architecture with batch normalization
- Trained on Google Colab with free T4 GPU
- 100 epochs training (~2 hours)
- Automatic checkpointing every 10 epochs

## Usage

### Training
```bash
python train.py
```

### Generate Images
```bash
python generate.py
```

## Results
See sample_results.png for generated images after 100 epochs of training.

## Architecture
- **Generator**: 100D latent vector → 3x32x32 RGB image
- **Discriminator**: 3x32x32 RGB image → probability

## Training Details
- Learning Rate: 0.0002
- Batch Size: 128
- Optimizer: Adam (beta1=0.5)
- Loss: Binary Cross Entropy

## License
MIT License

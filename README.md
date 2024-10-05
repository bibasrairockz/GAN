# GAN  
Generated Image  
![image](https://github.com/user-attachments/assets/f60d9728-e751-4733-82a7-526983266677)  

Real Image  
![image](https://github.com/user-attachments/assets/c6654ecd-097a-4916-8dea-e14a2e675f5a)  
   
## Overview

This project implements a Generative Adversarial Network (GAN) to generate images based on the CIFAR-10 dataset, specifically targeting frog images. The GAN consists of two neural networks: a generator and a discriminator. The generator creates fake images, while the discriminator evaluates their authenticity. Through adversarial training, both networks improve, ultimately producing realistic images.

## Features

- Generate images from random latent vectors.
- Train the GAN on the CIFAR-10 dataset.
- Save generated images during training for evaluation.
- Option to visualize generator and discriminator losses.

## Requirements

To run this project, you need the following packages:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib (optional for visualization)

You can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib
```

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/gan-image-generator.git
cd gan-image-generator
```

Install the required dependencies as mentioned in the Requirements section.

The training process will output the discriminator and adversarial losses at regular intervals and save generated images to the specified directory.

Training Process
The GAN is trained using the following steps:

Data Preparation: Load CIFAR-10 dataset and extract frog images.
Model Architecture: Build the generator and discriminator models using Keras.
Training Loop:
Generate random latent vectors.
Use the generator to create fake images.
Train the discriminator with real and fake images.
Train the GAN by freezing the discriminator and optimizing the generator.
The training will continue for a specified number of iterations, saving the model weights and generated images periodically.


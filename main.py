import random
import collections
import numpy as np
import skimage
import torch
from torchvision.datasets import CIFAR10, MNIST
from PIL import Image

from model import *
from train import *
from downstream import *


random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cifar10 = CIFAR10(root='.', train=True, download=True, transform=transforms.ToTensor())
cifar10 = Subset(cifar10, list(range(0, len(cifar10), 5)))  # Select only 10000 units
cifar10_train, cifar10_test, cifar10_val = random_split(cifar10, [7144, 1428, 1428])
mnist = MNIST(root='.', train=True, download=True, transform=transforms.ToTensor())
mnist_in = Subset(mnist, list(range(0, len(mnist), 2)))
mnist_out = Subset(mnist, list(range(1, len(mnist), 2)))  # Test the generalization capacity
mnist_train, mnist_test, mnist_val = random_split(mnist_in, [21428, 4286, 4286])

"""
Configuration 1 MNIST
"""

torch.manual_seed(42)
md = LatentModulatedSiren(hidden_width=256, hidden_depth=5, out_dim=1, latent_dim=256).to(device)
for siren_layer in md.siren_layers:
    siren_layer.to(device)
meta_learning(md, mnist_in, 1000, 128, 3, 100, outer_lr=3e-5, image_size=(28, 28), image_channel=1, save_path='mnist_functa_1.pt')  # 230K updates


"""
Configuration 2 MNIST
"""

torch.manual_seed(42)
md = LatentModulatedSiren(hidden_width=512, hidden_depth=15, out_dim=1, latent_dim=512).to(device)
for siren_layer in md.siren_layers:
    siren_layer.to(device)
meta_sgd_learning(md, mnist_in, 250, 32, 3, 100, outer_lr=3e-6, image_size=(28, 28), image_channel=1, save_path='mnist_functa_2.pt')


"""
Configuration 1 CIFAR10
"""

torch.manual_seed(42)
md = LatentModulatedSiren(hidden_width=256, hidden_depth=5, out_dim=3, latent_dim=256).to(device)
for siren_layer in md.siren_layers:
    siren_layer.to(device)
meta_learning(md, cifar10, 4000, 128, 3, 100, outer_lr=3e-5, save_path='cifar_functa_1.pt')  # 312K updates


"""
Configuration 3 CIFAR10
"""

torch.manual_seed(42)
md = SpatialFuncta(hidden_width=256, hidden_depth=5, out_dim=3, latent_size=8, latent_channel=16).to(device)
for siren_layer in md.siren_layers:
    siren_layer.to(device)
meta_sgd_learning(md, cifar10, 2000, 64, 3, 100, outer_lr=3e-5, save_path='cifar_functa_2.pt')  # 312K updates


"""
Downstream 1 MNIST
"""

model = torch.load("mnist_functa_1.pt").to(device)
for siren_layer in model.siren_layers:
    siren_layer.to(device)
mnist_testset = build_functaset(mnist_test, model)
test_loader = DataLoader(mnist_testset, batch_size=256, num_workers=12)
mnist_trainset = build_functaset(mnist_train, model)
train_loader = DataLoader(mnist_trainset, batch_size=256, shuffle=True, num_workers=12)
mnist_valset = build_functaset(mnist_val, model)
val_loader = DataLoader(mnist_valset, batch_size=256, num_workers=12)
classifier = Classifier(256, 10).to(device)
train_classification(train_loader, test_loader, val_loader, classifier)

"""
Downstream 2 MNIST
"""

model = torch.load("mnist_functa_2.pt").to(device)
for siren_layer in model.siren_layers:
    siren_layer.to(device)
mnist_testset = build_functaset(mnist_test, model)
test_loader = DataLoader(mnist_testset, batch_size=256, num_workers=12)
mnist_trainset = build_functaset(mnist_train, model)
train_loader = DataLoader(mnist_trainset, batch_size=256, shuffle=True, num_workers=12)
mnist_valset = build_functaset(mnist_val, model)
val_loader = DataLoader(mnist_valset, batch_size=256, num_workers=12)
classifier = Classifier(512, 10).to(device)
train_classification(train_loader, test_loader, val_loader, classifier)

"""
Downstream 1 CIFAR10
"""

model = torch.load("cifar_functa_1.pt").to(device)
for siren_layer in model.siren_layers:
    siren_layer.to(device)
cifar10_testset = build_functaset(cifar10_test, model)
test_loader = DataLoader(cifar10_testset, batch_size=256, num_workers=12)
cifar10_trainset = build_functaset(cifar10_train, model)
train_loader = DataLoader(cifar10_trainset, batch_size=256, shuffle=True, num_workers=12)
cifar10_valset = build_functaset(cifar10_val, model)
val_loader = DataLoader(cifar10_valset, batch_size=256, num_workers=12)
classifier = Classifier(256, 10).to(device)
train_classification(train_loader, test_loader, val_loader, classifier)

"""
Downstream 2 CIFAR10
"""

model = torch.load("cifar_functa_2.pt").to(device)
for siren_layer in model.siren_layers:
    siren_layer.to(device)
cifar10_testset = build_functaset(cifar10_test, model)
test_loader = DataLoader(cifar10_testset, batch_size=256, num_workers=12)
cifar10_trainset = build_functaset(cifar10_train, model)
train_loader = DataLoader(cifar10_trainset, batch_size=256, shuffle=True, num_workers=12)
cifar10_valset = build_functaset(cifar10_val, model)
val_loader = DataLoader(cifar10_valset, batch_size=256, num_workers=12)
classifier = ConvClassifier(8, 16, 10).to(device)
train_classification(train_loader, test_loader, val_loader, classifier)

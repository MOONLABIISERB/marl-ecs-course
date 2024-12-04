from google.colab import drive
drive.mount('/content/drive')
import os

checkpoint_path = '/content/drive/MyDrive/rdl_project'
os.makedirs(checkpoint_path, exist_ok=True)

!pip install torch torchvision timm fvcore tqdm
!pip install torchinfo scikit-learn


import torch
import torch.nn as nn
import timm
import random
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from fvcore.nn import FlopCountAnalysis
import torch.profiler
from tqdm import tqdm
import logging

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, 100)
checkpoint = torch.load('/content/drive/MyDrive/rdl_project/vit_cifar100_epoch_5.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Define the test set transformations and loader
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
])
test_dataset = datasets.CIFAR100(root='data', train=False, transform=transform_test, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
train_dataset = datasets.CIFAR100(root='data', train=True, transform=transform_test, download=True)
train_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)



import torch
import random
from tqdm import tqdm
import time
from sklearn.metrics import f1_score

def calculate_manual_flops(model, image_size, patch_size, num_classes):

    embed_dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads
    num_layers = len(model.blocks)
    seq_len = (image_size // patch_size) ** 2 + 1

    total_flops = 0

    # Patch Embedding
    patch_embedding_flops = seq_len * embed_dim * (patch_size ** 2 * 3)
    total_flops += patch_embedding_flops

    # Transformer Encoder Blocks
    for _ in range(num_layers):
        # Multi-Head Attention
        attention_flops = 3 * seq_len * embed_dim ** 2
        head_dim = embed_dim // num_heads
        attention_scoring_flops = num_heads * seq_len ** 2 * head_dim
        attention_output_flops = seq_len ** 2 * embed_dim
        total_attention_flops = attention_flops + attention_scoring_flops + attention_output_flops

        # MLP
        mlp_hidden_dim = model.blocks[0].mlp.fc1.out_features
        mlp_flops = 2 * seq_len * embed_dim * mlp_hidden_dim

        # Total FLOPs for this block
        total_flops += total_attention_flops + mlp_flops


    classification_flops = embed_dim * num_classes
    total_flops += classification_flops

    return total_flops

def evaluate_model_with_manual_flops_and_throughput(model, test_loader, image_size, patch_size, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_true_labels = []
    all_predicted_labels = []

    manual_flops = calculate_manual_flops(model, image_size, patch_size, num_classes)
    progress_bar = tqdm(test_loader, desc="Testing")
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    print()
    print(f"Test Accuracy: {accuracy:.2f}%")

    f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
    print(f"F1 Score (Macro): {f1:.4f}")

    manual_gflops = manual_flops / 1e9
    print(f"Manual Total GFLOPs: {manual_gflops:.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_images = total
    throughput = total_images / elapsed_time

    print(f"Throughput: {throughput:.2f} images per second")

    return accuracy, f1, manual_gflops, throughput


image_size = 224
patch_size = 16
num_classes = 100

accuracy, f1, manual_gflops, throughput = evaluate_model_with_manual_flops_and_throughput(
    model, test_loader, image_size, patch_size, num_classes
)



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
from sklearn.metrics import f1_score
from tqdm import tqdm
import time

def evaluate_model_with_attention_pruning_layer_by_layer(model, input_image, retain_ratio=0.5):
    model.eval()
    input_image = input_image.to(device)

    with torch.no_grad():
        x = model.patch_embed(input_image)
        batch_size = x.shape[0]
        cls_token = model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + model.pos_embed

        num_tokens = x.shape[1]
        num_to_retain = int(num_tokens * retain_ratio)
        attention_scores = model.blocks[0].attn(x)


        if len(attention_scores.shape) == 4:
            cls_attention_scores = attention_scores[:, :, 0, 1:]
            cls_attention_scores = cls_attention_scores.mean(dim=1)
        elif len(attention_scores.shape) == 3:
            attention_scores_per_patch = attention_scores.mean(dim=-1)
            cls_attention_scores = attention_scores_per_patch[:, 1:]
        else:
            raise ValueError("Unexpected shape for attention scores")

        topk_scores, topk_indices = torch.topk(cls_attention_scores, k=num_to_retain, dim=1, largest=True)

        # Flatten the indices and prepend 0 for the first index
        retain_indices = [0] + topk_indices.flatten().tolist()

        # Sort the retained indices to ensure order is preserved
        retain_indices.sort()

        # Select the features corresponding to the retained indices
        x = x[:, retain_indices]

        for i, block in enumerate(model.blocks):
            x = block(x)

        x = model.norm(x)
        cls_output = x[:, 0]
        output = model.head(cls_output)

    return output


def calculate_manual_flops_with_pruning(model, image_size, patch_size, embed_dim, num_classes, retain_ratio):
    num_patches = (image_size // patch_size) ** 2
    seq_len = int(num_patches * retain_ratio) + 1
    num_heads = model.blocks[0].attn.num_heads
    num_layers = len(model.blocks)

    total_flops = 0

    patch_embedding_flops = num_patches * (patch_size ** 2 * 3) * embed_dim
    total_flops += patch_embedding_flops

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

        total_flops += total_attention_flops + mlp_flops

    classification_flops = embed_dim * num_classes
    total_flops += classification_flops

    return total_flops

def evaluate_model_with_pruning_metrics_manual_flops(model, test_loader, image_size, patch_size, embed_dim, num_classes, retain_ratio=0.5):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    manual_flops = calculate_manual_flops_with_pruning(
        model, image_size, patch_size, embed_dim, num_classes, retain_ratio
    )

    start_time = time.time()

    progress_bar = tqdm(test_loader, desc="Testing with Attention Pruning")

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = evaluate_model_with_attention_pruning_layer_by_layer(model, inputs, retain_ratio=retain_ratio)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(accuracy=100 * correct / total)

    end_time = time.time()

    elapsed_time = end_time - start_time
    throughput = total / elapsed_time

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    gflops_total = manual_flops / 1e9  # Convert FLOPs to GFLOPs
    print()
    print(f"Pruned Test Accuracy: {accuracy:.2f}%")
    print(f"Pruned F1 Score: {f1:.4f}")
    print(f"Manual Total GFLOPs (with pruning): {gflops_total:.2f}")
    print(f"Throughput: {throughput:.2f} images per second")

    return accuracy, f1, gflops_total, throughput

image_size = 224
patch_size = 16
embed_dim = model.embed_dim
num_classes = 100
retain_ratio = 0.5

accuracy_pruned, f1_pruned, manual_gflops_pruned, throughput_pruned = evaluate_model_with_pruning_metrics_manual_flops(
    model, test_loader, image_size, patch_size, embed_dim, num_classes, retain_ratio
)

import matplotlib.pyplot as plt
import numpy as np

# Define a function to unnormalize and convert the tensor image to a NumPy array
def unnormalize_and_convert(tensor_img, mean, std):
    """
    Unnormalize and convert a PyTorch tensor image to a NumPy array for visualization.

    Args:
    - tensor_img (torch.Tensor): The tensor image to unnormalize and convert.
    - mean (list): The mean values used for normalization.
    - std (list): The standard deviation values used for normalization.

    Returns:
    - np.ndarray: The unnormalized image as a NumPy array.
    """
    img = tensor_img.clone()  # Clone to avoid modifying the original tensor
    for channel, m, s in zip(img, mean, std):
        channel.mul_(s).add_(m)  # Unnormalize the channel
    return np.transpose(img.numpy(), (1, 2, 0))  # Convert from CxHxW to HxWxC

# Get the first batch of data
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Select the first image and its label
first_image = images[2]
first_label = labels[2]

# Unnormalize the first image
mean = [0.5071, 0.4865, 0.4409]
std = [0.2673, 0.2564, 0.2762]
unnormalized_image = unnormalize_and_convert(first_image, mean, std)

# Plot the first image
plt.figure(figsize=(4, 4))
plt.imshow(unnormalized_image)
plt.title(f"Label: {test_dataset.classes[first_label]}")
plt.axis('off')
plt.show()


import matplotlib.pyplot as plt
import numpy as np

def visualize_pruned_patches_with_label(original_image, mask, mean, std, label, class_names):
    """
    Visualize the original image and the pruned image side by side with the label.

    Args:
    - original_image (torch.Tensor): The original unnormalized image.
    - mask (torch.Tensor): Binary mask indicating the pruned patches (0 for pruned, 1 for kept).
    - mean (list): Mean values used for normalization.
    - std (list): Standard deviation values used for normalization.
    - label (int): The label of the image.
    - class_names (list): List of class names corresponding to labels.
    """
    # Unnormalize the original image
    img = original_image.clone()
    for channel, m, s in zip(img, mean, std):
        channel.mul_(s).add_(m)
    img_np = np.transpose(img.numpy(), (1, 2, 0))  # Convert from CxHxW to HxWxC

    # Define patch dimensions
    img_height, img_width = img_np.shape[:2]
    num_patches = mask.size(0)
    patch_dim = int(np.sqrt(num_patches))  # Assuming a square grid of patches
    patch_height = img_height // patch_dim
    patch_width = img_width // patch_dim

    # Create a mask overlay for the pruned image
    pruned_image = img_np.copy()
    for idx in range(num_patches):
        if mask[idx] == 0:  # Mask out pruned patches
            row = idx // patch_dim
            col = idx % patch_dim
            pruned_image[
                row * patch_height : (row + 1) * patch_height,
                col * patch_width : (col + 1) * patch_width,
            ] = [1, 1, 1]  # Red mask for pruned patches

    # Plot the original and pruned images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image\nLabel: {class_names[label]}")
    axes[0].axis('off')

    axes[1].imshow(pruned_image)
    axes[1].set_title(f"Pruned Image\nLabel: {class_names[label]}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage with RL agent's output
# Assuming `agent_attention` and `agent_similarity` are trained agents
# and `first_image` is a tensor of shape (C, H, W)

# Preprocess the first image to get embeddings and scores
x = model.patch_embed(first_image.unsqueeze(0).to(device))
cls_token = model.cls_token.expand(x.size(0), -1, -1)
x = torch.cat((cls_token, x), dim=1)
x = x + model.pos_embed
attention_scores = model.blocks[0].attn(x).mean(dim=-1)
x_normalized = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
similarity_matrix = torch.matmul(x_normalized, x_normalized.transpose(1, 2))
similarity_scores = similarity_matrix.mean(dim=-1)

# Get agent actions
env = MultiAgentTokenPruningEnv(model, device, target_prune_ratio=0.5)
states_attention, states_similarity = env.reset(
    x, attention_scores, similarity_scores, torch.tensor([first_label]).to(device)
)
action_probs_attention, _ = agent_attention(states_attention)
action_probs_similarity, _ = agent_similarity(states_similarity)

actions_attention = (action_probs_attention > 0.5).int()
actions_similarity = (action_probs_similarity > 0.5).int()

# Combine the masks from both agents
combined_mask = actions_attention[0] & actions_similarity[0]

# Visualize the original and pruned images with labels
visualize_pruned_patches_with_label(first_image, combined_mask.cpu(), mean, std, first_label, test_dataset.classes)

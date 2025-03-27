import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# We're getting a special computer brain that's already learned to see pictures
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# We need to make our pictures the right size and shape for the computer to look at them
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Let's get a picture of a panda to look at
dataset = datasets.ImageFolder(root="images", transform=transform)

# Get the folder name from the dataset's root path
folder_name = dataset.root.split('/')[-1]
results_filename = f'results_{folder_name}.txt'

# Move this function definition to before the main processing loop
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Process all images in the dataset
with open(results_filename, 'w') as f:
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        image = image.unsqueeze(0)
        image.requires_grad = True

        # Get original prediction
        output = model(image)
        original_pred = output.max(1, keepdim=True)[1]
        original_confidence = torch.nn.functional.softmax(output, dim=1).max().item() * 100

        # Create target for attack
        target_class = (original_pred.item() + 1) % 1000
        target = torch.tensor([target_class])

        # Perform attack
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = image.grad.data

        # Generate perturbed image
        epsilon = 0.1
        perturbed_image = fgsm_attack(image, epsilon, data_grad)

        # Get prediction for perturbed image
        perturbed_output = model(perturbed_image)
        perturbed_pred = perturbed_output.max(1, keepdim=True)[1]
        perturbed_confidence = torch.nn.functional.softmax(perturbed_output, dim=1).max().item() * 100

        # Write results for this image
        f.write(f"\nImage {idx + 1}:\n")
        f.write(f"Original Image (from {dataset.imgs[idx][0]}):\n")
        f.write(f"Predicted class: {original_pred.item()} [This is the number that tells us what the computer thinks it sees!]\n")
        f.write(f"Confidence: {original_confidence:.2f}% [This is how sure the computer is about its guess]\n\n")
        f.write(f"Perturbed Image:\n")
        f.write(f"Predicted class: {perturbed_pred.item()} [This is what the computer thinks it sees in our sneaky changed picture!]\n")
        f.write(f"Confidence: {perturbed_confidence:.2f}% [This shows how sure the computer is about the sneaky picture]\n")
        f.write(f"Epsilon used: {epsilon}\n")
        f.write("-" * 50 + "\n")

        # Save comparison plot for each image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze().permute(1, 2, 0).detach().numpy())
        plt.title(f"Original Image {idx + 1}")

        plt.subplot(1, 2, 2)
        plt.imshow(perturbed_image.squeeze().permute(1, 2, 0).detach().numpy())
        plt.title(f"Perturbed Image {idx + 1}")

        # Save the comparison plot with unique name for each image
        plt.savefig(f'comparison_{folder_name}_image_{idx + 1}.png')
        plt.close()  # Close the figure to free memory

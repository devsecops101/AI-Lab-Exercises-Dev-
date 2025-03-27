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
dataset = datasets.ImageFolder(root=".", transform=transform)
image, label = dataset[0]  # We're just looking at the first picture

# We need to wrap our picture in a special way so the computer can look at it
image = image.unsqueeze(0)
image.requires_grad = True

# This is like making a sneaky change to our picture that the computer won't notice
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Let's show our picture to the computer and see what it thinks it is
output = model(image)
original_pred = output.max(1, keepdim=True)[1]
original_confidence = torch.nn.functional.softmax(output, dim=1).max().item() * 100

# Fix: Create a target tensor in the correct format
target_class = (original_pred.item() + 1) % 1000  # Choose a different class
target = torch.tensor([target_class])  # Create a 1D tensor with a single value

# Now we're going to make the computer confused about what it sees
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)  # Use the properly formatted target
model.zero_grad()
loss.backward()
data_grad = image.grad.data

# Let's make our sneaky change to the picture
epsilon = 0.1
perturbed_image = fgsm_attack(image, epsilon, data_grad)

# Get prediction for perturbed image
perturbed_output = model(perturbed_image)
perturbed_pred = perturbed_output.max(1, keepdim=True)[1]
perturbed_confidence = torch.nn.functional.softmax(perturbed_output, dim=1).max().item() * 100

# Save the predictions to a text file
with open('results.txt', 'w') as f:
    f.write(f"Original Image:\n")
    f.write(f"Predicted class: {original_pred.item()} [This is the number that tells us what the computer thinks it sees!]\n")
    f.write(f"Confidence: {original_confidence:.2f}% [This is how sure the computer is about its guess - like being really really sure or just kind of sure!]\n\n")
    f.write(f"Perturbed Image:\n")
    f.write(f"Predicted class: {perturbed_pred.item()} [This is what the computer thinks it sees in our sneaky changed picture!]\n")
    f.write(f"Confidence: {perturbed_confidence:.2f}% [This shows how sure the computer is about the sneaky picture - we want this to be different!]\n")
    f.write(f"Epsilon used: {epsilon} [This is like how much we changed the picture - bigger numbers mean bigger changes!]\n")

# Now let's show both pictures side by side - the real one and our sneaky one
plt.figure(figsize=(10, 5))  # Optional: set a specific figure size
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("The Real Picture")

plt.subplot(1, 2, 2)
plt.imshow(perturbed_image.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("Our Sneaky Picture")

# Save the comparison plot
plt.savefig('comparison.png')
plt.show()  # Optional: you can still display it too

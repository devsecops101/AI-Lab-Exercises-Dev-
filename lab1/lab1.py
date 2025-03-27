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
pred = output.max(1)[1]

# Now we're going to make the computer confused about what it sees
criterion = nn.CrossEntropyLoss()
loss = criterion(output, pred)
model.zero_grad()
loss.backward()
data_grad = image.grad.data

# Let's make our sneaky change to the picture
epsilon = 0.1
perturbed_image = fgsm_attack(image, epsilon, data_grad)

# Now let's show both pictures side by side - the real one and our sneaky one
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("The Real Picture")

plt.subplot(1, 2, 2)
plt.imshow(perturbed_image.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("Our Sneaky Picture")

plt.show()

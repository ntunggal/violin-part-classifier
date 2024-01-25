import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as pyplot
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 4
batch_size = 16
learning_rate = 0.01


classes = sorted(['violin_back', 'back_zoom', 'violin_left', 'front_zoom',
           'scroll_front', 'scroll_left', 'label', 'scroll_back',
           'violin_front', 'scroll_right', 'violin_right'])


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class ViolinData(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))

        # A dict mapping classes to list of file paths
        self.class_file_paths = {c: [] for c in self.classes}
        for c in self.classes:
            class_folder = os.path.join(root_dir, c)
            self.class_file_paths[c] = [os.path.join(class_folder, file) for file in os.listdir(class_folder)]

    def __len__(self):
        return sum(len(f) for _,_,f in os.walk(self.root_dir))
    
    def __getitem__(self, index):
        # Navigate to the correct class folder
        class_idx = 0
        while index >= len(self.class_file_paths[self.classes[class_idx]]):
            index -= len(self.class_file_paths[self.classes[class_idx]])
            class_idx += 1

        # Navigate to the correct image
        label = self.classes[class_idx]
        image_path = self.class_file_paths[label][index]
        image = io.imread(image_path)

        # Apply transforms, if any
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(class_idx, dtype=torch.long)
        
        return image, label


train_dataset = ViolinData(r'violin_data/train', transform=transform)

test_dataset = ViolinData(r'violin_data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # conv layer: (width-filler + 2*padding)/stride + 1
        self.conv1 = nn.Conv2d(3, 6, 5) # input is 3 for 3 color channels
        self.pool = nn.MaxPool2d(2, 2) # kernel size, stride
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120) # output channels*shape
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11) # output size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # does conv1, applies relu, pools
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) # flattens the layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''

# Chat GPT generated
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 6 * 6, 128)  # Adjusted size based on the actual output size
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(128, 11)  # 11 output classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(-1, 32 * 6 * 6)  # Reshape before fully connected layer
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss() # loss calculation
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #print(labels)

        # Forward pass
        outputs = model(images)
        #print(outputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], step [{i+1}/{n_total_steps}], loss: {loss.item():.4f}')

print('Finished training')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(11)]
    n_class_samples = [0 for i in range(11)]
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            #print(f'label: {label}')
            pred = predicted[i]
            #print(f'pred: {pred}')
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

        #print('one img,label-----------------')

    acc = 100.0 * (n_correct / n_samples)
    print(f'Accuracy of the network: {acc}%')

    for i in range(10):
        acc = 100.0 * (n_class_correct[i] / n_class_samples[i])
        print(f'Accuracy of {classes[i]}: {acc}%')
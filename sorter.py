from matplotlib import pyplot as plt
import torch.optim.adam
import torch, torchvision
from torchvision import transforms

transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(50), transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(root='dataset', transform=transform)

train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

batch = 12
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=True, drop_last=True)

net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(50 * 50 * 3, 16 * 53 * 53),
    torch.nn.ReLU(),
    torch.nn.Linear(16 * 53 * 53, 1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 4)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1)
train_losses = []
test_losses = []

for module in net.modules():
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()

for epoch in range(100):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
        test_losses.append(loss.item())
    

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the test images: {100 * correct / total}%')

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()
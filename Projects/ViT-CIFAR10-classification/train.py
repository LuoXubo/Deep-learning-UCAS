"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/05/13 21:18:56
"""

import torch
import torchvision
from torch import nn
from models.vit import ViT

def train_fn(model, data_loader, optimizer, device):
    model.train()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()

        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx} Loss: {loss.item()}")


def eval_fn(model, data_loader, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def predict_fn(model, data_loader, device):
    model.eval()
    fin_outputs = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            outputs = model(data)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return fin_outputs

def main():

    # load model
    device = "cuda"
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=6,
        heads=8,
        mlp_dim=128,
        dropout=0.1
    )
    model.to(device)
    print(model)

    # load cifar-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="../../Dataset/",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="../../Dataset/",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # train model
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        train_fn(model, train_loader, optimizer, device)
        outputs, targets = eval_fn(model, test_loader, device)
        outputs = torch.tensor(outputs)
        targets = torch.tensor(targets)
        accuracy = (outputs.argmax(1) == targets).float().mean()
        print(f"Epoch {epoch} Accuracy: {accuracy}")

        # save model
        torch.save(model.state_dict(), f"caches/vit_{epoch}.bin")
        



if __name__ == '__main__':
    main()
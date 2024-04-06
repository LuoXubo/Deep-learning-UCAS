import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn
import argparse
from net import Net, Vanilla_Net
from utils.dataloader import LoadData
 
n_epochs = 4
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

train_losses = []
train_counter = []
test_losses = []

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((max(0, epoch - 1)) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './caches/model.pth')
            torch.save(optimizer.state_dict(), './caches/optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--method', type=str, default='vanilla', metavar='M', help='which method to use (default: vanilla)')
    parser.add_argument('--train', type=bool, default=True, metavar='T', help='whether to train the model (default: True)')

    args = parser.parse_args()
    batch_size_train = args.batch_size
    batch_size_test = args.test_batch_size
    n_epochs = args.epochs
    learning_rate = args.lr
    momentum = args.momentum
    random_seed = args.seed
    log_interval = args.log_interval
    method = args.method
    train_flag = args.train

    if method == 'vanilla':
        net = Vanilla_Net()
    elif method == 'resnet':
        net = Net()
    elif method == 'ml':
        net = sklearn.linear_model.LogisticRegression()
    
    # ----------------------------------------------------------- #
    # load data
    # ----------------------------------------------------------- #
    train_loader, test_loader = LoadData('../../Dataset/mnist/', batch_size_train, batch_size_test)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


    # ----------------------------------------------------------- #
    # train
    # ----------------------------------------------------------- #
    if train_flag:
        network = net
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
        for epoch in range(1, n_epochs + 1):
            train(epoch)
            test()
    
    
    # ----------------------------------------------------------- #
    # test
    # ----------------------------------------------------------- #
    continued_network = net
    continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    
    network_state_dict = torch.load('./caches/model.pth')
    continued_network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load('./caches/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)
    
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = network(example_data)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
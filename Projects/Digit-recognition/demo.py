import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from net import *
from utils.dataloader import LoadData

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
            torch.save(network.state_dict(), './caches/' + method + '_model.pth')
            torch.save(network.state_dict(), './caches/' + method + '_optimizer.pth')

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
    parser.add_argument('--method', type=str, default='res_cnn', metavar='M', help='which method to use "vanilla_cnn, res_cnn, vanilla_transformer" (default: vanilla)')
    parser.add_argument('--train', type=bool, default=True, metavar='T', help='whether to train the model (default: True)')
    parser.add_argument('--data_path', type=str, default='../../Dataset/mnist/', metavar='D', help='data path (default: ../../Dataset/mnist/)')

    # Transformer (if method is vanilla_transformer)
    parser.add_argument('--n_channels', type=int, default=1, help='number of input channels')
    parser.add_argument("--embed_dim", type=int, default=64, help="dimensionality of the latent space")
    parser.add_argument("--n_attention_heads", type=int, default=4, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=2, help="forward multiplier")
    parser.add_argument("--n_layers", type=int, default=6, help="number of encoder layers")
    parser.add_argument("--image_size", type=int, default=28, help="image size")
    parser.add_argument("--patch_size", type=int, default=4, help="patch size")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes")

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
    data_path = args.data_path

    print('--------------------------------------')
    print('batch_size_train: ', batch_size_train)
    print('batch_size_test: ', batch_size_test)
    print('n_epochs: ', n_epochs)
    print('learning_rate: ', learning_rate)
    print('momentum: ', momentum)
    print('random_seed: ', random_seed)
    print('log_interval: ', log_interval)
    print('method: ', method)
    print('data_path: ', data_path)
    print('--------------------------------------')

    if method == 'vanilla_cnn':
        net = Vanilla_CNN()
    elif method == 'res_cnn':
        net = ResCNN()
    elif method == 'vanilla_transformer':
        n_channels = args.n_channels
        embed_dim = args.embed_dim
        n_layers = args.n_layers
        n_attention_heads = args.n_attention_heads 
        forward_mul = args.forward_mul
        image_size = args.image_size
        patch_size = args.patch_size
        n_classes = args.n_classes

        net = Vanilla_Transformer(n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes)
    else:
        raise ValueError('Method not found! (vanilla_cnn, res_cnn, vanilla_transformer)')
    
    # ----------------------------------------------------------- #
    # load data
    # ----------------------------------------------------------- #
    train_loader, test_loader = LoadData(data_path, batch_size_train, batch_size_test)
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
    network_state_dict = torch.load('./caches/' + method + '_model.pth')
    continued_network.load_state_dict(network_state_dict)

    
    #-----------------------------------------------------------#
    # visualize
    #-----------------------------------------------------------#
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = continued_network(example_data)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/06/10 20:38:19
"""

import torch
import random
from utils import prepareData, generate, gen_acrostic, train
from model import LSTM, DoubleLSTM
import argparse
import os
# os.path.abspath(os.path.dirname(__file__))

random.seed(20)
torch.manual_seed(20)
EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
LSTM_OUTDIM = 512
LR = 0.001
MAX_GEN_LEN = 200
EPOCHS = 20
DROP_PROB = 0.5
LSTM_LAYER = 3
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../Dataset/tang.npz')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model', type=str, default='DoubleLSTM', help='DoubleLSTM or LSTM')
    parser.add_argument('--model_path', type=str, default=None, help='The path of the model. If you want to train the model, you can set it to None.')
    args = parser.parse_args()

    # Load data
    print('-----------------------------------')
    print('Loading data from %s ...'%args.data_path)
    poem_loader, ix2word, word2ix = prepareData(args.data_path, args.batch_size)
    print('Data loaded.')
    print('-----------------------------------')


    # Load model
    print('-----------------------------------')
    print('Loading %s ...'%args.model)
    if args.model == 'DoubleLSTM':
        model = DoubleLSTM(len(word2ix), EMBEDDING_DIM, HIDDEN_DIM)
    else:
        model = LSTM(len(word2ix), EMBEDDING_DIM, HIDDEN_DIM)
    print('%s loaded.'%args.model)
    print('-----------------------------------')

    # Train model
    print('-----------------------------------')
    print('Training ...')
    if args.model_path is None:
        train(args.model, model, args.epochs, poem_loader, word2ix, device)
    print('Training finished.')
    print('-----------------------------------')

    # Load trained model
    print('-----------------------------------')
    model_path = 'caches/%s_%s.pth'%(args.model, args.epochs-1)
    model = DoubleLSTM(len(word2ix), EMBEDDING_DIM, HIDDEN_DIM,)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print('Load model from %s ...'%{model_path})
    print('-----------------------------------')
    

    results1 = generate(model, '朝辞白帝彩云间', ix2word, word2ix, device)
    results2 = gen_acrostic(model, '毕业快乐', ix2word, word2ix, device)

    print('The generated result with first line: 朝辞白帝彩云间')
    print(results1)
    print('-----------------------------------')
    print('The generated result with first line: 毕业快乐')
    print(results2)

    print('-----------------------------------')

    caches = ['caches/DoubleLSTM_0.pth', 'caches/DoubleLSTM_5.pth', 'caches/DoubleLSTM_10.pth', 'caches/DoubleLSTM_15.pth', 'caches/DoubleLSTM_19.pth']
    with open('generated_poems.txt', 'w') as f:
        
        for cache in caches:
            # model = DoubleLSTM(len(word2ix), EMBEDDING_DIM, HIDDEN_DIM,)
            model.load_state_dict(torch.load(cache, map_location=torch.device('cpu')))
            results1 = generate(model, '朝辞白帝彩云间', ix2word, word2ix, device)
            # results2 = gen_acrostic(model, '毕业快乐', ix2word, word2ix, device)
            print('The generated result with model %s'%cache)
            print(results1)
            print('-----------------------------------')
            f.write('model %s: \n'%cache)
            f.write(results1)
            f.write('\n-----------------------------------\n')
    f.close()
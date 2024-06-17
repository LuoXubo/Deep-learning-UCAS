"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/06/15 14:21:28
"""

from model import *
from utils import *
import tqdm
import argparse
import os

# root_path = r'../../Dataset/sample/sample-submission-version/TM-training-set/'
root_path = r'../../Dataset/'
ch_path = root_path + 'tgt.txt'
en_path = root_path + 'src.txt'


# 训练
def train(model_path='./model/translation_99.pt', epochs=100, device='cuda'): 
    save_path = model_path.split('/')[:-1]
    save_path = '/'.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Transformer(tokenizer, device=device)
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    model = model.to(device)
    criteria = LabelSmoothing(tokenizer.get_vocab_size(), tokenizer.word_2_index['<pad>'])
    optimizer = NoamOpt(256, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    lossF = SimpleLossCompute(model.generator, criteria, optimizer)

    model.train()
    loss_all = []
    print('词表大小', tokenizer.get_vocab_size())
    t = time.time()
    data_loader = tokenizer.get_dataloader(tokenizer.data_)
    # random_integers = random.sample(range(len(tokenizer.test)-10), 6)  # 随机选100个句子
    batchs=[]
    for index, data in enumerate(data_loader):
        src, tgt = data
        # 处理一个batch
        batch = Batch(src, tgt, tokenizer=tokenizer, device=device)
        batchs.append(batch)
    

    for epoch in tqdm.tqdm(range(epochs)):
        p=0
        for batch in batchs:
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            out = lossF(out, batch.trg_y, batch.ntokens)
            if (p+1) % 1000 == 0:
                model.eval()
                print('epoch', epoch, 'loss', float(out / batch.ntokens))
                model.train()
                print('time', time.time() - t)
                if float(out / batch.ntokens)<2.2:
                    random_integers = random.sample(range(len(tokenizer.test)), 100)
                    # nu=compute_bleu4(tokenizer, random_integers, model, device)

            p+=1
        if epoch % 10 == 0 or epoch == epochs - 1:
            save_model_path = os.path.join(save_path, f'translation_{epoch}.pt')
            torch.save(model.state_dict(), save_model_path)
        
        loss_all.append(float(out / batch.ntokens))
        

def eval(model_path='./model/translation_99.pt', device='cuda'):

    model1 = Transformer(tokenizer)
    model1.load_state_dict(torch.load(model_path))
    model1 = model1.to(device)
    model1.eval()
    all_=[]
    for i in range(10):
        random_integers = range(len(tokenizer.test))[i*10:i*10+10]  # 评估
        end=compute_bleu4(tokenizer, random_integers, model1, device)
        if end==0:
            continue
        all_.append(end)
    print(sum(all_)/len(all_)) # 输出bleu4得分

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='./model/translation_99.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    tokenizer = Tokenizer(en_path, ch_path, count_min=3)

    if args.train:
        train(model_path=args.model_path, epochs=args.epochs, device=args.device)
    if args.eval:
        eval(model_path=args.model_path, device=args.device)

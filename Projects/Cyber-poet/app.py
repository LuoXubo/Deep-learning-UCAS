import streamlit as st
from main import generate
from config import Config
import torch
from model import LSTM, DoubleLSTM
from utils import *
import random

opt = Config()
data, word2ix, ix2word = get_data(opt)

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

# Load data
print('-----------------------------------')
print('Loading data from ../../Dataset/tang.npz ...')
poem_loader, ix2word, word2ix = prepareData('../../Dataset/tang.npz', BATCH_SIZE)
print('Data loaded.')
print('-----------------------------------')

model = DoubleLSTM(len(word2ix), EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('caches/DoubleLSTM_19.pth', map_location=torch.device('cpu')))


st.title('🦜🔗 Cyber Poet')

with st.form('myform'):
    topic_text = st.text_input('Enter prompt:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        results = generate(model, topic_text, ix2word, word2ix, torch.device('cpu'))

        st.info("".join([i+str('  \n') if i in ['。', '！', '？'] else i for i in results]))
import streamlit as st
from main import generate
from config import Config
import torch
from model import Model
from utils import fPrint, get_data


opt = Config()
data, word2ix, ix2word = get_data(opt)

model = Model(len(word2ix), 64, 128)
model.load_state_dict(torch.load(r'caches/[2024-05-28-23_29_34]_99.pth', \
                                 map_location=torch.device("cpu")))

st.title('🦜🔗 Cyber Poet')

with st.form('myform'):
    topic_text = st.text_input('Enter prompt:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        st.info("".join([i+str('  \n') if i in ['。', '！', '？'] else i for i in generate(model, start_words=topic_text)]))
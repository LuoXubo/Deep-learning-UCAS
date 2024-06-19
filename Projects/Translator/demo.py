"""
@Description :   A online demo for machine translation.
@Author      :   Xubo Luo 
@Time        :   2024/06/19 20:07:55
"""
# coding:utf-8
import streamlit as st
from one_trans import machine_translate
from utils import *

"""
Usage:
    $ streamlit run demo.py
"""


st.title('🦜🔗 BERT Translator')
st.write('英译中翻译器')

with st.form('myform'):
    topic_text = st.text_input('输入要翻译的英文:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        U_src_lan= 'en'
        U_tgt_lan = 'zh'

        cn_trans = translate_large(topic_text, U_src_lan, U_tgt_lan)
        st.info(cn_trans)
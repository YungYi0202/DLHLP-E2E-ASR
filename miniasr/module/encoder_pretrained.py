'''
    File      [ encoder_pretrained.py ]
    Author    [ Yung-Yi Chen (NTUCSIE) ]
    Synopsis  [ Pretrained encoder. ]
'''

import torch
from torch import nn
from transformers import AutoModel

import os
from miniasr.utils import load_from_checkpoint

def asr_url(ckpt):
    '''
        ASR model from an url.
    '''
    ckpt = torch.hub.load_state_dict_from_url(ckpt)
    model, _, _ = load_from_checkpoint(ckpt, 'cpu')
    return model

def ctc_eng_ls960_hubert_base_char():
    '''
        Language: English
        Data: LibriSpeech 960h
        Feature: HuBERT base
        Vocab: char
    '''
    return asr_url('https://www.dropbox.com/s/1k3mpngqpinihlo/ctc_ls960_hubert_base_char.ckpt?dl=0')




class PretrainedEncoder(nn.Module):
    '''
        Pretrained encoder.
    '''

    def __init__(self, model_name):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        # self.model = ctc_eng_ls960_hubert_base_char()
        for p in self.model.parameters():
            # Unfreeze the pretrained model
            p.requires_grad = True

        # Output dimension
        self.out_dim = 768

    def forward(self, waveform: torch.Tensor, wave_len: torch.Tensor):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
                out_len [long tensor]: encoded feature lengths
        '''
        output = self.model(waveform)
        last_hidden_state = output.last_hidden_state
        
        enc_len = wave_len * ( last_hidden_state.shape[1] / torch.max(wave_len) )
        enc_len = enc_len.floor().to(dtype=torch.int)
        #print("enc_len")
        #print(enc_len)
        #print("last_hidden_state.shape")
        #print(last_hidden_state.shape)

        return last_hidden_state, enc_len


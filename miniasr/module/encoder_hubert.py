'''
    File      [ encoder_pretrained.py ]
    Author    [ Yung-Yi Chen (NTUCSIE) ]
    Synopsis  [ Pretrained encoder. ]
'''

import torch
from torch import nn
from transformers import AutoModel, HubertModel, HubertConfig

class HubertEncoder(nn.Module):
    '''
        Hubert encoder.
    '''

    # def __init__(self, model_name, trainable):
    def __init__(self):
        super().__init__()
        
        configuration = HubertConfig()
        self.model = HubertModel(configuration)
        
        for p in self.model.parameters():
            # Unfreeze the pretrained model
            p.requires_grad = True

        # Output dimension
        self.out_dim = self.model.config.hidden_size

    def forward(self, inputs: torch.Tensor, wave_len: torch.Tensor):
        output = self.model(inputs)
        last_hidden_state = output.last_hidden_state
        
        enc_len = wave_len * ( last_hidden_state.shape[1] / torch.max(wave_len) )
        enc_len = enc_len.floor().to(dtype=torch.int)

        return last_hidden_state, enc_len


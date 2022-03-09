'''
    File      [ encoder_pretrained.py ]
    Author    [ Yung-Yi Chen (NTUCSIE) ]
    Synopsis  [ Pretrained encoder. ]
'''

import torch
from torch import nn
from transformers import AutoModel, HubertModel, HubertConfig

class PretrainedEncoder(nn.Module):
    '''
        Pretrained encoder.
    '''

    def __init__(self, model_name, trainable, start_from_zero):
        super().__init__()
        print(start_from_zero)
        if start_from_zero == False:
            print("AutoModel from pretrained")
            self.model = AutoModel.from_pretrained(model_name)
        else:
            if 'hubert' in model_name or 'Hubert' in model_name or 'HuBERT' in model_name:
                print("Start from zero! HuBERT")
                configuration = HubertConfig()
                self.model = HubertModel(configuration)
            else:
                raise NotImplementedError(f'Unknown encoder {model_name}(start_from_zero).')
        #self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

        for p in self.model.parameters():
            # Unfreeze the pretrained model
            p.requires_grad = trainable

        # Output dimension
        self.out_dim = self.model.config.hidden_size

    def forward(self, inputs: torch.Tensor, wave_len: torch.Tensor):
        output = self.model(inputs)
        last_hidden_state = output.last_hidden_state
        
        enc_len = wave_len * ( last_hidden_state.shape[1] / torch.max(wave_len) )
        enc_len = enc_len.floor().to(dtype=torch.int)
        #print("enc_len")
        #print(enc_len)
        #print("last_hidden_state.shape")
        #print(last_hidden_state.shape)

        return last_hidden_state, enc_len


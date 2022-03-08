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

    def __init__(self, model_name, trainable):
        super().__init__()

        # self.model = AutoModel.from_pretrained(model_name)
        configuration = HubertConfig(conv_dim = (512, 512, 512, 512, 512, 512), conv_stride = (5, 2, 2, 2, 2, 2), conv_kernel = (10, 3, 3, 3, 2, 2))
        self.model = HubertModel(configuration)

        for p in self.model.parameters():
            # Unfreeze the pretrained model
            p.requires_grad = trainable

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


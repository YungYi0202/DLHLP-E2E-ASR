'''
    File      [ encoder_pretrained.py ]
    Author    [ Yung-Yi Chen (NTUCSIE) ]
    Synopsis  [ Pretrained encoder. ]
'''

import torch
from torch import nn
from transformers import AutoModel

class PretrainedEncoder(nn.Module):
    '''
        Pretrained encoder.
    '''

    def __init__(self, model_name):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            # Unfreeze the pretrained model
            p.requires_grad = True

        # Output dimension
        self.out_dim = 768

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
                out_len [long tensor]: encoded feature lengths
        '''

        output = self.model(feat)
        print("PretrainedEncoder: output.shape")
        print(output.shape)
        last_hidden_state = output.last_hidden_state
        # TODO: Directly return output?
        print("PretrainedEncoder: last_hidden_state[:,0,:].shape")
        print(last_hidden_state[:,0,:].shape)
        return last_hidden_state[:,0,:], feat_len


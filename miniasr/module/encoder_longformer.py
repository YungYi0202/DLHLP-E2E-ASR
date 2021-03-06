'''
    File      [ encoder_rnn.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Transformer encoder. ]
'''

import torch
from torch import nn
from transformers import LongformerConfig, LongformerModel


class LongformerEncoder(nn.Module):
    '''
        Transformer encoder.
        in_dim [int]: input feature dimension
        hid_dim [int]: hidden feature dimension
        n_layers [int]: number of layers
        dropout [float]: dropout rate
    '''

    def __init__(self, in_dim, hid_dim, n_layers, dropout=0, n_head=8):
        super().__init__()

        configuration = LongformerConfig()
        configuration.hidden_size = hid_dim
        configuration.num_hidden_layers = n_layers
        configuration.num_attention_heads = n_head
        self.encoder = LongformerModel(configuration)

        # Output dimension
        self.out_dim = in_dim

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
                out_len [long tensor]: encoded feature lengths
        '''

        #if not self.training:
        #    self.rnn.flatten_parameters()
        #print("TransformerEncoder: feat.shape")
        #print(feat.shape)
        #out, _ = self.rnn(feat)
        out = self.encoder(feat.transpose(0,1))
        #print("TransformerEncoder: out.shape")
        #print(out.shape)
        return out.transpose(0,1) , feat_len

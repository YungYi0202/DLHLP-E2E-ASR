'''
    File      [ encoder_rnn.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Transformer encoder. ]
'''

import torch
from torch import nn


class TransformerEncoder(nn.Module):
    '''
        Transformer encoder.
        in_dim [int]: input feature dimension
        hid_dim [int]: hidden feature dimension
        n_layers [int]: number of layers
        dropout [float]: dropout rate
    '''

    def __init__(self, in_dim, hid_dim, n_layers, dropout=0, n_head=8):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, 
            nhead=n_head,
            dim_feedforward=hid_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=n_layers
        )

        # Output dimension
        self.out_dim = 1000

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
        print("TransformerEncoder: feat.shape")
        print(feat.shape)
        #out, _ = self.rnn(feat)
        out = self.transformer_encoder(feat)
        print("TransformerEncoder: out.shape")
        print(out.shape)
        return out, feat_len

'''
    File      [ encoder_rnn.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Conformer encoder. ]
'''

import torch
from torch import nn
# import torchaudio
import conformer

class ConformerEncoder(nn.Module):
    '''
        Conformer encoder.
        in_dim [int]: input feature dimension
        hid_dim [int]: hidden feature dimension
        n_layers [int]: number of layers
        dropout [float]: dropout rate
    '''

    def __init__(self, in_dim, hid_dim, n_layers, dropout=0.0, n_head=8, conv_kernel=31):
        super().__init__()
        # self.model = torchaudio.models.Conformer(
        #             input_dim = in_dim, 
        #             num_heads = n_head, 
        #             ffn_dim = hid_dim, 
        #             num_layers = n_layers, 
        #             depthwise_conv_kernel_size = conv_kernel, 
        #             dropout = dropout
        #         )
        self.model = conformer.encoder.ConformerEncoder(
            input_dim = in_dim,
            encoder_dim = hid_dim,
            num_layers = n_layers,
            num_attention_heads = n_head,
            conv_kernel_size = conv_kernel,
            conv_dropout_p = dropout
        )

        # Output dimension
        self.out_dim = hid_dim

    def forward(self, feat: torch.Tensor, feat_len: torch.Tensor):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
                out_len [long tensor]: encoded feature lengths
        '''

        out, out_len= self.model(feat, feat_len)
        
        return out, out_len

'''
    File      [ encoder_rnn.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ RNN-based encoder. ]
'''

import torch
from torch import nn


class CLDNNEncoder(nn.Module):

    def __init__(self, in_dim, args):
        super().__init__()
        # input: (N, 1, seq_len, feature_len = in_dim)
        self.conv1 = args.model.encoder.conv1
        self.maxpool = args.model.encoder.maxpool
        self.conv2 = args.model.encoder.conv2
        self.lstm = args.model.encoder.lstm
        self.reduced_feat_dim = args.model.encoder.reduced_feat_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=self.conv1.out_channels, 
                kernel_size=self.conv1.kernel, 
                padding=self.conv1.padding
            ),
            nn.MaxPool2d(self.maxpool.kernel, self.maxpool.stride, self.maxpool.padding),
            nn.Conv2d(
                in_channels=self.conv1.out_channels,
                out_channels=self.conv2.out_channels, 
                kernel_size=self.conv2.kernel, 
                padding=self.conv2.padding
            )
        )
        
        # For CNN layers
        self.conv_out_dim = self.conv2.out_channels * self.cal_conv_layers_out_feature_dim(in_dim)
        
        # A layer to reduce the parameters
        self.reduced_layer = nn.Linear(
            self.conv_out_dim, self.reduced_feat_dim)

        # LSTM model
        self.lstm_block = getattr(nn, 'GRU')(
            input_size=self.reduced_feat_dim,
            hidden_size=self.lstm.hid_dim,
            num_layers=self.lstm.n_layers,
            dropout=self.lstm.dropout,
            bidirectional=self.lstm.bidirectional,
            batch_first=True)

        # Output dimension = output feature length
        self.out_dim = self.lstm.hid_dim * (2 if self.lstm.bidirectional else 1)
    
    def cal_out_dim(self, in_dim, padding, kernel, stride):
        return int((in_dim + 2 * padding - kernel)/stride + 1)
    
    def cal_conv_layers_out_feature_dim(self, in_dim):
        conv1_dim = self.cal_out_dim(in_dim, self.conv1.padding, self.conv1.kernel, 1)
        maxpool_dim = self.cal_out_dim(conv1_dim, self.maxpool.padding, self.maxpool.kernel, self.maxpool.stride)
        conv2_dim = self.cal_out_dim(maxpool_dim, self.conv2.padding, self.conv2.kernel, 1)
        return conv2_dim
        

    def forward(self, feat: torch.Tensor, feat_seq_len: torch.Tensor):
        '''
            Input:
                feat [float tensor]: acoustic feature sequence
                feat_seq_len [long tensor]: feature lengths
            Output:
                out [float tensor]: encoded feature sequence
                out_len [long tensor]: encoded feature lengths
        '''
        # feat.shape = [batch_size, seq_len, feature_dim]
        # print("CLDNNEncoder: feat.shape")
        # print(feat.shape)
        feat = feat.view(feat.shape[0], 1, feat.shape[1], feat.shape[2])
        # feat.shape = [batch_size, 1, seq_len, feature_dim]

        out = self.conv_layers(feat)
        # out.shape = [batch_size, chanels, new seq_len, new feature_dim]
        out = out.transpose(1,2)
        # out.shape = [batch_size, new seq_len, chanels, new feature_dim]
        out = out.flatten(2)
        # out.shape = [batch_size, new seq_len, out_dim]

        # Ratioly modify the lens.
        enc_len = feat_seq_len * ( out.shape[1] / torch.max(feat_seq_len) )
        enc_len = enc_len.floor().to(dtype=torch.int)

        # print("CLDNNEncoder: out.shape")
        # print(out.shape)
        # print(f"CLDNNEncoder: self.out_dim: {self.out_dim}")
        
        #-----End Conv Layers
        out = self.reduced_layer(out)
        # out.shape = [batch_size, new seq_len, out_dim]
        out, _ = self.lstm_block(out, None)

        return out, enc_len

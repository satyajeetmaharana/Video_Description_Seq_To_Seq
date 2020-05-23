import torch.nn as nn


class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, target_variable=None, mode='train', opt={}):
        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target_variable, mode, opt)
        return seq_prob, seq_preds

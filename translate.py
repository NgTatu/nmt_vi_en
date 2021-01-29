import torch
from d2l import torch as d2lt
from torch import nn
from dataset import load_data_nmt
from model import Seq2SeqEncoder, Seq2SeqDecoder, EncoderDecoder
from loss import MaskedSoftmaxCELoss
from CONFIG import *
from pathlib import Path
import os

import math
import collections


def predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device):
    """Predict sequences (defined in Chapter 9)."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2lt.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Once the end-of-sequence token is predicted, the generation of
        # the output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return (' '.join(tgt_vocab.to_tokens(output_seq))).replace('_', ' ')


def bleu(pred_seq, label_seq, k):  # @save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def translate(engs, fras, model, src_vocab, tgt_vocab, num_steps, device):
    """Translate text sequences."""
    for eng, fra in zip(engs, fras):
        translation = predict_s2s_ch9(
            model, eng, src_vocab, tgt_vocab, num_steps, device)
        print(
            f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

def evaluate(engs, vies, model, src_vocab, tgt_vocab, num_steps, device):
  blue_score = 0
  for eng, vie in zip(engs, vies):
    translation = predict_s2s_ch9(
            model, eng, src_vocab, tgt_vocab, num_steps, device)
    blue_score += bleu(translation, vie, k=2)
  return blue_score/len(engs)
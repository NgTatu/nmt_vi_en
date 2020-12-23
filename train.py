import torch
from d2l import torch as d2lt
from torch import nn
from dataset import load_data_nmt
from model import Seq2SeqEncoder, Seq2SeqDecoder, EncoderDecoder
from loss import MaskedSoftmaxCELoss
from CONFIG import *
from pathlib import Path
import os

PATH_MODEL = Path(PATH_MODEL)
checkpoint_path = MODEL_CHECKPOINT_PATH


def xavier_init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(m._parameters[param])


def train(resume_training=True):
    """Train a model for sequence to sequence (defined in Chapter 9)."""
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2lt.try_gpu()

    ### Load data
    data_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

    ### Load model
    model = EncoderDecoder(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ### Load checkpoint
    if resume_training and PATH_MODEL.exists() and os.path.getsize(PATH_MODEL) > 0:
        model, optimizer, last_epoch = load_checkpoint(model, optimizer)
        print("Continue training from last checkpoint...")
    else:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open(PATH_MODEL, 'w') as fp:
            pass
        print('No prior checkpoint existed, created new save files for checkpoint.')
        model.apply(xavier_init_weights)
        last_epoch = 0

    # model.apply(xavier_init_weights)
    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ### Initialize Loss functions
    loss = MaskedSoftmaxCELoss()

    ### Train
    model.train()
    # animator = d2lt.Animator(xlabel='epoch', ylabel='loss',
    # xlim=[10, num_epochs])
    for epoch in range(last_epoch, num_epochs):
        timer = d2lt.Timer()
        metric = d2lt.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2lt.grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            # animator.add(epoch + 1, (metric[0] / metric[1],))
            print(f'epoch {epoch + 1} - ' f'loss {metric[0] / metric[1]:.3f}')

        ### Save checkpoint
        save_checkpoint(epoch, model, optimizer)
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def save_checkpoint(epoch, model, optimizer):
    checkpoint_model = {
        'epoch': epoch,
        'model': model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint_model, PATH_MODEL)
    if (epoch + 1) % 10 == 0:
        print("Save checkpoint successfully!")


def load_checkpoint(model, optimizer):
    checkpoint_model = torch.load(PATH_MODEL)
    model.load_state_dict(checkpoint_model['model_state_dict'])
    optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
    epoch = checkpoint_model['epoch']
    print('Load checkpoint successfully! Start training from epoch: ' + str(epoch))
    return model, optimizer, epoch


if __name__ == '__main__':
    train()


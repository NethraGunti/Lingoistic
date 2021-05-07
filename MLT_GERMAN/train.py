import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import time
from model import Transformer, LabelSmoothedCE
from dataloader import SequenceLoader
from utils import *


data_folder = './transformer data'

# Model parameters
d_model = 512
n_heads = 8
d_queries = 64
d_values = 64
d_inner = 2048
n_layers = 6
dropout = 0.1
positional_encoding = get_positional_encoding(d_model=d_model,
                                              max_length=160)

# Hyperparameters
checkpoint = 'transformer_checkpoint.pth.tar'
tokens_in_batch = 2000
batches_per_step = 25000 // tokens_in_batch
print_frequency = 20
n_steps = 100000
warmup_steps = 8000
step = 1
lr = get_lr(step=step, d_model=d_model,
            warmup_steps=warmup_steps)
start_epoch = 0
betas = (0.9, 0.98)
epsilon = 1e-9
label_smoothing = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = False


def main():
    global checkpoint, step, start_epoch, epoch, epochs

    train_loader = SequenceLoader(data_folder=data_folder
                                  source_suffix="en",
                                  target_suffix="de",
                                  split="train",
                                  tokens_in_batch=tokens_in_batch)
    val_loader = SequenceLoader(data_folder=data_folder
                                source_suffix="en",
                                target_suffix="de",
                                split="val",
                                tokens_in_batch=tokens_in_batch)

    if checkpoint is None:
        model = Transformer(vocab_size=train_loader.bpe_model.vocab_size(),
                            positional_encoding=positional_encoding,
                            d_model=d_model,
                            n_heads=n_heads,
                            d_queries=d_queries,
                            d_values=d_values,
                            d_inner=d_inner,
                            n_layers=n_layers,
                            dropout=dropout)
        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad],
                                     lr=lr,
                                     betas=betas,
                                     eps=epsilon)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    criterion = LabelSmoothedCE(eps=label_smoothing)

    model = model.to(device)
    criterion = criterion.to(device)

    epochs = (n_steps // (train_loader.n_batches // batches_per_step)) + 1

    for epoch in range(start_epoch, epochs):
        step = epoch * train_loader.n_batches // batches_per_step

        # One epoch's training
        train_loader.create_batches()
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              step=step)

        # One epoch's validation
        val_loader.create_batches()
        validate(val_loader=val_loader,
                 model=model,
                 criterion=criterion)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch, step):
    model.train()

    data_time = AverageMeter()  # data loading time
    step_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss

    # Starting time
    start_data_time = time.time()
    start_step_time = time.time()

    # Batches
    for i, (source_sequences, target_sequences, source_sequence_lengths, target_sequence_lengths) in enumerate(
            train_loader):

        source_sequences = source_sequences.to(device)
        target_sequences = target_sequences.to(device)
        source_sequence_lengths = source_sequence_lengths.to(device)
        target_sequence_lengths = target_sequence_lengths.to(device)

        # Time taken to load data
        data_time.update(time.time() - start_data_time)

        # Forward prop.
        predicted_sequences = model(source_sequences, target_sequences, source_sequence_lengths,
                                    target_sequence_lengths)

        loss = criterion(inputs=predicted_sequences,
                         targets=target_sequences[:, 1:],
                         lengths=target_sequence_lengths - 1)  # scalar

        # Backward prop.
        (loss / batches_per_step).backward()

        losses.update(loss.item(), (target_sequence_lengths - 1).sum().item())

        if (i + 1) % batches_per_step == 0:
            optimizer.step()
            optimizer.zero_grad()

            step += 1

            change_lr(optimizer, new_lr=get_lr(step=step, d_model=d_model, warmup_steps=warmup_steps))

            step_time.update(time.time() - start_step_time)

            if step % print_frequency == 0:
                print('Epoch {0}/{1}-----'
                      'Batch {2}/{3}-----'
                      'Step {4}/{5}-----'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----'
                      'Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                      'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(epoch + 1, epochs,
                                                                        i + 1, train_loader.n_batches,
                                                                        step, n_steps,
                                                                        step_time=step_time,
                                                                        data_time=data_time,
                                                                        losses=losses))

            start_step_time = time.time()

            if epoch in [epochs - 1, epochs - 2] and step % 1500 == 0:  # 'epoch' is 0-indexed
                save_checkpoint(epoch, model, optimizer, prefix='step' + str(step) + "_")

        start_data_time = time.time()


def validate(val_loader, model, criterion):
    model.eval()

    with torch.no_grad():
        losses = AverageMeter()
        for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
                tqdm(val_loader, total=val_loader.n_batches)):
            source_sequence = source_sequence.to(device)
            target_sequence = target_sequence.to(device)
            source_sequence_length = source_sequence_length.to(device)
            target_sequence_length = target_sequence_length.to(device)

            # Forward prop.
            predicted_sequence = model(source_sequence, target_sequence, source_sequence_length,
                                       target_sequence_length)

            loss = criterion(inputs=predicted_sequence,
                             targets=target_sequence[:, 1:],
                             lengths=target_sequence_length - 1)

            losses.update(loss.item(), (target_sequence_length - 1).sum().item())

        print("\nValidation loss: %.3f\n\n" % losses.avg)


if __name__ == '__main__':
    main()

import math
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn


class ParallelLanguageDataset(Dataset):
    def __init__(self, data_path_1, data_path_2, tokenizer, max_len, integrate=False):
        self.data_1, self.data_2 = self.load_data(data_path_1, data_path_2)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.integrate = integrate
 
    def __len__(self):
        return len(self.data_1)
 
    def __getitem__(self, item_idx):
        sent1 = str(self.data_1[item_idx])
        encoded_output_sent1 = self.tokenizer.encode_plus(
            sent1,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
            truncation=True
        )
        encoded_output_sent1["attention_mask"][
            encoded_output_sent1["attention_mask"] == 1
        ] = 2
        encoded_output_sent1["attention_mask"][
            encoded_output_sent1["attention_mask"] == 0
        ] = True
        encoded_output_sent1["attention_mask"][
            encoded_output_sent1["attention_mask"] == 2
        ] = False
        encoded_output_sent1["attention_mask"] = encoded_output_sent1[
            "attention_mask"
        ].type(torch.bool)
        
        if not self.integrate:
            sent2 = str(self.data_2[item_idx])
            encoded_output_sent2 = self.tokenizer.encode_plus(
                sent2,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt",
                truncation=True
            )
    
            encoded_output_sent2["attention_mask"][
                encoded_output_sent2["attention_mask"] == 1
            ] = 2
            encoded_output_sent2["attention_mask"][
                encoded_output_sent2["attention_mask"] == 0
            ] = True
            encoded_output_sent2["attention_mask"][
                encoded_output_sent2["attention_mask"] == 2
            ] = False
            encoded_output_sent2["attention_mask"] = encoded_output_sent2[
                "attention_mask"
            ].type(torch.bool)
    
            return_dict = {
                "ids1": encoded_output_sent1["input_ids"].flatten(),
                "ids2": encoded_output_sent2["input_ids"].flatten(),
                "masks_sent1": encoded_output_sent1["attention_mask"].flatten(),
                "masks_sent2": encoded_output_sent2["attention_mask"].flatten(),
            }
        else:
            return_dict = {
                "ids1": encoded_output_sent1["input_ids"].flatten(),
                "masks_sent1": encoded_output_sent1["attention_mask"].flatten(),
            }            
        return return_dict
 
    def load_data(self, data_path_1, data_path_2):
        if self.integrate:
            data_1 = data_path_1
            data_2 = data_path_2
        else:
            with open(data_path_1, "r") as f:
                data_1 = f.read().splitlines()[1:200]
            with open(data_path_2, "r") as f:
                data_2 = f.read().splitlines()[1:200]
        return data_1, data_2

class LanguageTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        pos_dropout,
        trans_dropout,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(vocab_size, d_model)
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            trans_dropout,
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        tgt_mask,
    ):
        src = rearrange(src, "n s -> s n")
        tgt = rearrange(tgt, "n t -> t n")
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = rearrange(output, "t n e -> n t e")
        return self.fc(output)


# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


'''A wrapper class for optimizer '''
# From https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            

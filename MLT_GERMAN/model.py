import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries

        self.in_decoder = in_decoder

        self.cast_queries = nn.Linear(d_model, n_heads * d_queries)

        self.cast_keys_values = nn.Linear(d_model, n_heads * (d_queries + d_values))

        self.cast_output = nn.Linear(n_heads * d_values, d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm = nn.LayerNorm(d_model)

        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, query_sequences, key_value_sequences, key_value_sequence_lengths):
        batch_size = query_sequences.size(0)
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_value_sequences.size(1)

        self_attention = torch.equal(key_value_sequences, query_sequences)

        input_to_add = query_sequences.clone()

        query_sequences = self.layer_norm(query_sequences)

        if self_attention:
            key_value_sequences = self.layer_norm(key_value_sequences)

        queries = self.cast_queries(query_sequences)
        keys, values = self.cast_keys_values(key_value_sequences).split(split_size=self.n_heads * self.d_keys,
                                                                        dim=-1)

        queries = queries.contiguous().view(batch_size, query_sequence_pad_length, self.n_heads,
                                            self.d_queries)
        keys = keys.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads,
                                      self.d_keys)
        values = values.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads,
                                          self.d_values)

        queries = queries.permute(0, 2, 1, 3).contiguous().view(-1, query_sequence_pad_length,
                                                                self.d_queries)
        keys = keys.permute(0, 2, 1, 3).contiguous().view(-1, key_value_sequence_pad_length,
                                                          self.d_keys)
        values = values.permute(0, 2, 1, 3).contiguous().view(-1, key_value_sequence_pad_length,
                                                              self.d_values)

        attention_weights = torch.bmm(queries, keys.permute(0, 2,
                                                            1))

        attention_weights = (1. / math.sqrt(
            self.d_keys)) * attention_weights

        not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand_as(
            attention_weights).to(device)
        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.n_heads).unsqueeze(
            1).unsqueeze(2).expand_as(
            attention_weights)
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float(
            'inf'))

        if self.in_decoder and self_attention:
            not_future_mask = torch.ones_like(
                attention_weights).tril().bool().to(
                device)

            attention_weights = attention_weights.masked_fill(~not_future_mask, -float(
                'inf'))

        attention_weights = self.softmax(
            attention_weights)

        attention_weights = self.apply_dropout(
            attention_weights)
        sequences = torch.bmm(attention_weights, values)

        sequences = sequences.contiguous().view(batch_size, self.n_heads, query_sequence_pad_length,
                                                self.d_values).permute(0, 2, 1,
                                                                       3)

        sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length,
                                                -1)

        sequences = self.cast_output(sequences)

        sequences = self.apply_dropout(sequences) + input_to_add

        return sequences


class PositionWiseFCNetwork(nn.Module):
    
    def __init__(self, d_model, d_inner, dropout):
        super(PositionWiseFCNetwork, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_inner)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_inner, d_model)
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        input_to_add = sequences.clone()
        sequences = self.layer_norm(sequences)
        sequences = self.apply_dropout(self.relu(self.fc1(sequences)))
        sequences = self.fc2(sequences)
        sequences = self.apply_dropout(sequences) + input_to_add
        return sequences


class Encoder(nn.Module):
    
    def __init__(self, vocab_size, positional_encoding, d_model, n_heads, d_queries, d_values, d_inner, n_layers,
                 dropout):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding.requires_grad = False
        self.encoder_layers = nn.ModuleList([self.make_encoder_layer() for i in range(n_layers)])
        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def make_encoder_layer(self):
        encoder_layer = nn.ModuleList([MultiHeadAttention(d_model=self.d_model,
                                                          n_heads=self.n_heads,
                                                          d_queries=self.d_queries,
                                                          d_values=self.d_values,
                                                          dropout=self.dropout,
                                                          in_decoder=False),
                                       PositionWiseFCNetwork(d_model=self.d_model,
                                                             d_inner=self.d_inner,
                                                             dropout=self.dropout)])

        return encoder_layer

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        pad_length = encoder_sequences.size(1)
        encoder_sequences = self.embedding(encoder_sequences) * math.sqrt(self.d_model) + self.positional_encoding[:,
                                                                                          :pad_length, :].to(
            device)

        encoder_sequences = self.apply_dropout(encoder_sequences)

        for encoder_layer in self.encoder_layers:
            encoder_sequences = encoder_layer[0](query_sequences=encoder_sequences,
                                                 key_value_sequences=encoder_sequences,
                                                 key_value_sequence_lengths=encoder_sequence_lengths)
            encoder_sequences = encoder_layer[1](sequences=encoder_sequences)

        encoder_sequences = self.layer_norm(encoder_sequences)

        return encoder_sequences


class Decoder(nn.Module):
    def __init__(self, vocab_size, positional_encoding, d_model, n_heads, d_queries, d_values, d_inner, n_layers,
                 dropout):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding.requires_grad = False
        self.decoder_layers = nn.ModuleList([self.make_decoder_layer() for i in range(n_layers)])
        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def make_decoder_layer(self):
        decoder_layer = nn.ModuleList([MultiHeadAttention(d_model=self.d_model,
                                                          n_heads=self.n_heads,
                                                          d_queries=self.d_queries,
                                                          d_values=self.d_values,
                                                          dropout=self.dropout,
                                                          in_decoder=True),
                                       MultiHeadAttention(d_model=self.d_model,
                                                          n_heads=self.n_heads,
                                                          d_queries=self.d_queries,
                                                          d_values=self.d_values,
                                                          dropout=self.dropout,
                                                          in_decoder=True),
                                       PositionWiseFCNetwork(d_model=self.d_model,
                                                             d_inner=self.d_inner,
                                                             dropout=self.dropout)])

        return decoder_layer

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        pad_length = decoder_sequences.size(1)
        decoder_sequences = self.embedding(decoder_sequences) * math.sqrt(self.d_model) + self.positional_encoding[:,
                                                                                          :pad_length, :].to(
            device)

        decoder_sequences = self.apply_dropout(decoder_sequences)
        for decoder_layer in self.decoder_layers:
            decoder_sequences = decoder_layer[0](query_sequences=decoder_sequences,
                                                 key_value_sequences=decoder_sequences,
                                                 key_value_sequence_lengths=decoder_sequence_lengths)
            decoder_sequences = decoder_layer[1](query_sequences=decoder_sequences,
                                                 key_value_sequences=encoder_sequences,
                                                 key_value_sequence_lengths=encoder_sequence_lengths)
            decoder_sequences = decoder_layer[2](sequences=decoder_sequences)
        
        decoder_sequences = self.layer_norm(decoder_sequences)

        decoder_sequences = self.fc(decoder_sequences)

        return decoder_sequences


class Transformer(nn.Module):
    
    def __init__(self, vocab_size, positional_encoding, d_model=512, n_heads=8, d_queries=64, d_values=64,
                 d_inner=2048, n_layers=6, dropout=0.1):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout
        self.encoder = Encoder(vocab_size=vocab_size,
                               positional_encoding=positional_encoding,
                               d_model=d_model,
                               n_heads=n_heads,
                               d_queries=d_queries,
                               d_values=d_values,
                               d_inner=d_inner,
                               n_layers=n_layers,
                               dropout=dropout)

        self.decoder = Decoder(vocab_size=vocab_size,
                               positional_encoding=positional_encoding,
                               d_model=d_model,
                               n_heads=n_heads,
                               d_queries=d_queries,
                               d_values=d_values,
                               d_inner=d_inner,
                               n_layers=n_layers,
                               dropout=dropout)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.)
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=math.pow(self.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.decoder.fc.weight = self.decoder.embedding.weight

        print("Model initialized.")

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths):
        encoder_sequences = self.encoder(encoder_sequences,
                                         encoder_sequence_lengths)

        decoder_sequences = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences,
                                         encoder_sequence_lengths)

        return decoder_sequences


class LabelSmoothedCE(torch.nn.Module):
    
    def __init__(self, eps=0.1):
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, lengths):
        inputs, _, _, _ = pack_padded_sequence(input=inputs,
                                               lengths=lengths,
                                               batch_first=True,
                                               enforce_sorted=False)
        targets, _, _, _ = pack_padded_sequence(input=targets,
                                                lengths=lengths,
                                                batch_first=True,
                                                enforce_sorted=False)

        target_vector = torch.zeros_like(inputs).scatter(dim=1, index=targets.unsqueeze(1),
                                                         value=1.).to(device)
        target_vector = target_vector * (1. - self.eps) + self.eps / target_vector.size(
            1)

        loss = (-1 * target_vector * F.log_softmax(inputs, dim=1)).sum(dim=1)

        loss = torch.mean(loss)

        return loss

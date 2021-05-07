import torch
import torch.nn.functional as F
import youtokentome
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bpe_model = youtokentome.BPE(model="./transformer data/bpe.model")

checkpoint = torch.load("averaged_transformer_checkpoint.pth.tar")
model = checkpoint['model'].to(device)
model.eval()


def translate(source_sequence, beam_size=4, length_norm_coefficient=0.6):
    with torch.no_grad():
        k = beam_size

        n_completed_hypotheses = min(k, 10)

        vocab_size = bpe_model.vocab_size()

        if isinstance(source_sequence, str):
            encoder_sequences = bpe_model.encode(source_sequence,
                                                 output_type=youtokentome.OutputType.ID,
                                                 bos=False,
                                                 eos=False)
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0)
        else:
            encoder_sequences = source_sequence
        encoder_sequences = encoder_sequences.to(device)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(device)

        encoder_sequences = model.encoder(encoder_sequences=encoder_sequences,
                                          encoder_sequence_lengths=encoder_sequence_lengths)

        hypotheses = torch.LongTensor([[bpe_model.subword_to_id('<BOS>')]]).to(device)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)

        hypotheses_scores = torch.zeros(1).to(device)

        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        step = 1

        while True:
            s = hypotheses.size(0)
            decoder_sequences = model.decoder(decoder_sequences=hypotheses,
                                              decoder_sequence_lengths=hypotheses_lengths,
                                              encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                                              encoder_sequence_lengths=encoder_sequence_lengths.repeat(
                                                  s))

            scores = decoder_sequences[:, -1, :]
            scores = F.log_softmax(scores, dim=-1)

            scores = hypotheses_scores.unsqueeze(1) + scores

            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)

            prev_word_indices = unrolled_indices // vocab_size
            next_word_indices = unrolled_indices % vocab_size 

            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                         dim=1)

            complete = next_word_indices == bpe_model.subword_to_id('<EOS>')

            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            hypotheses = top_k_hypotheses[~complete]
            hypotheses_scores = top_k_hypotheses_scores[~complete]
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)

            if step > 100:
                break
            step += 1

        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()

        all_hypotheses = list()
        for i, h in enumerate(bpe_model.decode(completed_hypotheses, ignore_ids=[0, 2, 3])):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses


if __name__ == '__main__':
    translate("It was the best of times, it was the worst of times.")

import torch
import sacrebleu
from translate import translate
from tqdm import tqdm
from dataloader import SequenceLoader
import youtokentome
import codecs
import os

sacrebleu_in_python = False
test_loader = SequenceLoader(data_folder="./transformer data",
                             source_suffix="en",
                             target_suffix="de",
                             split="test",
                             tokens_in_batch=None)
test_loader.create_batches()
with torch.no_grad():
    hypotheses = list()
    references = list()
    for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
            tqdm(test_loader, total=test_loader.n_batches)):
        hypotheses.append(translate(source_sequence=source_sequence,
                                    beam_size=4,
                                    length_norm_coefficient=0.6)[0])
        references.extend(test_loader.bpe_model.decode(target_sequence.tolist(), ignore_ids=[0, 2, 3]))
    if sacrebleu_in_python:
        print("\n13a tokenization, cased:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references]))
        print("\n13a tokenization, caseless:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True))
        print("\nInternational tokenization, cased:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'))
        print("\nInternational tokenization, caseless:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True))
        print("\n")
    else:
        with codecs.open("translated_test.de", "w", encoding="utf-8") as f:
            f.write("\n".join(hypotheses))
        print("\n13a tokenization, cased:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de")
        print("\n13a tokenization, caseless:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -lc")
        print("\nInternational tokenization, cased:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -tok intl")
        print("\nInternational tokenization, caseless:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -tok intl -lc")
        print("\n")
    
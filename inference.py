import torch
import os
from fairseq.models.bart import BARTModel
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

test_path = "QMSum/test.source"
output_path = "QMSum/test.hypo"

print("input:", test_path)
print("output:", output_path)

bart = BARTModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='EXP/QMSum-bin'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 2
with open(test_path) as source, open(output_path, 'w') as fout:
    for sline in tqdm(source):
        sline = sline.strip()
        with torch.no_grad():
            try:
                hypothese = bart.sample(sline)
                fout.write(hypothese + '\n')
            except:
                print(len(sline), sline)
                fout.write("NA\n")
            fout.flush()

    # three bugs in this version: no change with different epochs, low performance, bugs!

    # sline = source.readline().strip()
    # slines = [sline]
    # for sline in tqdm(source):
    #     if count % bsz == 0:
    #         with torch.no_grad():
    #             # hypotheses_batch = bart.sample(slines, max_len_a=0.1, max_len_b=15)
    #             hypotheses_batch = bart.sample(slines)
    #             # , beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3
    #         for hypothesis in hypotheses_batch:
    #             fout.write(hypothesis + '\n')
    #             fout.flush()
    #         slines = []
    #
    #     slines.append(sline.strip())
    #     count += 1
    # if slines != []:
    #     hypotheses_batch = bart.sample(slines)
    #     for hypothesis in hypotheses_batch:
    #         fout.write(hypothesis + '\n')
    #         fout.flush()

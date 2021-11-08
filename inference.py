import torch
import os
from fairseq.models.bart import BARTModel
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

test_path = "QMSum/test.source"
output_path = "QMSum/test_bart_cnn.hypo"

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
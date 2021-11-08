# LongDialSumm
The source code and scripts for our paper: "[An Exploratory Study on Long Dialogue Summarization:What Works and What’s Next](https://arxiv.org/pdf/2109.04609.pdf)".

### Running on QMSum
- Install [fairseq](https://github.com/pytorch/fairseq) and the other dependencies in requirements.txt.
- Run `bash run.sh` shell script to finetune the model (a BART-large-cnn model will be downloaded automaticly as the start point of training).
- Run `python inference.py` to generate summaries using the finetuned model.
- Evaluate the summaries and generate ROUGE scores. In this project, we leverage [AnyROUGE](https://github.com/chatc/AnyROUGE).

### Running on Other Datasets
- Install [fairseq](https://github.com/pytorch/fairseq) and dependencies.
- Prepare the dataset as shown in [fairseq](https://github.com/pytorch/fairseq) website, i.e. each sample takes up one line. (e.g /QMSum/test.source)
- Modify the names and parameters in `run.sh` and `inference.py`.
- Do the rest of the steps in the "Running Example" session.
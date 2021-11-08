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
- Do the rest of the steps in the "Running on QMSum" session.

### Citation
``` bibtex
@inproceedings{zhang-etal-2021-exploratory-study,
    title = "An Exploratory Study on Long Dialogue Summarization: What Works and What{'}s Next",
    author = "Zhang, Yusen  and
      Ni, Ansong  and
      Yu, Tao  and
      Zhang, Rui  and
      Zhu, Chenguang  and
      Deb, Budhaditya  and
      Celikyilmaz, Asli  and
      Awadallah, Ahmed Hassan  and
      Radev, Dragomir",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.377",
    pages = "4426--4433",
    abstract = "Dialogue summarization helps readers capture salient information from long conversations in meetings, interviews, and TV series. However, real-world dialogues pose a great challenge to current summarization models, as the dialogue length typically exceeds the input limits imposed by recent transformer-based pre-trained models, and the interactive nature of dialogues makes relevant information more context-dependent and sparsely distributed than news articles. In this work, we perform a comprehensive study on long dialogue summarization by investigating three strategies to deal with the lengthy input problem and locate relevant information: (1) extended transformer models such as Longformer, (2) retrieve-then-summarize pipeline models with several dialogue utterance retrieval methods, and (3) hierarchical dialogue encoding models such as HMNet. Our experimental results on three long dialogue datasets (QMSum, MediaSum, SummScreen) show that the retrieve-then-summarize pipeline models yield the best performance. We also demonstrate that the summary quality can be further improved with a stronger retrieval model and pretraining on proper external summarization datasets.",
}
```

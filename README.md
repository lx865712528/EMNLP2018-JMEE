# Jointly Multiple Events Extraction (JMEE)
This is the code of the Jointly Multiple Events Extraction (JMEE) in [our EMNLP 2018 paper](https://arxiv.org/abs/1809.09078).

### Updated Answers

1. We upload the data split files `qi_filelist` using in preprocessing with stanford corenlp.
2. We provide an example calling the runner for training.

### Requirement
- python 3
- [pytorch](http://pytorch.org) == 0.4.0
- [torchtext](https://github.com/pytorch/text) == 0.2.3
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [seqeval](https://github.com/chakki-works/seqeval)

To install the requirements, run `pip -r requirements.txt`.

### How to run the code?
After preprocessing the ACE 2005 dataset and put it under `ace-05-splits`, the main entrance is in `enet/run/ee/runner.py`.
We cannot include the data in this release due to licence issues.

But we offer a piece of data sample in `ace-05-splits/sample.json`, the format should be followed.

THE CODE IS A BASIC PRELIMINARY VERSION AND IS LIKE "AS IS", WITHOUT WARRANTY OF ANY KIND.

### Cite
Please cite our EMNLP 2018 paper:
```bibtex
@inproceedings{DBLP:conf/emnlp/LiuLH18,
  author    = {Xiao Liu and
               Zhunchen Luo and
               Heyan Huang},
  title     = {Jointly Multiple Events Extraction via Attention-based Graph Information
               Aggregation},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural
               Language Processing, Brussels, Belgium, October 31 - November 4, 2018},
  pages     = {1247--1256},
  year      = {2018},
  crossref  = {DBLP:conf/emnlp/2018},
  url       = {https://aclanthology.info/papers/D18-1156/d18-1156},
  timestamp = {Sat, 27 Oct 2018 20:04:50 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/emnlp/LiuLH18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

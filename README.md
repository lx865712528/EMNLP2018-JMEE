# Jointly Multiple Events Extraction (JMEE)
This is the code of the Jointly Multiple Events Extraction (JMEE) in [our EMNLP 2018 paper](https://arxiv.org/abs/1809.09078).

### Requirement
- python 3
- [pytorch](http://pytorch.org) >= 0.4.0
- [torchtext](https://github.com/pytorch/text) >= 0.2.3
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
@inproceedings{liu2018,
  author    = {Xiao Liu and
               Zhunchen Luo and
               Heyan Huang},
  title     = {Jointly Multiple Events Extraction via Attention-based Graph Information
               Aggregation},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural
               Language Processing},
  year      = {2018},
  url       = {http://arxiv.org/abs/1809.09078}
}
```

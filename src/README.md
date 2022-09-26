# README

```bash
git clone https://github.com/yeyupiaoling/AudioClassification-Pytorch.git
```

## Libraries and dependencies
- [pytorch](https://github.com/pytorch/pytorch)
- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [fairseq](https://github.com/pytorch/fairseq) (For Wav2vec)
- [huggingface transformers](https://huggingface.co) (For Wav2vec2)
- [faiss](https://github.com/facebookresearch/faiss) (For running clustering)
- [tal-edu-bert](https://github.com/tal-tech/edu-bert)

## dataset 
### ESD
- Publicly Available Emotional Speech Dataset (ESD) for Speech Synthesis and Voice Conversion 用于语音合成和语音转换的公开情感语音数据集
    - 这个数据集包含了10个以普通话为母语的人和10个以英语为母语的人所说的350个平行的话语，这些话语有5种情绪状态(中性、快乐、愤怒、悲伤和惊讶)。提供了文字记录。https://github.com/HLTSingapore/Emotional-Speech-Data

```
@inproceedings{zhou2021seen,
  title={Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset},
  author={Zhou, Kun and Sisman, Berrak and Liu, Rui and Li, Haizhou},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={920--924},
  year={2021},
  organization={IEEE}
}
@article{zhou2021emotional,
title = {Emotional voice conversion: Theory, databases and ESD},
journal = {Speech Communication},
volume = {137},
pages = {1-18},
year = {2022},
issn = {0167-6393}
}
```

### CASIA

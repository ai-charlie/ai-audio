import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()
        print(config.num_labels)
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


# load model from hub
device = 'cuda'
model_name = 'facebook/wav2vec2-base'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model_ft = EmotionModel.from_pretrained(model_name)

# 法1：冻结卷积层
for param in model_ft.parameters():
    param.requires_grad = False

# print("conv1.weights[0, 0, ...]".format(model_ft.conv1.weight[0, 0, ...]))


# dummy signal
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)


# define loss function (criterion), optimizer, and learning rate scheduler
# criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda(args.gpu)

# optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)

# """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model_ft(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

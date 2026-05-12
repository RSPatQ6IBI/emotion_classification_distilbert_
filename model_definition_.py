

import torch
from torch import nn
import torch.nn.functional as F

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def forward(self, input):
        return mish(input)

class EmoModel(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
      super().__init__()
      self.base_model = base_model
      self.classifier = nn.Sequential(
        nn.Dropout(dropout),
        # AUTO-ASSOCIATION LAYER
        nn.Linear(base_model_output_size, base_model_output_size),
        Mish(),
        nn.Dropout(dropout),
        # PROJECTION LAYER
        nn.Linear(base_model_output_size, n_classes)
      )
      for layer in self.classifier:
        if isinstance(layer, nn.Linear):
          layer.weight.data.normal_(mean=0.0, std=0.02)
          if layer.bias is not None:
            layer.bias.data.zero_()

    def forward(self, input_, *args):
      print("passed input -->> ", input_)
      X, attention_mask = input_
      hidden_states = self.base_model(X, attention_mask=attention_mask)
      # maybe do some pooling / RNNs... go crazy here!
      # use the <s> representation
      return self.classifier(hidden_states[0][:, 0, :])
import pytorch_lightning as pl
from model_definition_ import EmoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from typing import List
import torch.nn.functional as F
from torch.optim import AdamW as AdamW
from torch_lr_finder import LRFinder
from argparse import Namespace
from functools import lru_cache
from torch.utils.data import DataLoader, Dataset
from data_preparation_ import prepare_data_for_train_val_test_sets_, EmoDataset
import data_ingestion_
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import get_linear_schedule_with_warmup

class TokenizersCollateFn:
    def __init__(self, max_tokens=512):

      ## RoBERTa uses BPE tokenizer similar to GPT
      t = ByteLevelBPETokenizer(
          "tokenizer/vocab.json",
          "tokenizer/merges.txt"
      )
      t._tokenizer.post_processor = BertProcessing(
          ("</s>", t.token_to_id("</s>")),
          ("<s>", t.token_to_id("<s>")),
      )
      t.enable_truncation(max_tokens)
      t.enable_padding(length=max_tokens, pad_id=t.token_to_id("<pad>"))
      self.tokenizer = t

    def __call__(self, batch):
      encoded = self.tokenizer.encode_batch([x[0] for x in batch])
      sequences_padded = torch.tensor([enc.ids for enc in encoded])
      attention_masks_padded = torch.tensor([enc.attention_mask for enc in encoded])
      labels = torch.tensor([x[1] for x in batch])

      return (sequences_padded, attention_masks_padded), labels

class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = EmoModel(AutoModelForCausalLM.from_pretrained("distilroberta-base").base_model, 30) #len(desired_emotions_lab_))
        self.loss = nn.CrossEntropyLoss() ## combines LogSoftmax() and NLLLoss()
        #self.hparams = hparams
        self.hparams.update(vars(hparams))

    def step(self, batch, step_name="train"):
        X, y = batch
        loss = self.loss(self.forward(X), y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
               "progress_bar": {loss_key: loss}}

    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def validation_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def train_dataloader(self):
        print('CREATING TRAIN DATALOADER -- >> ')
        return self.create_data_loader(self.hparams.train_path, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.hparams.val_path)

    def test_dataloader(self):
        return self.create_data_loader(self.hparams.test_path)

    def create_data_loader(self, ds_path: str, shuffle=False):
        return DataLoader(
                    EmoDataset(ds_path),
                    batch_size=self.hparams.batch_size,
                    shuffle=shuffle,
                    collate_fn=TokenizersCollateFn()
        )

    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs

    def configure_optimizers(self):
        ## use AdamW optimizer -- faster approach to training NNs
        ## read: https://www.fast.ai/2018/07/02/adam-weight-decay/
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.hparams.warmup_steps,
                    num_training_steps=self.total_steps(),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
    

    
# from pytorch_lr_finder.torch_lr_finder import LRFinder

lr=0.1 ## uper bound LR
goemotions_location_ = "data_goemotions_/data/train_val_test_sets_/content/"
train_path = goemotions_location_+"train.txt"
val_path = goemotions_location_+"val.txt"
test_path = goemotions_location_+"test.txt"

hparams_tmp = Namespace(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    batch_size=16,
    warmup_steps=100,
    epochs=1,
    lr=lr,
    accumulate_grad_batches=1,
)
module = TrainingModule(hparams_tmp)
print('Loaded module -- >> ',type(module.train_dataloader()))
print(module.train_dataloader())
print('Iterating on dataloader -- >> ')
data_iter = iter(module.train_dataloader())
A, B = next(data_iter)
print('>>--->><<----<<',A)
print('>>--->><<----<<',B)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(module.parameters(), lr=5e-7) ## lower bound LR
print('Learning rate finder : Initialization -- >> ')
lr_finder = LRFinder(module, optimizer, criterion, device="cpu")
print('Learning rate finder : Range test -- >> ')
lr_finder.range_test(module.train_dataloader(), end_lr=100, num_iter=100) # , accumulation_steps=hparams_tmp.accumulate_grad_batches)
# # lr_finder.plot()
# # lr_finder.reset()
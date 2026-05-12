from finding_learning_rate_torchlrfinder_ import TrainingModule
from model_definition_ import EmoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pytorch_lightning as pl
from torch import nn
from argparse import Namespace
from torch.optim import AdamW as AdamW
import matplotlib.pyplot as plt

goemotions_location_ = "data_goemotions_/data/train_val_test_sets_/content/"
train_path = goemotions_location_+"train.txt"
val_path = goemotions_location_+"val.txt"
test_path = goemotions_location_+"test.txt"

lr_ = 3.6e-3

hparams = Namespace(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    batch_size=32,
    warmup_steps=100,
    epochs=1,
    lr=lr_,
    accumulate_grad_batches=1
)
module = TrainingModule(hparams)


## garbage collection
import gc; gc.collect()
torch.cuda.empty_cache()


## train roughly for about 10-15 minutes with GPU enabled.
# gpus=1, , progress_bar_refresh_rate=10
trainer = pl.Trainer(max_epochs=hparams.epochs,accumulate_grad_batches=hparams.accumulate_grad_batches)

trainer.fit(module)


'''
INFERENCE PART
with torch.no_grad():
    progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
    module.eval()
    true_y, pred_y = [], []
    for i, batch_ in enumerate(module.test_dataloader()):
        (X, attn), y = batch_
        batch = (X.cuda(), attn.cuda())
        print(progress[i % len(progress)], end="\r")
        y_pred = torch.argmax(module(batch), dim=1)
        true_y.extend(y.cpu())
        pred_y.extend(y_pred.cpu())
print("\n" + "_" * 80)
print(classification_report(true_y, pred_y, target_names=label2int.keys(), digits=len(emotions)))
'''
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
      print(f"🔡 Initializing BPE Tokenizer with max_tokens={max_tokens}...")
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
      print("✅ Tokenizer loaded and post-processor configured.")

    def __call__(self, batch):
      encoded = self.tokenizer.encode_batch([x[0] for x in batch])
      sequences_padded = torch.tensor([enc.ids for enc in encoded])
      attention_masks_padded = torch.tensor([enc.attention_mask for enc in encoded])
      labels = torch.tensor([x[1] for x in batch])
      
      # Useful for debugging first batch shapes
      # print(f"💎 Batch Encoded: IDs shape {sequences_padded.shape} | Labels shape {labels.shape}")

      return (sequences_padded, attention_masks_padded), labels


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        print(f"🏗️  Initializing TrainingModule with model: distilroberta-base")
        self.model = EmoModel(AutoModelForCausalLM.from_pretrained("distilroberta-base").base_model, 30)
        self.loss = nn.CrossEntropyLoss() 
        self.hparams.update(vars(hparams))
        print(f"✅ Hyperparameters loaded: {self.hparams}")

    def step(self, batch, step_name="train"):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss(logits, y)
        
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        # Debug print for every 50th batch to avoid terminal spam
        if self.global_step % 50 == 0:
            emoji = "🔥" if step_name == "train" else "🧪"
            print(f"{emoji} [{step_name.upper()}] Step: {self.global_step} | Loss: {loss.item():.4f}")

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
        print(f"📊 [EPOCH END] Average Validation Loss: {loss:.4f} 📉")
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def train_dataloader(self):
        print(f"📂 Loading Train Dataloader from: {self.hparams.train_path} 🟢")
        return self.create_data_loader(self.hparams.train_path, shuffle=True)

    def val_dataloader(self):
        print(f"📂 Loading Val Dataloader from: {self.hparams.val_path} 🟡")
        return self.create_data_loader(self.hparams.val_path)

    def test_dataloader(self):
        print(f"📂 Loading Test Dataloader from: {self.hparams.test_path} 🔵")
        return self.create_data_loader(self.hparams.test_path)

    def create_data_loader(self, ds_path: str, shuffle=False):
        try:
            loader = DataLoader(
                        EmoDataset(ds_path),
                        batch_size=self.hparams.batch_size,
                        shuffle=shuffle,
                        collate_fn=TokenizersCollateFn()
            )
            return loader
        except Exception as e:
            print(f"❌ Error loading data from {ds_path}: {e}")
            raise e

    @lru_cache()
    def total_steps(self):
        steps = len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs
        print(f"🔢 Total Training Steps calculated: {steps}")
        return steps

    def configure_optimizers(self):
        print(f"⚙️  Configuring Optimizers (AdamW) and LR Scheduler...")
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        
        t_steps = self.total_steps()
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.hparams.warmup_steps,
                    num_training_steps=t_steps,
        )
        print(f"🚀 Optimizer ready. Warmup steps: {self.hparams.warmup_steps} | Total: {t_steps}")
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

# --- Initialization ---
print("🏁 Starting Project Setup...")
lr=0.1 
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
print("✨ Module successfully instantiated and ready for Trainer!")

print('Loaded module -- >> ',type(module.train_dataloader()))
print(f"📦 Dataloader object: {module.train_dataloader()}")
print('🔍 Iterating on dataloader for sanity check -- >> ')

try:
    data_iter = iter(module.train_dataloader())
    A, B = next(data_iter)
    print('✅ Successfully fetched first batch!')
    print(f'Input Tensors (A) shapes: {[t.shape for t in A]}')
    print(f'Label Tensor (B) shape: {B.shape}')
except Exception as e:
    print(f"❌ Failed to iterate on dataloader: {e}")

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(module.parameters(), lr=5e-7) ## lower bound LR

print('📈 Learning rate finder: Initialization...')
lr_finder = LRFinder(module, optimizer, criterion, device="cpu")

print('🏃 Starting LR Range Test (this may take a moment)...')
lr_finder.range_test(module.train_dataloader(), end_lr=100, num_iter=100) 

print('📊 Range Test complete! Plotting results...')
lr_finder.plot()
from finding_learning_rate_torchlrfinder_ import TrainingModule
from model_definition_ import EmoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch import nn
from argparse import Namespace
from torch.optim import AdamW as AdamW
import matplotlib.pyplot as plt



from argparse import Namespace

def find_lr(module, init_value = 1e-8, final_value=10., beta = 0.98):
    trn_loader = module.train_dataloader()
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in trn_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


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

# net = EmoModel(AutoModelForSequenceClassification.from_pretrained("distilroberta-base").base_model, len(desired_emotions_lab_))
net = EmoModel(AutoModelForSequenceClassification.from_pretrained("distilroberta-base").base_model, len(desired_emotions_lab_))
optimizer = AdamW(module.parameters(), lr=5e-7)
criterion = nn.CrossEntropyLoss()


logs,losses = find_lr(module)
plt.plot(logs[10:-5],losses[10:-5])
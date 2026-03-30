

from model_definition_ import EmoModel
from data_preparation_ import prepare_data_for_train_val_test_sets_, EmoDataset
import data_ingestion_
import os
import numpy as np

if __name__ == "__main__":
    print('Preparing dataset prior to training')
    the_project_info_ = data_ingestion_.get_toml_info_()
    # -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # Getting data from google apis using wget
    # data_ingestion_.get_goemotions_data_from_googleapis_(the_project_info_)
    # -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    goemotions_dl_loc_ = the_project_info_["dataset_goemotions"]["goemotions_download_location_"]
    # -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    dataframe_emotions_, label_emotions_ = data_ingestion_.prepare_dataframe(the_project_info_)
    # print(dataframe_emotions_)
    # Plotting histogram of all classes 
    # data_ingestion_.plot_emotion_class_data_distribution(dataframe_emotions_, label_emotions_)
    # -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # MAKE THE TRAIN, TEST, AND VALIDATION DATA READY
    goemotions_ml_loc_ = the_project_info_["dataset_goemotions"]["goemotions_mldata_location_"]
    # prepare_data_for_train_val_test_sets_(dataframe_emotions_, goemotions_ml_loc_)
    # -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    # TEST THE DATALOADER CLASS 
    train_ds_ = EmoDataset(goemotions_ml_loc_+"train.txt")
    # batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # print(train_ds_)
    # text_ = (train_ds_[19][0])
    # -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
    from tokenizer_fn_ import tokenize_function
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_ = AutoModelForSequenceClassification.from_pretrained(model_name)
    any_random_entry_ = 174
    the_text_ = train_ds_[any_random_entry_][0]
    tokens_ = tokenizer(the_text_, padding="max_length", truncation=True, max_length=128)
    # # print(type(tokens_))
    # # print(tokens_['input_ids'])
    # # print(tokens_['attention_mask'])
    # # text = "Elvis is the king of rock!"
    # from transformers import pipeline
    # classifier = pipeline("sentiment-analysis", model=model_, tokenizer=tokenizer)
    # classifier = pipeline("sentiment-analysis", model=model_, tokenizer=tokenizer)
    # enc = tokenizer(the_text_)
    # print(10*'-->>','\n ',the_text_, classifier(the_text_))

    out = model_(torch.tensor(tokens_["input_ids"]).unsqueeze(0), torch.tensor(tokens_["attention_mask"]).unsqueeze(0))
    # out = model_(torch.tensor(res), torch.tensor(res))
    print(out)

    # tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # from transformers import AutoModelWithLMHead, AutoTokenizer
    # import torch
    # classifier = EmoModel(AutoModelWithLMHead.from_pretrained("distilroberta-base").base_model, 3)

    # X = torch.tensor(enc["input_ids"]).unsqueeze(0).to('cpu')
    # attn = torch.tensor(enc["attention_mask"]).unsqueeze(0).to('cpu') 


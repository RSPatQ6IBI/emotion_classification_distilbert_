from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import pipeline

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# Function to tokenize the text
def tokenize_function(the_text_):
    return tokenizer(the_text_, padding="max_length", truncation=True, max_length=128)


def pretrained_model_emoClassification(the_text_):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model_ = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model_, tokenizer=tokenizer)
    return classifier(the_text_)



# Tokenize dataset
# from transformers import AutoModelForSequenceClassification

# # Load pre-trained DistilBERT model with 27 output labels (GoEmotions has 27 emotions)
# num_labels = 27
# model_name = "distilbert-base-uncased"
# model = AutoModelForSequenceClassification.from_pretrained(
# model_name,
# num_labels=num_labels,
# )


# def convert_labels_to_tensor(examples):
#     labels = examples["labels"]


#     if isinstance(labels[0], float):

#         multi_hot = torch.zeros((len(labels), num_labels), dtype=torch.float32)
#         for i, sample_labels in enumerate(labels):
#             if 0 <= int(sample_labels) < num_labels:
#                 multi_hot[i, int(sample_labels)] = 1
#         examples["labels"] = multi_hot
#     else:

#         max_labels = max(len(sample_labels) for sample_labels in labels)
#         padded_labels = [sample_labels + [-1] * (max_labels - len(sample_labels)) for sample_labels in labels]

#         multi_hot = torch.zeros((len(labels), num_labels), dtype=torch.float32)
#         for i, sample_labels in enumerate(padded_labels):
#             for label in sample_labels:
#                 if 0 <= label < num_labels:
#                     multi_hot[i, label] = 1

#         examples["labels"] = multi_hot

#     return examples

# tokenized_datasets = tokenized_datasets.map(convert_labels_to_tensor, batched=True)
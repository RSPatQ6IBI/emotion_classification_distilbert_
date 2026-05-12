from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = model.base_model
## load pretrained tokenizer information
tokenizer.save_pretrained("tokenizer")
print("DONE")
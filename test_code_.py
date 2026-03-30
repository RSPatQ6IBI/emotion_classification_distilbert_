from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

#Testing the fine tuned model
model_path = "go_emotions_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)


emotions = emotion_classifier("What a bright sunny day!")


most_confident = max(emotions[0], key=lambda x: x['score'])

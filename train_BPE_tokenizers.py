from tokenizers import ByteLevelBPETokenizer

goemotions_location_ = "data_goemotions_/data/train_val_test_sets_/content/"
train_path = goemotions_location_+"train.txt"

# # Initialize and train
# tokenizer = ByteLevelBPETokenizer()
# tokenizer.train(files=[train_path], vocab_size=5000)

# Initialize and train
tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
# And then train
tokenizer.train(
    files=[train_path],
    vocab_size=5000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["<s>", "<pad>", "</s>"],
)


# Export vocab.json and merges.txt
tokenizer.save_model("tokenizer")


import os
print('The Tokenizer Dir : ')
print(os.listdir("tokenizer"))

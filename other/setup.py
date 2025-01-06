# Tensorflow Dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
# HuggingFace
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, TFBartForConditionalGeneration, pipeline
from datasets import load_dataset

# --------------------------------------
# DATASET DOWNLOAD (do this once)
# --------------------------------------

# Download reddit-tifu/long datasets (Tensorflow)
# reddit_long_ds = tfds.load('reddit_tifu/long', split='train[:10%]') 
# assert isinstance(reddit_ds, tf.data.Dataset)

# --------------------------------------
# BART - LARGE
# --------------------------------------

# Load in reddit_tifu/long dataset (Tensorflow)
reddit_ds = tfds.load('reddit_tifu/long', download=False, split='train[:10%]') 
assert isinstance(reddit_ds, tf.data.Dataset)

# Load in booksum dataset (Huggingface) (need to convert to TF?): 
# ds = load_dataset("ubaada/booksum-complete-cleaned", "books")
# ds = load_dataset("ubaada/booksum-complete-cleaned", "chapters")

# Load model directly (do this once)
        # checkpoint = "facebook/bart-large"
        # model = TFBartForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True)
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", local_files_only=True)

# Tokenize the dataset
def tokenize_reddit(data):
    # Ensure data is in string format
    data["documents"] = str(data["documents"])
    data["tldr"] = str(data["tldr"])

    # Tokenizing the input and reference summary texts ("documents" and "tldr")
    inputs = tokenizer(data["documents"], truncation=True, padding="max_length", max_length=1024, return_attention_mask=True, return_tensors="pt") # Pad to get constant length. Need attention to ensure pads are not focused on.
    targets = tokenizer(data["tldr"], truncation=True, padding="max_length", max_length=150, return_attention_mask=True, return_tensors="pt")
    
    # Return the tokenized inputs and reference summaries
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], 
            "summary_ids": targets["input_ids"]}

# Apply the tokenization to tensor (NOTE, FOR HUGGINGFACE DATA, you can use normal subscripting as it will not be a tensor)
tokenized_reddit = reddit_ds.map(tokenize_reddit) 

# Generate Summaries
def generate_summary(single_tokenized_data):
    # Convert tensors to concrete values
    input_ids = torch.tensor(single_tokenized_data["input_ids"].numpy())
    attention_mask = torch.tensor(single_tokenized_data["attention_mask"].numpy())
    
    # Generate summary
    summary_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)
    
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# for example in tokenized_reddit.take(3): 
#   print(generate_summary(example))

# Process single example
for example in reddit_ds.take(1):
    print("Original example:", example)
    tokenized = tokenize_reddit(example)
    print("Tokenized output:", tokenized)
    
    # Generate summary
    summary = generate_summary(tokenized)
    print("Generated summary:", summary)


# Error Encountered
# Tensor("args_0:0", shape=(), dtype=string) Why is it printing these? Also something about the cache.
# Moved over to Google Collab for easier use. To Do: Try downloading the dataset later. Actually its shit on collab ill stay here. also try running pipeline
# Maybe the problem is the model we are importing? CNN model? O













# Generate summaries for a batch
# for batch in train_loader:
#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     generated_summaries = [generate_summary(input_ids[i:i+1], attention_mask[i:i+1]) for i in range(len(input_ids))]
#     break  # Just running for one batch for testing
# Strategy: Beam Search?

# pipe = pipeline("feature-extraction", model="facebook/bart-large")
import os

# Tensorflow Dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
# HuggingFace
from transformers import AutoTokenizer,AutoModel, AutoModelForSeq2SeqLM, BartForConditionalGeneration, BartTokenizer, TFBartForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from datasets import load_dataset
# Testing
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# --------------------------------------
# DATASET DOWNLOAD (do this once)
# --------------------------------------

# Download reddit_tifu/long datasets (Tensorflow)
# reddit_long_ds = tfds.load('reddit_tifu/long', split='train[:10%]')
# assert isinstance(reddit_ds, tf.data.Dataset)

# --------------------------------------
# DATASET SETUP
# --------------------------------------

# # Load in reddit_tifu/long dataset (Tensorflow)
# reddit_ds = tfds.load('reddit_tifu/long', download=False, split='train[:10%]')
# assert isinstance(reddit_ds, tf.data.Dataset)

# # Load in booksum dataset (Huggingface) (need to convert to TF?):
# book_ds = load_dataset("ubaada/booksum-complete-cleaned", "books")
# chapter_ds = load_dataset("ubaada/booksum-complete-cleaned", "chapters")

# --------------------------------------
# MODELS
# --------------------------------------

# Create summarization pipeline
# checkpoint = "facebook/bart-large"
# checkpoint = "facebook/bart-large-cnn"
# checkpoint = "jordanfan/bart_extractive_1024_750"
# checkpoint = "google/pegasus-cnn_dailymail"
# checkpoint = "google/pegasus-large"
# tokenizer = BartTokenizer.from_pretrained(checkpoint)
# model = BartForConditionalGeneration.from_pretrained(checkpoint)

# summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device = 0)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

tokenizer = PegasusTokenizer.from_pretrained(checkpoint)
model = PegasusForConditionalGeneration.from_pretrained(checkpoint)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device = 0)

# --------------------------------------
# TESTING
# --------------------------------------

# Store results
results = []

# Process multiple examples for Reddit
# for example in reddit_ds.take(100):
#     # Get raw text and reference summary
#     text = example["documents"].numpy().decode()
#     reference = example["tldr"].numpy().decode()

#     # Generate summary
#     generated = summarizer(text,
#                           min_length=40,
#                           max_length=150,
#                           length_penalty=1.,
#                           num_beams=4,
#                           early_stopping=True)[0]['summary_text']

#     # Calculate ROUGE scores
#     rouge_scores = scorer.score(reference, generated)

#     # Calculate BERTScore
#     P, R, F1 = bert_score([generated], [reference], lang='en', device='cpu')
#     bert_f1 = F1.numpy()[0]  # Get F1 score

#     # Store results
#     results.append({
#         'reference': reference,
#         'generated': generated,
#         'rouge_scores': rouge_scores,
#         'bert_score': bert_f1
#     })

# # Process multiple examples for BookSum
# for example in chapter_ds['test'].select(range(100)):  # Select first 100 examples
#     # Get raw text and reference summary
#     text = example['text']  # Ensure correct field name
#     reference = example['summary'][0]['text']  # Ensure correct field name

#     # Ensure both text and reference are strings
#     if isinstance(text, list):
#         text = " ".join(text)
#     if isinstance(reference, list):
#         reference = " ".join(reference)

#     # Tokenize the input text with truncation
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=1024)
#     truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

#     # Generate summary
#     generated = summarizer(truncated_text,
#                           min_length=500,
#                           max_length=1000,
#                           length_penalty=1,
#                           num_beams=4,
#                           early_stopping=True)[0]['summary_text']

#     # Calculate ROUGE scores
#     rouge_scores = scorer.score(reference, generated)

#     # Calculate BERTScore
#     P, R, F1 = bert_score([generated], [reference], lang='en', device='cpu')
#     bert_f1 = F1.numpy()[0]  # Get F1 score


#     # Store results
#     results.append({
#         'reference': reference,
#         'generated': generated,
#         'rouge_scores': rouge_scores,
#         'bert_score': bert_f1,
#     })

# --------------------------------------
# RESULT OUTPUT
# --------------------------------------

rouge1_total = rouge2_total = rougeL_total = bert_total = 0

print(f"\n{checkpoint} - Results for 10 examples:")
for i, result in enumerate(results, 1):
    print(f"\nExample {i}:")
    print(f"Reference: {result['reference']}")
    print(f"Generated: {result['generated']}")
    print(f"ROUGE scores: {result['rouge_scores']}")
    print(f"BERTScore: {result['bert_score']:.3f}")

    rouge1_total += result['rouge_scores']['rouge1'].fmeasure
    rouge2_total += result['rouge_scores']['rouge2'].fmeasure
    rougeL_total += result['rouge_scores']['rougeL'].fmeasure
    bert_total += result['bert_score']

print("\nAverage scores:")
print(f"ROUGE-1: {rouge1_total/10:.3f}")
print(f"ROUGE-2: {rouge2_total/10:.3f}")
print(f"ROUGE-L: {rougeL_total/10:.3f}")
print(f"BERTScore: {bert_total/10:.3f}")
print("--------------------")
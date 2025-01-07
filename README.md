# Exploration in Summarization

Read our paper [here](./public/BC3997_Final_Report__Lam_Labasan.pdf)!

# Table of Contents
1. [Abstract](#abstract)
2. [Models & Datasets](#models-&-datasets)
    - [Models](#models)
    - [SQL Mappings](#sql-mappings)
    - [Explanation of Entities and Relations](#explanation-of-entities-and-relations)
    - [Potential Interaction Flow](#potential-interaction-flow)
3. [Getting Started](#getting-started)
4. [Reflection](#reflection)
5. [Notes](#notes)

# Abstract
This project evaluates the performance of BART and Pegasus large language models (LLMs) for summarizing text from Reddit microblogs and books, two domains with distinct challenges in text length, structure, and summarization needs. We compare five models—BART-large, BART-large-CNN, BART-Extractive, Pegasus-large, and Pegasus- CNN-DailyMail—on the Reddit_TIFU/long corpus and the BookSum corpus, using ROUGE and BERTScore metrics. Results show that BART models seem to fair better in both abstract and extractive summarization for both the Reddit and BookSum datasets when compared to Pegasus models. Our qualitative analysis of the generated summaries also shows that BART faired better in developing abstract summaries for longer texts compared to Pega- sus, which was a surprise.

# Models & Datasets
## Models
1. [facebook/bart-large](https://huggingface.co/facebook/bart-large)
2. [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
3. [jordanfan/bart_extractive_1024_750](https://huggingface.co/jordanfan/bart_extractive_1024_750)
4. [google/pegasus-large](https://huggingface.co/google/pegasus-large)
5. [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)

## Datasets
1. [reddit_tifu/long](https://www.tensorflow.org/datasets/catalog/reddit_tifu)
2. [ubaada/booksum-complete-cleaned](https://huggingface.co/datasets/ubaada/booksum-complete-cleaned)

# Replicating Our Experiment
## Installation: 
1. Clone this repository to your system. In the terminal, type in `git clone https://github.com/MatthewLabasan/nlp-project.git`
2. From the terminal, run `pip install -r requirements.txt`
3. Alternatively, you can run this in Google Colab by downloading the file in `public/BC3997_Final_Project_Code.py` and import into Google Colab.

## Usage:
1. Uncomment the `DATASET DOWNLOAD` section and download the Reddit dataset to your system.
2. Our results were developed by running each model on one dataset at a time. Uncomment the dataset in `DATASET SETUP` and the model in `MODELS` that you want to use.
3. Under `TESTING`, uncomment the code matching the dataset you will be using. This includes the code below `Process multiple examples for Reddit` or `Process multiple examples for BookSum`. You can optionally change the generator settings in this section. _Note_: The BookSum dataset requires you to truncate the input due to the models being limited to a 1024 token input.
4. Click run. Your results for that model and dataset will appear in the terminal.

# Reflection
Through this project, I gained experience in using and testing pretrained models and developing an NLP experiement, as well as deepened my understanding in how different model architectures and fine-tuning impact summarization results. It took a lot of trial and error for me to setup and run the models, which helped me to solidify what I learned in class about tokenization and summarization. In the future, I hope to delve deeper into the training aspect of each model, and learn how to fine tune a model for my own purposes.
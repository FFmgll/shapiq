import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from datasets import load_dataset


def tokenize_input(input_text):
    return np.asarray(tokenizer(input_text)['input_ids'][1:-1])


if __name__ == "__main__":
    dataset = load_dataset('imdb')

    data = dataset['train']
    model = pipeline(model="lvwerra/distilbert-imdb", task="sentiment-analysis")
    tokenizer = model.tokenizer

    short_reviews = []

    for sample in tqdm(data):
        review = sample["text"]
        sentiment = sample["label"]
        token_input = tokenize_input(review)
        n_tokens = len(token_input)
        if n_tokens < 14:
            continue
        token_input = token_input[0:15]
        short_reviews.append(tokenizer.decode(token_input))

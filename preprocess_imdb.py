import numpy as np

import pandas as pd

from transformers import pipeline
from tqdm import tqdm
from datasets import load_dataset


def tokenize_input(input_text):
    return np.asarray(tokenizer(input_text)['input_ids'][1:-1])


if __name__ == "__main__":
    dataset = load_dataset('imdb')
    model = pipeline(model="lvwerra/distilbert-imdb", task="sentiment-analysis")
    tokenizer = model.tokenizer

    sentences = []
    sentences_length = {10, 11, 12, 13, 14}

    sentence_id = 0

    data = dataset['train']
    for i in tqdm(range(0, len(data))):
        sample = data[i]
        review = str(sample["text"])
        sentiment = sample["label"]
        first_sentence = review.replace("!", ".").replace(":", ".").replace(";", ".").replace(",", "").replace("\"", "").replace("<br />", "").split(".")[0]
        token_input = tokenize_input(first_sentence)
        n_tokens = len(token_input)
        if n_tokens in sentences_length:
            sentences.append({"text": first_sentence, "length": n_tokens, "id": sentence_id})
            sentence_id += 1

    data = dataset['test']
    for i in tqdm(range(0, len(data))):
        sample = data[i]
        review = str(sample["text"])
        sentiment = sample["label"]
        first_sentence = review.replace("!", ".").replace(":", ".").replace(";", ".").replace(",", "").replace("\"", "").replace("<br />", "").split(".")[0]
        token_input = tokenize_input(first_sentence)
        n_tokens = len(token_input)
        if n_tokens in sentences_length:
            sentences.append({"text": first_sentence, "length": n_tokens, "id": sentence_id})
            sentence_id += 1

    df = pd.DataFrame(sentences)
    df.to_csv("data/simplified_imdb.csv", index=False)

import gensim
from gensim.models import Word2Vec
import pandas as pd
import re

patterns = "[!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
CLOUD_INFRA_DATASET = ("./resources/clean_result/merged/cloud_infra.txt", "CLOUD_INFRA_MODEL")
MEDICINE_DATASET = ("./resources/clean_result/merged/medicine.txt", "MEDICINE_MODEL")

def train_model(dataset:  tuple[str, str]) -> Word2Vec:
    response = []
    with open(dataset[0], encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            temp = line.split('\t')
            response.append(re.sub(patterns, ' ', temp[0]))

    data = pd.DataFrame(list(zip(response)))
    data.columns = ['response']
    response_base = data.response.apply(gensim.utils.simple_preprocess)

    model = Word2Vec(
        sentences=response_base,
        min_count=10,
        window=2,
        vector_size=16,
        alpha=0.03,
        negative=15,
        min_alpha=0.0007,
        sample=6e-5
    )

    model.build_vocab(response_base, update=True)
    model.train(response_base, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(f"resources/{dataset[1]}.model")

    return model

def print_model_stats_by_word(
        model_name: str,
        model: Word2Vec,
        word: str,
        most_similar: list[str]
):
    print(f"""
    Model: {model_name}
    model.corpus_count: {model.corpus_count},
    model.wv.has_index_for {word}: {model.wv.has_index_for(word)},
    model.wv.similar_by_vector {word}: {model.wv.similar_by_vector(model.wv[word])}
    model.wv.most_similar_to_given {word} {most_similar}: {model.wv.most_similar_to_given(word, most_similar)}
    """)

if __name__ == "__main__":
    cloud_infra_model = train_model(CLOUD_INFRA_DATASET)
    print_model_stats_by_word('cloud_infra_model', cloud_infra_model, 'контейнер', ['приложение', 'сервер'])

    medicine_model = train_model(MEDICINE_DATASET)
    print_model_stats_by_word('medicine_model', medicine_model, 'pneumonia', ['disease', 'illness'])
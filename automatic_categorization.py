import re

import numpy as np
from gensim.models import Word2Vec

MODELS = [
    "resources/CLOUD_INFRA_MODEL.model",
    "resources/MEDICINE_MODEL.model"
]

SAMPLE_TEXT_DIRECTORY = "resources/texts"

TEXTS = [
    "resources/clean_result/arvi_fcimb-10-00225.txt",
    "resources/clean_result/arvi_fcimb-12-817532.txt",
    "resources/clean_result/arvi_fcimb-13-1082925.txt",
    "resources/clean_result/arvi_fimmu-08-00948.txt",
    "resources/clean_result/arvi_fimmu-09-02640.txt",
    "resources/clean_result/arvi_fimmu-09-02943.txt",
    "resources/clean_result/arvi_fmed-01-00041.txt",
    "resources/clean_result/arvi_fmed-07-00420.txt",
    "resources/clean_result/arvi_fmicb-09-02762.txt",
    "resources/clean_result/arvi_fmicb-14-1279159.txt",
    "resources/clean_result/arvi_fvets-06-00354.txt",
    "resources/clean_result/co_avtomatizirovannyy-instrumentariy-razvertyvaniya-oblachnyh-servisov.txt",
    "resources/clean_result/co_ispolzovanie-metodov-mashinnogo-obucheniya-dlya-predskazaniya-urovnya-zagruzki-resursov-sistemy.txt",
    "resources/clean_result/co_ispolzovanie-tehnologii-konteynerizatsii-kak-komponenta-obespecheniya-informatsionnoy-bezopasnosti.txt",
    "resources/clean_result/co_issledovanie-metodov-postroeniya-oblachnyh-platformennyh-servisov-i-realizatsiy-standarta-tosca.txt",
    "resources/clean_result/co_keshirovanie-dannyh-v-multikonteynernyh-sistemah.txt",
    "resources/clean_result/co_obzor-tehnologiy-organizatsii-tumannyh-vychisleniy.txt",
    "resources/clean_result/co_optimizatsiya-vysokonagruzhennyh-veb-proektov-s-ispolzovaniem-mikroservisnoy-arhitektury.txt",
    "resources/clean_result/co_postroenie-trebovaniy-i-arhitektury-oblachnogo-orkestratora-platformennyh-servisov.txt",
    "resources/clean_result/co_tehnologii-izolyatsii-prilozheniy-i-instrumentalnye-sredstva-dlya-upravleniya-konteynerami.txt",
    "resources/clean_result/co_upravlenie-konteynerami-pri-postroenii-raspredelennyh-sistem-s-mikroservisnoy-arhitekturoy.txt",
    "resources/clean_result/ocd_fnhum-17-1280512.txt",
    "resources/clean_result/ocd_fphar-10-01362.txt",
    "resources/clean_result/ocd_fpsyg-09-00620.txt",
    "resources/clean_result/ocd_fpsyt-10-00097.txt",
    "resources/clean_result/ocd_fpsyt-10-00523.txt",
    "resources/clean_result/ocd_fpsyt-12-659401.txt",
    "resources/clean_result/ocd_fpsyt-12-661807.txt",
    "resources/clean_result/ocd_fpsyt-13-822976.txt",
    "resources/clean_result/ocd_fpsyt-13-833394.txt",
    "resources/clean_result/ocd_fpsyt-15-1350978.txt",
    "resources/clean_result/ocd_nihms602028.txt",
    "resources/clean_result/sla_devops-v-epohu-oblachnyh-tehnologiy-sovremennye-praktiki-i-perspektivy-razvitiya.txt",
    "resources/clean_result/sla_dinamicheskie-tumannye-vychisleniya-i-besservernaya-arhitektura-na-puti-k-zelenym-ikt.txt",
    "resources/clean_result/sla_effektivnoe-ispolzovanie-oblachnyh-tehnologiy-v-smeshannom-obuchenii.txt",
    "resources/clean_result/sla_oblachnye-vychisleniya-sovremennye-tendentsii-problemy-i-perspektivy.txt",
    "resources/clean_result/sla_osobennosti-tehnologiy-besservernyh-vychisleniy.txt",
    "resources/clean_result/sla_razrabotka-besservernyh-mobilnyh-prilozheniy.txt",
    "resources/clean_result/sla_sovremennaya-oblachnaya-infrastruktura-besservernye-vychisleniya.txt",
    "resources/clean_result/sla_strategii-optimizatsii-dlya-vysokonagruzhennyh-prilozheniy-povyshenie-obschey-proizvoditelnosti.txt",
    "resources/clean_result/sla_strategiya-razvertyvaniya-mikroservisov-v-oblake.txt",
    "resources/clean_result/sla_vysvobozhdenie-proizvoditelnosti-glubokoe-pogruzhenie-v-prilozheniya-s-vysokoy-nagruzkoy.txt",
    ]

PATTERN = r'[^\w\s]'


def get_words_from_file(file_path):
    word_set = set()
    pattern = r'[^\w\s]'

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            cleaned_line = re.sub(pattern, '', line)
            words = cleaned_line.lower().split()
            word_set.update(words)

    return set(filter(lambda word: len(word) > 3, word_set))

def count_word_vector(model, file_name) -> float:
    result = 0
    words = 0

    text_words = get_words_from_file(file_name)
    for word in text_words:
        if model.wv.has_index_for(word):
            result += model.wv.get_vector(word).sum()
            words += 1

    return result / words

def split_dict_into_chunks(input_dict, chunks = 4):
    items = list(input_dict.items())
    submaps = []
    n = len(items)
    chunk_size = n // chunks
    remainder = n % chunks

    start = 0
    for i in range(chunks):
        size = chunk_size + (1 if i < remainder else 0)
        submaps.append(dict(items[start:start + size]))
        start += size

    return submaps

def calculate_arithmetic_mean(input_dict):
    return sum(input_dict.values()) / len(input_dict)

def process_files_with_model(model_file):
    print(model_file)
    model = Word2Vec.load(model_file)
    files_avg_vectors = { file_name:count_word_vector(model, file_name) for file_name in TEXTS }
    sorted_files_avg_vectors = dict(sorted(files_avg_vectors.items(), key=lambda item: item[1]))
    chunked_files_avg_vectors = split_dict_into_chunks(sorted_files_avg_vectors)
    i = 0
    for chunk in chunked_files_avg_vectors:

        i += 1
        avg = calculate_arithmetic_mean(chunk)
        sim_words = model.wv.similar_by_vector(vector=np.array(avg), topn=5)

        print(f"""
        Group: {i}
        Average_vector: {avg}
        Keywords: {sim_words}
        """)


for model_name in MODELS:
    process_files_with_model(model_name)

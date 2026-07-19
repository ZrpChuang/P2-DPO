import nltk
from nltk.stem import WordNetLemmatizer
import json
import spacy
from tqdm import tqdm
import warnings
import argparse
import os
from collections import defaultdict

nlp = spacy.load("en_core_web_lg")
warnings.filterwarnings("ignore", category=UserWarning)


def get_args():
    parser = argparse.ArgumentParser(description="Run evaluation on model inference data.")
    parser.add_argument("--word_association", type=str, default="data/AMBER/relation.json", help="Path to the word association JSON file.")
    parser.add_argument("--safe_words", type=str, default="data/AMBER/safe_words.txt", help="Path to the safe words text file.")
    parser.add_argument("--inference_data", type=str, required=True, help="Path to the model's inference output JSON file.")
    parser.add_argument("--annotation", type=str, default="data/AMBER/annotations.json", help="Path to the ground truth annotations JSON file.")
    
    parser.add_argument("--output_file", type=str, default="outputs/AMBER/llava_AMBER_result.txt", help="File to save the evaluation results.")

    parser.add_argument("--similarity_score", type=float, default=0.8, help="Threshold for word similarity.")
    parser.add_argument('--evaluation_type', choices=['a', 'g', 'd', 'de', 'da', 'dr'], default='a', help='a: all tasks and dimensions, g: generative, d: discriminative, de: existence, da: attribute, dr: relation')
    
    args = parser.parse_args()
    return args


def check_synonyms_word(doc1, doc2, similarity_score):
    if doc1 and doc2 and doc1.vector_norm and doc2.vector_norm:
        similarity = doc1.similarity(doc2)
        return similarity > similarity_score
    return False


def extract_nouns(text, lemmatizer):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word.lower()) for word, pos in tagged if pos.startswith('NN')]
    return nouns


def process_generative_item(inference_item, gt_item, association, hallucination_words, global_safe_words, word_vectors, lemmatizer, args):
    metrics_update = defaultdict(int)
    
    response_nouns = extract_nouns(inference_item['response'], lemmatizer)
    
    after_process_nouns = [noun for noun in response_nouns if noun in hallucination_words]

    truth_words = gt_item['truth']
    hallu_gt_words = gt_item['hallu']

    safe_words = []
    safe_list_map = {}
    for i, word in enumerate(truth_words):
        safe_words.append(word)
        safe_list_map[word] = 0
        assoc_words = association.get(word, [])
        safe_words.extend(assoc_words)
        for aw in assoc_words:
            safe_list_map[aw] = 0

    ha_words = []
    ha_list_map = {}
    for i, word in enumerate(hallu_gt_words):
        ha_words.append(word)
        ha_list_map[word] = 0
        assoc_words = association.get(word, [])
        ha_words.extend(assoc_words)
        for aw in assoc_words:
            ha_list_map[aw] = 0

    safe_flag_list = [0] * len(after_process_nouns)

    for idx, noun in enumerate(after_process_nouns):
        if noun in global_safe_words:
            continue

        is_safe = False
        
        if noun in safe_words:
            safe_list_map[noun] = 1
            is_safe = True
        
        if noun in ha_words:
            ha_list_map[noun] = 1

        noun_doc = word_vectors.get(noun)
        if not noun_doc: continue
        
        for check_word in ha_words:
            if check_synonyms_word(noun_doc, word_vectors.get(check_word), args.similarity_score):
                ha_list_map[check_word] = 1
                break
        
        if not is_safe:
            for check_word in safe_words:
                if check_synonyms_word(noun_doc, word_vectors.get(check_word), args.similarity_score):
                    safe_list_map[check_word] = 1
                    is_safe = True
                    break

        if not is_safe:
            safe_flag_list[idx] = 1

    metrics_update['chair_score'] = sum(safe_flag_list)
    metrics_update['chair_num'] = len(safe_flag_list)
    metrics_update['safe_cover_score'] = sum(1 for w in truth_words if safe_list_map.get(w) == 1)
    metrics_update['safe_cover_num'] = len(truth_words)
    metrics_update['hallu_cover_score'] = sum(1 for w in hallu_gt_words if ha_list_map.get(w) == 1)
    metrics_update['hallu_cover_num'] = len(hallu_gt_words)
    if sum(safe_flag_list) == 0:
        metrics_update['non_hallu_score'] = 1
    metrics_update['non_hallu_num'] = 1

    return metrics_update


def process_discriminative_item(inference_item, gt_item):
    metrics_update = defaultdict(int)
    truth = gt_item['truth']
    
    response_raw = inference_item['response'].strip().lower()
    
    model_answer = None
    if response_raw.startswith('yes'):
        model_answer = 'yes'
    elif response_raw.startswith('no'):
        model_answer = 'no'

    gt_type = gt_item['type']

    type_prefix_map = {
        'discriminative-attribute-state': 'as_',
        'discriminative-attribute-number': 'an_',
        'discriminative-attribute-action': 'aa_',
        'discriminative-hallucination': 'ha_',
        'discriminative-relation': 'asso_'
    }
    prefix = type_prefix_map.get(gt_type, '')

    metrics_update['qa_correct_num'] = 1
    if prefix:
        metrics_update[f'{prefix}qa_correct_num'] = 1
    
    is_correct = (truth == model_answer)

    if is_correct:
        metrics_update['qa_correct_score'] = 1
        if prefix:
            metrics_update[f'{prefix}qa_correct_score'] = 1
            
    if truth == 'no':
        metrics_update['qa_no_num'] = 1
        if prefix:
            metrics_update[f'{prefix}qa_no_num'] = 1
        if model_answer == 'no':
            metrics_update['qa_no_score'] = 1
            if prefix:
                metrics_update[f'{prefix}qa_no_score'] = 1

    if model_answer == 'no':
        metrics_update['qa_ans_no_num'] = 1
        if prefix:
            metrics_update[f'{prefix}qa_ans_no_num'] = 1
        if truth == 'no':
            metrics_update['qa_ans_no_score'] = 1
            if prefix:
                metrics_update[f'{prefix}qa_ans_no_score'] = 1
    
    return metrics_update


def save_results_to_file(metrics, dimension, output_file):
    results_lines = []

    def safe_division(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0

    if dimension['g']:
        CHAIR = round(safe_division(metrics['chair_score'], metrics['chair_num']) * 100, 1)
        Cover = round(safe_division(metrics['safe_cover_score'], metrics['safe_cover_num']) * 100, 1)
        Ha = round(safe_division(metrics['hallu_cover_score'], metrics['hallu_cover_num']) * 100, 1)
        Ha_p = round(100 - safe_division(metrics['non_hallu_score'], metrics['non_hallu_num']) * 100, 1)
        results_lines.append("Generative Task:")
        results_lines.append(f"CHAIR:\t\t {CHAIR}")
        results_lines.append(f"Cover:\t\t {Cover}")
        results_lines.append(f"Hal:\t\t {Ha_p}")
        results_lines.append(f"Cog:\t\t {Ha}\n")

    if dimension['de'] and dimension['da'] and dimension['dr']:
        Accuracy = round(safe_division(metrics['qa_correct_score'], metrics['qa_correct_num']) * 100, 1)
        Precision = round(safe_division(metrics['qa_ans_no_score'], metrics['qa_ans_no_num']) * 100, 1)
        Recall = round(safe_division(metrics['qa_no_score'], metrics['qa_no_num']) * 100, 1)
        F1 = round(2 * Precision * Recall / (Precision + Recall + 1e-6), 1)
        results_lines.append("Discriminative Task (Overall):")
        results_lines.append(f"Accuracy:\t {Accuracy}")
        results_lines.append(f"Precision:\t {Precision}")
        results_lines.append(f"Recall:\t\t {Recall}")
        results_lines.append(f"F1:\t\t {F1}\n")

    if dimension['de']:
        Accuracy = round(safe_division(metrics['ha_qa_correct_score'], metrics['ha_qa_correct_num']) * 100, 1)
        Precision = round(safe_division(metrics['ha_qa_ans_no_score'], metrics['ha_qa_ans_no_num']) * 100, 1)
        Recall = round(safe_division(metrics['ha_qa_no_score'], metrics['ha_qa_no_num']) * 100, 1)
        F1 = round(2 * Precision * Recall / (Precision + Recall + 1e-6), 1)
        results_lines.append("Existence:")
        results_lines.append(f"Accuracy:\t {Accuracy}")
        results_lines.append(f"Precision:\t {Precision}")
        results_lines.append(f"Recall:\t\t {Recall}")
        results_lines.append(f"F1:\t\t {F1}\n")

    if dimension['da']:
        attr_correct_score = metrics['as_qa_correct_score'] + metrics['an_qa_correct_score'] + metrics['aa_qa_correct_score']
        attr_correct_num = metrics['as_qa_correct_num'] + metrics['an_qa_correct_num'] + metrics['aa_qa_correct_num']
        attr_ans_no_score = metrics['as_qa_ans_no_score'] + metrics['an_qa_ans_no_score'] + metrics['aa_qa_ans_no_score']
        attr_ans_no_num = metrics['as_qa_ans_no_num'] + metrics['an_qa_ans_no_num'] + metrics['aa_qa_ans_no_num']
        attr_no_score = metrics['as_qa_no_score'] + metrics['an_qa_no_score'] + metrics['aa_qa_no_score']
        attr_no_num = metrics['as_qa_no_num'] + metrics['an_qa_no_num'] + metrics['aa_qa_no_num']
        
        attr_Accuracy = round(safe_division(attr_correct_score, attr_correct_num) * 100, 1)
        attr_Precision = round(safe_division(attr_ans_no_score, attr_ans_no_num) * 100, 1)
        attr_Recall = round(safe_division(attr_no_score, attr_no_num) * 100, 1)
        attr_F1 = round(2 * attr_Precision * attr_Recall / (attr_Precision + attr_Recall + 1e-6), 1)
        results_lines.append("Attribute (Overall):")
        results_lines.append(f"Accuracy:\t {attr_Accuracy}")
        results_lines.append(f"Precision:\t {attr_Precision}")
        results_lines.append(f"Recall:\t\t {attr_Recall}")
        results_lines.append(f"F1:\t\t {attr_F1}\n")
    
    if dimension['dr']:
        Accuracy = round(safe_division(metrics['asso_qa_correct_score'], metrics['asso_qa_correct_num']) * 100, 1)
        Precision = round(safe_division(metrics['asso_qa_ans_no_score'], metrics['asso_qa_ans_no_num']) * 100, 1)
        Recall = round(safe_division(metrics['asso_qa_no_score'], metrics['asso_qa_no_num']) * 100, 1)
        F1 = round(2 * Precision * Recall / (Precision + Recall + 1e-6), 1)
        results_lines.append("Relation:")
        results_lines.append(f"Accuracy:\t {Accuracy}")
        results_lines.append(f"Precision:\t {Precision}")
        results_lines.append(f"Recall:\t\t {Recall}")
        results_lines.append(f"F1:\t\t {F1}\n")

    output_content = "\n".join(results_lines)


def main():
    args = get_args()
    
    metrics = defaultdict(int)

    association = json.load(open(args.word_association, 'r', encoding='utf-8'))
    with open(args.safe_words, 'r', encoding='utf-8') as f:
        global_safe_words = {line.strip() for line in f}
    inference_data = json.load(open(args.inference_data, 'r', encoding='utf-8'))
    ground_truth = json.load(open(args.annotation, 'r', encoding='utf-8'))

    hallucination_words = set()
    for word1, associated_words in association.items():
        hallucination_words.add(word1)
        for word2 in associated_words:
            hallucination_words.add(word2)
    
    all_words_to_process = list(hallucination_words.union(global_safe_words))
    
    docs = nlp.pipe(all_words_to_process)
    word_vectors = {word: doc for word, doc in zip(all_words_to_process, docs)}

    dimension = {'g': False,'de': False, 'da': False, 'dr': False}
    if args.evaluation_type == 'a':
        dimension = {k: True for k in dimension}
    elif args.evaluation_type == 'g':
        dimension['g'] = True
    elif args.evaluation_type == 'd':
        dimension['de'] = dimension['da'] = dimension['dr'] = True
    else:
        dimension[args.evaluation_type] = True
    
    lemmatizer = WordNetLemmatizer()

    for i in tqdm(range(len(inference_data)), desc="Evaluating"):
        inference_item = inference_data[i]
        gt_item = ground_truth[inference_item['id'] - 1]
        
        updates = None
        if gt_item['type'] == 'generative' and dimension['g']:
            updates = process_generative_item(inference_item, gt_item, association, hallucination_words, global_safe_words, word_vectors, lemmatizer, args)
        elif gt_item['type'].startswith('discriminative') and (dimension['de'] or dimension['da'] or dimension['dr']):
            updates = process_discriminative_item(inference_item, gt_item)
        
        if updates:
            for key, value in updates.items():
                metrics[key] += value

    save_results_to_file(metrics, dimension, args.output_file)


if __name__ == "__main__":
    main()

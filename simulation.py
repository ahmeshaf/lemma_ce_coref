import pickle
from helper import remove_puncts, jc, generate_mention_pairs, get_topic2mention_ids
import numpy as np
from tqdm import tqdm
# from collections import defaultdict
import math
from random import random


def get_mention_pair_similarity_lemma_value(mention_pairs, mention_map):
    similarities = []

    # generate similarity using the mention text
    for pair in tqdm(mention_pairs, desc='Generating Similarities'):
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = remove_puncts(men_map1['mention_text'].lower())
        men_text2 = remove_puncts(men_map2['mention_text'].lower())
        lemma1 = remove_puncts(men_map1['lemma'].lower())
        lemma2 = remove_puncts(men_map2['lemma'].lower())

        # doc_id1 = men_map1['doc_id']
        # sent_id1 = int(men_map1['sentence_id'])
        # all_sent_ids1 = {str(sent_id1 - 1), str(sent_id1), str(sent_id1 + 1)}
        # all_sent_ids1 = {str(sent_id1)}
        #
        # doc_id2 = men_map2['doc_id']
        # sent_id2 = int(men_map2['sentence_id'])
        # all_sent_ids2 = {str(sent_id2 - 1), str(sent_id2), str(sent_id2 + 1)}
        #
        # all_sent_ids2 = {str(sent_id2)}

        # sentence_tokens1 = [tok for sent_id in all_sent_ids1 if sent_id in doc_sent_map[doc_id1]
        #                     for tok in doc_sent_map[doc_id1][sent_id]['sentence_tokens']]
        #
        # sentence_tokens2 = [tok for sent_id in all_sent_ids2 if sent_id in doc_sent_map[doc_id2]
        #                     for tok in doc_sent_map[doc_id2][sent_id]['sentence_tokens']]

        sentence_tokens1 = [tok.lower() for tok in men_map1['sentence_tokens']]

        sentence_tokens2 = [tok.lower() for tok in men_map2['sentence_tokens']]

        sent_sim = jc(set(sentence_tokens1), set(sentence_tokens2))
        # sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        # doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
        lemma_sim = float(lemma1 in men_text2 or lemma2 in men_text1
                          or men_text1 in lemma2)
        similarities.append(0.8*lemma_sim + 0.2*sent_sim)

    return np.array(similarities)


def run_incremental_simulation(evt_mention_map_split, pair_similarity_map, top_n, threshold=0):
    current_events = list(evt_mention_map_split.keys())

    # simulate by topic
    topic2mention_ids = get_topic2mention_ids(evt_mention_map_split, current_events)

    # simulation metrics
    comparisons = 0
    positive_comparisons = 0
    # negative_comparisons = 0
    total_positive_comparisons = 0

    for topic_id, topic_mention_ids in topic2mention_ids.items():
        topic_clusters = []
        for event_id in topic_mention_ids:
            cluster_similarities = []
            for clus in topic_clusters:
                clus_sim = max([pair_similarity_map[tuple(sorted([event_id, m_id]))] for m_id in clus])
                cluster_similarities.append(clus_sim)

            sorted_candidates = sorted(list(zip(topic_clusters, cluster_similarities)), key=lambda x: x[-1], reverse=True)
            prob_top_n = math.floor(top_n) + int(random() < (top_n - math.floor(top_n)))

            # pruning by top-n
            pruned_candidates = sorted_candidates[:prob_top_n]
            # pruning by threshold
            pruned_candidates = [c for c, sim in pruned_candidates if sim >= threshold]

            # using ground-truth
            event_true_cluster = evt_mention_map_split[event_id]['gold_cluster']
            all_candidates_true_clusters = [evt_mention_map_split[c[0]]['gold_cluster'] for c, sim in sorted_candidates]
            pruned_candidates_true_clusters = [evt_mention_map_split[c[0]]['gold_cluster'] for c in pruned_candidates]

            if event_true_cluster in pruned_candidates_true_clusters:
                positive_comparisons += 1
                total_positive_comparisons += 1
                predicted_clus_index = pruned_candidates_true_clusters.index(event_true_cluster)
                comparisons += predicted_clus_index + 1
                predicted_clus = pruned_candidates[predicted_clus_index]
                predicted_clus.append(event_id)
            else:
                comparisons += len(pruned_candidates)
                if event_true_cluster in all_candidates_true_clusters:
                    total_positive_comparisons += 1
                topic_clusters.append([event_id])

    return comparisons, positive_comparisons, total_positive_comparisons


def get_pair_sim_map(pairs, similarities):
    sim_map = {}
    for p, sim in zip(pairs, similarities):
        p_tuple = tuple(sorted(p))
        sim_map[p_tuple] = sim
    return sim_map


def run_annotation_simulation_lemma(dataset, split, ns=None):
    if ns is None:
        # ns = [i/2 for i in range(2, 41)]
        ns = [i/2 for i in range(2, 21)]
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    all_mention_pairs = generate_mention_pairs(evt_mention_map, split)
    print("Total mention pairs in the Test set:", len(all_mention_pairs))

    lemma_similarities = get_mention_pair_similarity_lemma_value(all_mention_pairs, evt_mention_map)

    pair_similarity_map = get_pair_sim_map(all_mention_pairs, lemma_similarities)
    all_results = []
    for n in ns:
        comparisons, positive_comparisons, total_positive_comparisons = run_incremental_simulation(evt_mention_map, pair_similarity_map, n)
        recall = positive_comparisons/total_positive_comparisons
        precision = positive_comparisons/comparisons
        all_results.append((n, comparisons, recall, precision))

    print(all_results)


run_annotation_simulation_lemma('gvc', 'test')
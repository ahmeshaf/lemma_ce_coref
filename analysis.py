import pickle
import pandas as pd
from helper import remove_puncts
from collections import Counter


def hard_fns(dataset, heu='lh_oracle', split='dev'):
    dataset_folder = f'./datasets/{dataset}/'

    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))

    mps, mps_trans = pickle.load(open(dataset_folder + f'/{heu}/mp_mp_t_{split}.pkl', 'rb'))

    tps, fps, tns, fns = mps

    tps_trans, fps_trans, tns_trans, fns_trans = mps_trans

    tns_sentence_pairs = []

    for m1, m2 in fns_trans:
        mention1 = mention_map[m1]
        mention2 = mention_map[m2]
        tns_sentence_pairs.append((m1, m2, mention1['bert_sentence'], mention2['bert_sentence']))

    analysis_folder = dataset_folder + "/analysis/"

    first, second = zip(*tns_sentence_pairs)
    df = pd.DataFrame({'first': first, 'second':second})
    df.to_csv(analysis_folder + 'hardfns.csv')


def lemma_pair_distributions(dataset, split='train'):
    dataset_folder = f'./datasets/{dataset}/'

    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))

    mps, mps_trans = pickle.load(open(dataset_folder + f'/lh_oracle/mp_mp_t_{split}.pkl', 'rb'))

    tps, fps, tns, fns = mps_trans

    coreferent_pairs = tps + fns + fps
    non_coref_pairs = tns

    coref_lemma_pairs = []
    non_coref_lemma_pairs = []

    same_lemma_and_coref = []
    diff_lemma_and_coref = []

    true_non_coref_pairs = []
    diff_lemma_pairs = []

    for m1, m2 in coreferent_pairs:
        lemma1 = mention_map[m1]['lemma']
        lemma2 = mention_map[m2]['lemma']

        men_text1 = mention_map[m1]['mention_text'].lower()
        men_text2 = mention_map[m2]['mention_text'].lower()

        lemma1 = remove_puncts(lemma1).lower()
        lemma2 = remove_puncts(lemma2).lower()

        lemma_sim = int(lemma1.lower() in men_text2 or lemma2.lower() in men_text1
                          or lemma1.lower() in lemma2.lower())

        lemma_pair = tuple(sorted([lemma1, lemma2]))
        if lemma_sim:
            same_lemma_and_coref.append((m1, m2))
        else:
            diff_lemma_and_coref.append((m1, m2))
            diff_lemma_pairs.append(lemma_pair)

        coref_lemma_pairs.append(lemma_pair)

    coref_lemmas_set = set(coref_lemma_pairs)

    for m1, m2 in non_coref_pairs:
        lemma1 = mention_map[m1]['lemma']
        lemma2 = mention_map[m2]['lemma']

        men_text1 = mention_map[m1]['mention_text'].lower()
        men_text2 = mention_map[m2]['mention_text'].lower()

        lemma1 = remove_puncts(lemma1).lower()
        lemma2 = remove_puncts(lemma2).lower()

        lemma_sim = int(lemma1.lower() in men_text2 or lemma2.lower() in men_text1
                        or lemma1.lower() in lemma2.lower())

        if not lemma_sim:
            lemma_pair = tuple(sorted([lemma1, lemma2]))
            if lemma_pair not in coref_lemmas_set:
                non_coref_lemma_pairs.append(lemma_pair)
                true_non_coref_pairs.append((m1, m2))

    non_lemma_pair_counter = Counter(non_coref_lemma_pairs)

    most_common = non_lemma_pair_counter.most_common()

    print(len(most_common))


    # print(len(same_lemma_and_coref))
    print(len(diff_lemma_and_coref))
    pass


if __name__=='__main__':
    hard_fns('ecb', 'dev')
    # lemma_pair_distributions('ecb')

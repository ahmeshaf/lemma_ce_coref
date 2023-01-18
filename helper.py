from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import string
import os
import numpy as np
import torch

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'


def load_lemma_dataset(tsv_path, force_balance=False):
    all_examples = []
    label_map = {'POS': 1, 'NEG': 0}
    with open(tsv_path) as tnf:
        for line in tnf:
            row = line.strip().split('\t')
            mention_pair = row[:2]
            label = label_map[row[2]]
            all_examples.append((mention_pair, label))
    if force_balance:
        from collections import defaultdict
        import random
        random.seed(42)
        label2eg = defaultdict(list)

        for eg in all_examples:
            label2eg[eg[1]].append(eg)

        min_label = min(label2eg.keys(), key=lambda x: len(label2eg[x]))
        min_label_len = len(label2eg[min_label])

        max_eg_len = max([len(val) for val in label2eg.values()])
        random_egs = random.choices(label2eg[min_label], k=max_eg_len-min_label_len)
        all_examples.extend(random_egs)

        label2eg = defaultdict(list)

        for eg in all_examples:
            label2eg[eg[1]].append(eg)

        # print([len(val) for val in label2eg.values()])

    return all_examples


def get_arg_attention_mask(input_ids, parallel_model):
    """
    Get the global attention mask and the indices corresponding to the tokens between
    the mention indicators.
    Parameters
    ----------
    input_ids
    parallel_model

    Returns
    -------
    Tensor, Tensor, Tensor
        The global attention mask, arg1 indicator, and arg2 indicator
    """
    input_ids.cpu()

    num_inputs = input_ids.shape[0]

    m_start_indicator = input_ids == parallel_model.module.start_id
    m_end_indicator = input_ids == parallel_model.module.end_id

    m = m_start_indicator + m_end_indicator

    # non-zero indices are the tokens corresponding to <m> and </m>
    nz_indexes = m.nonzero()[:, 1].reshape((num_inputs, 4))

    # Now we need to make the tokens between <m> and </m> to be non-zero
    q = torch.arange(m.shape[1])
    q = q.repeat(m.shape[0], 1)

    # all indices greater than and equal to the first <m> become True
    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the first </m> become True
    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
    # all indices greater than and equal to the second <m> become True
    msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the second </m> become True
    msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

    # excluding <m> and </m> gives only the indices between <m> and </m>
    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
    msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

    # Union of indices between first <m> and </m> and second <m> and </m>
    attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
    # attention_mask_g = None
    # attention_mask_g[:, 0] = 1

    # indices between <m> and </m> excluding the <m> and </m>
    arg1 = msk_0_ar.int() * msk_1_ar.int()
    arg2 = msk_2_ar.int() * msk_3_ar.int()

    return attention_mask_g, arg1, arg2


def forward_ab(parallel_model, ab_dict, device, indices, lm_only=False):
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    if am_g_ab is not None:
        am_g_ab.to(device)
    arg1_ab.to(device)
    arg2_ab.to(device)

    return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                          global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only)


def tokenize(tokenizer, mention_pairs, mention_map, m_end, max_sentence_len=1024, text_key='bert_doc', truncate=True):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (m1, m2) in mention_pairs:
        sentence_a = mention_map[m1][text_key]
        sentence_b = mention_map[m2][text_key]

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]), \
                   ' '.join([doc_start, sent_b, doc_end])

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = input_id[curr_start_index: m_end_index] + \
                           input_id[m_end_index: m_end_index + (max_sentence_len // 4)]
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))
            input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances):
        instances_a, instances_b = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab
                             }

        return tokenized_ab_dict

    if truncate:
        tokenized_ab = ab_tokenized(pairwise_bert_instances_ab)
        tokenized_ba = ab_tokenized(pairwise_bert_instances_ba)
    else:
        instances_ab = [' '.join(instance) for instance in pairwise_bert_instances_ab]
        instances_ba = [' '.join(instance) for instance in pairwise_bert_instances_ba]
        tokenized_ab = tokenizer(list(instances_ab), add_special_tokens=False, padding=True)

        tokenized_ab_input_ids = torch.LongTensor(tokenized_ab['input_ids'])

        tokenized_ab = {'input_ids': torch.LongTensor(tokenized_ab['input_ids']),
                         'attention_mask': torch.LongTensor(tokenized_ab['attention_mask']),
                         'position_ids': torch.arange(tokenized_ab_input_ids.shape[-1]).expand(tokenized_ab_input_ids.shape)}

        tokenized_ba = tokenizer(list(instances_ba), add_special_tokens=False, padding=True)
        tokenized_ba_input_ids = torch.LongTensor(tokenized_ba['input_ids'])
        tokenized_ba = {'input_ids': torch.LongTensor(tokenized_ba['input_ids']),
                        'attention_mask': torch.LongTensor(tokenized_ba['attention_mask']),
                        'position_ids': torch.arange(tokenized_ba_input_ids.shape[-1]).expand(tokenized_ba_input_ids.shape)}

    return tokenized_ab, tokenized_ba


def cluster_cc(affinity_matrix, threshold=0.8):
    """
    Find connected components using the affinity matrix and threshold -> adjacency matrix
    Parameters
    ----------
    affinity_matrix: np.array
    threshold: float

    Returns
    -------
    list, np.array
    """
    adjacency_matrix = csr_matrix(affinity_matrix > threshold)
    clusters, labels = connected_components(adjacency_matrix, return_labels=True, directed=False)
    return clusters, labels


def remove_puncts(target_str):
    return target_str
    # return target_str.translate(str.maketrans('', '', string.punctuation)).lower()


def jc(arr1, arr2):
    return len(set.intersection(arr1, arr2)) / len(set.union(arr1, arr2))


def generate_mention_pairs(mention_map, split):
    """

    Parameters
    ----------
    mention_map: dict
    split: str (train/dev/test)

    Returns
    -------
    list: A list of all possible mention pairs within a topic
    """
    split_mention_ids = sorted([m_id for m_id, m in mention_map.items() if m['split'] == split])
    topic2mentions = {}
    for m_id in split_mention_ids:
        try:
            topic = mention_map[m_id]['predicted_topic']   # specifically for the test set of ECB
        except KeyError:
            topic = None
        if not topic:
            topic = mention_map[m_id]['topic']
        if topic not in topic2mentions:
            topic2mentions[topic] = []
        topic2mentions[topic].append(m_id)

    mention_pairs = []

    for mentions in topic2mentions.values():
        list_mentions = list(mentions)
        for i in range(len(list_mentions)):
            for j in range(i + 1):
                if i != j:
                    mention_pairs.append((list_mentions[i], list_mentions[j]))

    return mention_pairs


def generate_key_file(coref_map_tuples, name, out_dir, out_file_path):
    """

    Parameters
    ----------
    coref_map_tuples: list
    name: str
    out_dir: str
    out_file_path: str

    Returns
    -------
    None
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    clus_to_int = {}
    clus_number = 0
    with open(out_file_path, 'w') as of:
        of.write("#begin document (%s);\n" % name)
        for i, map_ in enumerate(coref_map_tuples):
            en_id = map_[0]
            clus_id = map_[1]
            if clus_id in clus_to_int:
                clus_int = clus_to_int[clus_id]
            else:
                clus_to_int[clus_id] = clus_number
                clus_number += 1
                clus_int = clus_to_int[clus_id]
            of.write("%s\t0\t%d\t%s\t(%d)\n" % (name, i, en_id, clus_int))
        of.write("#end document\n")


def cluster(mentions, mention_pairs, similarities, threshold=0):
    n = len(mentions)
    m_id2ind = {m: i for i, m in enumerate(mentions)}

    mention_ind_pairs = [(m_id2ind[mp[0]], m_id2ind[mp[1]]) for mp in mention_pairs]
    rows, cols = zip(*mention_ind_pairs)

    # create similarity matrix from the similarities
    n = len(mentions)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] = similarities

    clusters, labels = cluster_cc(similarity_matrix, threshold=threshold)
    m_id2cluster = {m: i for m, i in zip(mentions, labels)}
    return m_id2cluster


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    return sum(predicted_labels == true_labels) / len(predicted_labels)


def precision(predicted_labels, true_labels):
    """
    Precision is True Positives / All Positives Predictions
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(predicted_labels)


def recall(predicted_labels, true_labels):
    """
    Recall is True Positives / All Positive Labels
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(true_labels)


def f1_score(predicted_labels, true_labels):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels)
    R = recall(predicted_labels, true_labels)
    return 2 * P * R / (P + R)

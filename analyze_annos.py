from prodigy.components.loaders import JSONL
import pickle
from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
from helper import generate_key_file
import numpy as np

EVENT_DESCRIPTORS = ['roleset_id', 'arg0', 'arg1', 'argL', 'argT']
# EVENT_DESCRIPTORS = ['roleset_id', 'arg0', 'arg1']
# EVENT_DESCRIPTORS = ['roleset_id']


def get_prov_from_task(task_):
    task_span_ = task_['spans'][0]
    for desc in EVENT_DESCRIPTORS:
        if desc not in task_:
            task_[desc] = ''
    return '_'.join([task_['doc_id'], task_['sentence_id'],
                     str(task_span_['token_start']), str(task_span_['token_end'])])


def get_event_descriptor_id(task_):
    for desc in EVENT_DESCRIPTORS:
        if desc not in task_:
            task_[desc] = ''
    return '|'.join([task_[desc].strip() for desc in EVENT_DESCRIPTORS])


def get_gold_cluster_task(task_, mention_map_, provenance2mention_id_):
    task_provenance_id = get_prov_from_task(task_)
    task_mention_id = provenance2mention_id_[task_provenance_id]
    return task_mention_id, mention_map_[task_mention_id]['gold_cluster']


def evaluate_annotation_clusters(source_, mention_map_):
    stream = JSONL(source_)
    stream = list(stream)

    provenance_mp = ['doc_id', 'sentence_id', 'start', 'end']
    provenance2mention_id = {'_'.join([str(mention[s]) for s in provenance_mp]): m_id for m_id, mention in
                             mention_map.items()}
    source_mention_ids = [provenance2mention_id[get_prov_from_task(task)] for task in stream]

    gold_mention_ids = list(mention_map.keys())

    common_mention_ids = sorted(set(source_mention_ids).intersection(gold_mention_ids))

    for i, mention in enumerate(common_mention_ids):
        if str(mention_map_[mention]['gold_cluster']).strip('0') == '':
            mention_map_[mention]['gold_cluster'] = mention_map_[mention]['gold_cluster'] + str(i)
    # generate gold clusters key file
    curr_gold_cluster_map = [(men, mention_map_[men]['gold_cluster']) for men in common_mention_ids]
    gold_key_file = './annotations/' + f'/evt_gold.keyfile'
    generate_key_file(curr_gold_cluster_map, 'evt', './annotations/', gold_key_file)

    task2event_descriptor = {provenance2mention_id[get_prov_from_task(task)]: get_event_descriptor_id(task) for task in stream}

    mid2cluster = [(m_id, task2event_descriptor[m_id]) for m_id in common_mention_ids]

    system_key_file = './annotations/evt_annotated.keyfile'
    generate_key_file(mid2cluster, 'evt', './annotations/', system_key_file)

    def read(key, response):
        return get_coref_infos('%s' % key, '%s' % response,
                               False, False, True)

    doc = read(gold_key_file, system_key_file)
    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    print('MUC', (mr, mp, mf))
    print('B-CUB', (br, bp, bf))
    print('CEAF', (cr, cp, cf))
    print('CONLL', (mf+bf+cf)/3)


source = "./annotations/george.jsonl"
mention_map = pickle.load(open('./annotations/mention_map.pkl', 'rb'))
# evaluate_annotation_clusters(source, mention_map)


def generate_roleset_info(source_):
    stream = JSONL(source_)
    stream = list(stream)

    roleset2event_descriptor = {}

    for task in stream:
        event_descriptor = get_event_descriptor_id(task)
        roleset, arg_0, arg_1, arg_l, arg_t = event_descriptor.split('|')
        if roleset not in roleset2event_descriptor:
            roleset2event_descriptor[roleset] = set()
        roleset2event_descriptor[roleset].add((arg_0, arg_1, arg_l, arg_t, task['text']))

    # print(len(stream))
    print(len(roleset2event_descriptor))

    roleset2event_descriptor = {roleset: sorted(val, key=lambda x: len([x_i for x_i in x[:-1] if x_i.strip() != '']), reverse=True)
                                for roleset, val in roleset2event_descriptor.items()}

    with open(source_+'_roleset_syn_all.tsv', 'w') as syn_ff:
        for roleset, descs in roleset2event_descriptor.items():
            for desc in descs:
                syn_ff.write(roleset + '\t' + '\t'.join(desc) + '\n')


generate_roleset_info(source)
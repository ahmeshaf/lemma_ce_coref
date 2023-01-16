from helper import *
import pickle
import numpy as np
from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
import torch
from models import CrossEncoder
from tqdm import tqdm


def read(key, response):
    return get_coref_infos('%s' % key, '%s' % response,
            False, False, True)


def predict_dpos(parallel_model, dev_ab, dev_ba, device, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    # new_batch_size = batching(n, batch_size, len(device_ids))
    # batch_size = new_batch_size
    all_scores_ab = []
    all_scores_ba = []
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]
            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            all_scores_ab.append(scores_ab.detach().cpu())
            all_scores_ba.append(scores_ba.detach().cpu())

    return torch.cat(all_scores_ab), torch.cat(all_scores_ba)


def predict_trained_model(mention_map, model_name, linear_weights_path, test_pairs):

    linear_weights = torch.load(linear_weights_path)
    scorer_module = CrossEncoder(is_training=False, model_name=model_name,
                                      linear_weights=linear_weights).to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    tokenizer = parallel_model.module.tokenizer
    # prepare data

    test_ab, test_ba = tokenize(tokenizer, test_pairs, mention_map, parallel_model.module.end_id)

    scores_ab, scores_ba = predict_dpos(parallel_model, test_ab, test_ba, device, batch_size=64)

    return scores_ab, scores_ba


def save_dpos_scores(dataset, split, dpos_folder, heu='lh', threshold=0.5):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    tps, fps, tns, fns = mps

    tps = tps[:50]
    fps = fps[:50]

    test_pairs = tps + fps
    test_labels = [1]*len(tps) + [0]*len(fps)

    linear_weights_path = dpos_folder + "/linear.chkpt"
    bert_path = dpos_folder + '/bert'

    scores_ab, scores_ba = predict_trained_model(evt_mention_map, bert_path, linear_weights_path, test_pairs)

    predictions = torch.squeeze((scores_ab + scores_ba) / 2) > threshold

    test_labels = torch.LongTensor(test_labels)

    print("Test accuracy:", accuracy(predictions, test_labels))
    print("Test precision:", precision(predictions, test_labels))
    print("Test recall:", recall(predictions, test_labels))
    print("Test f1:", f1_score(predictions, test_labels))

    pickle.dump(test_pairs, open(dataset_folder + f'/dpos/{split}_{heu}_pairs.pkl', 'wb'))
    pickle.dump(scores_ab, open(dataset_folder + f'/dpos/{split}_{heu}_scores_ab.pkl', 'wb'))
    pickle.dump(scores_ba, open(dataset_folder + f'/dpos/{split}_{heu}_scores_ba.pkl', 'wb'))


def predict_with_dpos(dataset, split, dpos_score_map, heu='lh', threshold=0.5):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    curr_mentions = sorted(evt_mention_map.keys())

    # generate gold clusters key file
    curr_gold_cluster_map = [(men, evt_mention_map[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = dataset_folder + f'/evt_gold_{split}.keyfile'
    generate_key_file(curr_gold_cluster_map, 'evt', dataset_folder, gold_key_file)

    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    _, _, _, fns = mps_trans
    tps, fps, tns, fns_nt = mps
    print(len(tps), len(fps), len(fns))
    # print(len(fps,))
    all_mention_pairs = tps + fps + tns + fns_nt
    similarities = np.array([1] * len(tps + fps) + [0] * len(tns + fns_nt))

    w_dpos_sims = []
    for p, sim in zip(all_mention_pairs, similarities):
        if tuple(p) in dpos_score_map:
            w_dpos_sims.append(np.mean(dpos_score_map[p]))
        elif (p[1], p[0]) in dpos_score_map:
            w_dpos_sims.append(np.mean(dpos_score_map[p[1], p[0]]))
        else:
            w_dpos_sims.append(sim)

    mid2cluster = cluster(curr_mentions, all_mention_pairs, w_dpos_sims, threshold)
    system_key_file = dataset_folder + f'/evt_gold_dpos_{threshold}_{heu}_{split}.keyfile'
    generate_key_file(mid2cluster.items(), 'evt', dataset_folder, system_key_file)
    doc = read(gold_key_file, system_key_file)

    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    conf = np.round((mf + bf + cf) / 3, 1)
    print(dataset, split)
    result_string = f'& {heu} && {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\'

    print(result_string)


def predict(dataset, split, heu='lh'):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    curr_mentions = sorted(evt_mention_map.keys())

    # generate gold clusters key file
    curr_gold_cluster_map = [(men, evt_mention_map[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = dataset_folder + f'/evt_gold_{split}.keyfile'
    generate_key_file(curr_gold_cluster_map, 'evt', dataset_folder, gold_key_file)

    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    _, _, _, fns = mps_trans
    tps, fps, tns, fns_nt = mps
    print(len(tps), len(fps), len(fns))
    # print(len(fps,))
    all_mention_pairs = tps + fps + tns + fns_nt
    similarities = np.array([1]*len(tps + fps) + [0]*len(tns + fns_nt))
    mid2cluster = cluster(curr_mentions, all_mention_pairs, similarities)
    system_key_file = dataset_folder + f'/evt_gold_{heu}_{split}.keyfile'
    generate_key_file(mid2cluster.items(), 'evt', dataset_folder, system_key_file)
    doc = read(gold_key_file, system_key_file)

    ## & \LH~+ \dPos && {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\

    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3)*100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3)*100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3)*100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3)*100, 1)

    conf = np.round((mf + bf + cf)/3, 1)
    print(dataset, split)
    result_string = f'& {heu} && {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\'

    print(result_string)


def dpos_tmp(dataset, split):
    dataset_folder = f'./datasets/{dataset}'
    dpos_folder = dataset_folder + '/dpos/'

    pairs = pickle.load(open(dpos_folder + f'/{split}_pairs.pkl', 'rb'))
    ab_scores = pickle.load(open(dpos_folder + f'/{split}_scores_ab.pkl', 'rb'))
    ba_scores = pickle.load(open(dpos_folder + f'/{split}_scores_ba.pkl', 'rb'))

    dpos_map = {}
    for p, ab, ba in zip(pairs, ab_scores, ba_scores):
        dpos_map[tuple(p)] = (float(ab), float(ba))
    return dpos_map


if __name__ == '__main__':
    device = torch.device('cuda:0')
    device_ids = list(range(1))
    print('tps', 'fps',  'fns')
    # predict('ecb', DEV)
    # predict('ecb', TEST)
    #
    # predict('ecb', DEV, heu='lh_oracle')
    # predict('ecb', TEST, heu='lh_oracle')
    #
    # predict('gvc', DEV)
    # predict('gvc', TEST)
    # #
    # # predict('gvc', DEV, heu='lh_oracle')
    # predict('gvc', TEST, heu='lh_oracle')
    # dpos = dpos_tmp('ecb', 'dev')
    # predict_with_dpos('ecb', 'dev', dpos, heu='lh_oracle')
    # predict_with_dpos('ecb', 'test', dpos, heu='lh')

    dpos_path = '/home/rehan/workspace/models/dpos/chk_5/'
    save_dpos_scores('ecb', 'dev', dpos_path, heu='lh_oracle')

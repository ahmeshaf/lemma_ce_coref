import pickle
import torch
from helper import tokenize, forward_ab, f1_score, accuracy, precision, recall
from prediction import predict_dpos
import random
from tqdm import tqdm
import os
from models import CrossEncoder


def train_dpos(dataset, model_name=None):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}

    device = torch.device('cuda:0')
    device_ids = list(range(1))
    train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_oracle/mp_mp_t_train.pkl', 'rb'))
    dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh_oracle/mp_mp_t_dev.pkl', 'rb'))

    tps_train, fps_train, _, _ = train_mp_mpt
    tps_dev, fps_dev, _, _ = dev_mp_mpt

    train_pairs = list(tps_train + fps_train)
    train_labels = [1]*len(tps_train) + [0]*len(fps_train)

    dev_pairs = list(tps_dev + fps_dev)
    dev_labels = [1] * len(tps_dev) + [0] * len(fps_dev)

    # model_name = 'roberta-base'
    scorer_module = CrossEncoder(is_training=True, model_name=model_name).to(device)

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    train(train_pairs, train_labels, dev_pairs, dev_labels, parallel_model, evt_mention_map, dataset_folder, device,
          batch_size=20, n_iters=10, lr_lm=0.000001, lr_class=0.0001)


def train(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          mention_map,
          working_folder,
          device,
          batch_size=16,
          n_iters=50,
          lr_lm=0.00001,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_easy_hard_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_ab, train_ba = tokenize(tokenizer, train_pairs, mention_map, parallel_model.module.end_id, text_key='bert_sentence', max_sentence_len=512)
    dev_ab, dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.end_id, text_key='bert_sentence', max_sentence_len=512)

    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)

    for n in range(n_iters):
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        # new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        new_batch_size = batch_size
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

            scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices)

            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)

            scores_mean = (scores_ab + scores_ba) / 2

            loss = bce_loss(scores_mean, batch_labels)

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        # iteration accuracy
        dev_scores_ab, dev_scores_ba = predict_dpos(parallel_model, dev_ab, dev_ba, device, batch_size)
        dev_predictions = (dev_scores_ab + dev_scores_ba)/2
        dev_predictions = dev_predictions > 0.5
        dev_predictions = torch.squeeze(dev_predictions)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev recall:", recall(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))
        if n % 2 == 0:
            scorer_folder = working_folder + f'/scorer/chk_{n}'
            if not os.path.exists(scorer_folder):
                os.makedirs(scorer_folder)
            model_path = scorer_folder + '/linear.chkpt'
            torch.save(parallel_model.module.linear.state_dict(), model_path)
            parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
            parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
            print(f'saved model at {n}')

    scorer_folder = working_folder + '/scorer/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')


if __name__ == '__main__':
    train_dpos('ecb', model_name='roberta-base')

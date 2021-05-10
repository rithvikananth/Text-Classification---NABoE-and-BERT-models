import numpy as np
import pandas as pd
import logging
import click
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import generate_features
from model import NABoE
from optimizer import AdamW

logger = logging.getLogger(__name__)


def train(data_set, embedding, tokenizer, entity_linker, min_count, max_word_length, max_entity_length, batch_size,
          patience,
          learning_rate, weight_decay, warmup_epochs, dropout_prob, use_gpu, use_word):
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data = generate_features(data_set, tokenizer, entity_linker, min_count, max_word_length, max_entity_length)

    data_loaded_train = DataLoader(data['train'], shuffle=True, batch_size=batch_size)
    data_loaded_dev = DataLoader(data['dev'], shuffle=False, batch_size=batch_size)
    dimen_size = embedding.syn0.shape[1]

    w_vocab = data['w_vocab']
    e_vocab = data['e_vocab']

    word_embedding = np.random.uniform(low=-0.05, high=0.05, size=(len(w_vocab), dimen_size))
    word_embedding[0] = np.zeros(dimen_size)

    for w, i in w_vocab.items():
        try:
            word_embedding[i] = embedding.get_word_vector(w)
        except KeyError:
            continue

    entity_embedding = np.random.uniform(low=-0.05, high=0.05, size=(len(e_vocab), dimen_size))
    entity_embedding[0] = np.zeros(dimen_size)

    for e, i in e_vocab.items():
        try:
            entity_embedding[i] = embedding.get_entity_vector(e)
        except KeyError:
            continue

    model = NABoE(word_embedding, entity_embedding, len(data_set.label_names), dropout_prob, use_word)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                      warmup=warmup_epochs * len(data_loaded_train))
    model.to(device)

    epoch = 0
    val_best_accuracy = 0.0
    b_weights = None
    n_eps_without_imprv = 0

    while True:
        with tqdm(data_loaded_train) as pbar:

            model.train()

            for batch in pbar:
                args = {}
                for k, v in batch.items():
                    if k != 'label':
                        args[k] = v.to(device)

                logits = model(**args)
                l = F.cross_entropy(logits, batch['label'].to(device))
                l.backward()
                optimizer.step()
                model.zero_grad()
                pbar.set_description(f'epoch: {epoch} loss: {l.item():.8f}')

        epoch = epoch + 1
        v_accuracy = evaluate(model, data_loaded_dev, device, 'dev')[0]

        if v_accuracy > val_best_accuracy:
            val_best_accuracy = v_accuracy

            b_weights = {}
            for k, v in model.state_dict().items():
                b_weights[k] = v.to('cpu')

            n_eps_without_imprv = 0

        else:
            n_eps_without_imprv = n_eps_without_imprv + 1

        if n_eps_without_imprv >= patience:
            model.load_state_dict(b_weights)
            break

    test_data_loader = DataLoader(data['test'], shuffle=False, batch_size=batch_size)

    return evaluate(model, test_data_loader, device, 'test')


def evaluate(model, data_loader, dev, fold):
    model.eval()

    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for batch in data_loader:
            args = {k: v.to(dev) for k, v in batch.items() if k != 'label'}
            log_its = model(**args)
            list_argmax = []
            list_argmax = torch.argmax(log_its, 1).to('cpu').tolist()
            predictions_list = predictions_list + list_argmax
            list_batch = []
            list_batch = batch['label'].to('cpu').tolist()
            labels_list = labels_list + list_batch

    # Calculating accuracy_score and f1_score
    test_accuracy = accuracy_score(labels_list, predictions_list)
    print(f'accuracy ({fold}): {test_accuracy:.4f}')

    test_f1_score = f1_score(labels_list, predictions_list, average='macro')
    print(f'f-measure ({fold}): {test_f1_score:.4f}')

    return test_accuracy, test_f1_score


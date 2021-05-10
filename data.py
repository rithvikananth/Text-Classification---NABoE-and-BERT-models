import functools
import logging
import os
import random
import re
import unicodedata
from collections import Counter
import numpy as np
from bs4 import BeautifulSoup
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

P_TOKEN = '<PAD>'
WS_REGEXP = re.compile(r'\s+')

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, name, instances, label_names):
        self.name = name
        self.instances = instances
        self.label_names = label_names

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __len__(self):
        return len(self.instances)

    def get_instances(self, fold=None):
        if fold is None:
            return self.instances
        else:
            return [instan for instan in self.instances if instan.fold == fold]


class DatasetInstance(object):
    def __init__(self, text, label, fold):
        self.text = text
        self.label = label
        self.fold = fold


def generate_features(dataset, tokenizer, entity_linker, min_count, max_word_length, max_entity_length):
    @functools.lru_cache(maxsize=None)
    def tokenize(text):
        return tokenizer.tokenize(text)

    @functools.lru_cache(maxsize=None)
    def detect_mentions(text):
        return entity_linker.detect_mentions(text)

    def np_seq_create(source_sequence, length, dtype):
        ret = np.zeros(length, dtype=dtype)
        source_sequence = source_sequence[:length]
        ret[:len(source_sequence)] = source_sequence
        return ret

    logger.info('Creating vocabulary...')
    w_ctr = Counter()
    e_ctr = Counter()
    for i in tqdm(dataset):
        w_ctr.update(t.text for t in tokenize(i.text))
        e_ctr.update(m.title for m in detect_mentions(i.text))

    words = []
    for w, c in w_ctr.items():
        if c >= min_count:
            words.append(w)

    w_vocab = {w: i for i, w in enumerate(words, 1)}
    w_vocab[P_TOKEN] = 0

    entity_t = []
    for t, c in e_ctr.items():
        if c >= min_count:
            entity_t.append(t)

    e_vocab = {t: i for i, t in enumerate(entity_t, 1)}
    e_vocab[P_TOKEN] = 0

    ret = dict(train=[], dev=[], test=[], w_vocab=w_vocab, e_vocab=e_vocab)

    for fold in ('train', 'dev', 'test'):
        for i in dataset.get_instances(fold):
            word_ids = []
            for token in tokenize(i.text):
                if token.text in w_vocab:
                    word_ids.append(w_vocab[token.text])

            entity_ids = []
            prior_probs = []
            for m in detect_mentions(i.text):
                if m.title in e_vocab:
                    entity_ids.append(e_vocab[m.title])
                    prior_probs.append(m.prior_prob)

            ret[fold].append(dict(word_ids=np_seq_create(word_ids, max_word_length, np.int),
                                  entity_ids=np_seq_create(entity_ids, max_entity_length, np.int),
                                  prior_probs=np_seq_create(prior_probs, max_entity_length, np.float32),
                                  label=i.label))

    return ret


def load_20ng_dataset(dev_size=0.05):
    data_train = []
    data_test = []

    for fold in ('train', 'test'):
        object_dataset = fetch_20newsgroups(subset=fold, shuffle=False)

        for te, la in zip(object_dataset['data'], object_dataset['target']):
            te = normalize_text(te)
            if fold == 'train':
                data_train.append((te, la))
            else:
                data_test.append((te, la))

    size_dev = len(data_train) * dev_size
    size_dev = int(size_dev)

    random.shuffle(data_train)

    data_instance = []
    for t, l in data_train[-size_dev:]:
        data_instance.append(DatasetInstance(t, l, 'dev'))
    for t, l in data_train[:-size_dev]:
        data_instance.append(DatasetInstance(t, l, 'train'))
    for t, l in data_test:
        data_instance.append(DatasetInstance(t, l, 'test'))
    return Dataset('20ng', data_instance, fetch_20newsgroups()['target_names'])


def load_r8_dataset(dataset_path, dev_size=0.05):
    l_names = ['grain', 'earn', 'interest', 'acq', 'trade', 'crude', 'ship', 'money-fx']
    index_lab = {text: index for index, text in enumerate(l_names)}

    data_train = []
    data_test = []

    for f_name in sorted(os.listdir(dataset_path)):
        if f_name.endswith('.sgm'):
            with open(os.path.join(dataset_path, f_name), encoding='ISO-8859-1') as fi:
                for node in BeautifulSoup(fi.read(), 'html.parser').find_all('reuters'):
                    text = normalize_text(node.find('text').text)
                    la_nodes = []
                    for n in node.topics.find_all('d'):
                        la_nodes.append(n.text)

                    if len(la_nodes) != 1:
                        continue

                    la = []
                    for l in la_nodes:
                        if l in index_lab:
                            la.append(index_lab[l])

                    if len(la) == 1:
                        if node['topics'] != 'YES':
                            continue
                        if node['lewissplit'] == 'TRAIN':
                            data_train.append((text, la[0]))
                        elif node['lewissplit'] == 'TEST':
                            data_test.append((text, la[0]))
                        else:
                            continue

    size_dev = len(data_train) * dev_size
    size_dev = int(size_dev)

    random.shuffle(data_train)

    data_instance = []
    for t, l in data_train[-size_dev:]:
        data_instance.append(DatasetInstance(t, l, 'dev'))
    for t, l in data_train[:-size_dev]:
        data_instance.append(DatasetInstance(t, l, 'train'))
    for t, l in data_test:
        data_instance.append(DatasetInstance(t, l, 'test'))

    return Dataset('r8', data_instance, l_names)


def normalize_text(text):
    text = text.lower()
    text = re.sub(WS_REGEXP, ' ', text)

    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = unicodedata.normalize('NFC', text)

    return text


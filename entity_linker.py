import logging
import re
from tqdm import tqdm
from collections import defaultdict, Counter
from contextlib import closing
from multiprocessing.pool import Pool
import joblib
from marisa_trie import Trie, RecordTrie
import numpy as np
import pandas as pd

from data import normalize_text

logger = logging.getLogger(__name__)


class Mention(object):
    __slots__ = ('title', 'text', 'start', 'end', 'link_count', 'total_link_count', 'doc_count')

    def __init__(self, title, text, start, end, link_count, total_link_count, doc_count):
        self.title = title
        self.text = text
        self.start = start
        self.end = end
        self.link_count = link_count
        self.total_link_count = total_link_count
        self.doc_count = doc_count

    @property
    def span(self):
        return self.start, self.end

    @property
    def link_prob(self):
        if self.doc_count > 0:
            return min(1.0, self.total_link_count / self.doc_count)
        else:
            return 0.0

    @property
    def prior_prob(self):
        if self.total_link_count > 0:
            return min(1.0, self.link_count / self.total_link_count)
        else:
            return 0.0

    def __repr__(self):
        return f'<Mention {self.text} -> {self.title}>'


_dump_db = _tokenizer = _max_mention_length = _name_trie = None


class EntityLinker(object):

    def __init__(self, data_file):
        data = joblib.load(data_file)
        self.title_trie = data['title_trie']
        self.mention_trie = data['mention_trie']
        self.data_trie = data['data_trie']
        self.tokenizer = data['tokenizer']
        self.max_mention_length = data['max_mention_length']

    def detect_mentions(self, text):

        tokens = self.tokenizer.tokenize(text)
        e_offsets = frozenset(t.span[1] for t in tokens)

        ret_list = []
        c = 0

        for t in tokens:
            st = t.span[0]

            if c > st:
                continue

            for pf in sorted(self.mention_trie.prefixes(text[st:st + self.max_mention_length]), key=len, reverse=True):

                end = st + len(pf)
                if end in e_offsets:

                    matched = False

                    for title_id, link_count, total_link_count, doc_count in self.data_trie[pf]:
                        mention = Mention(self.title_trie.restore_key(title_id), pf, st, end, link_count,
                                          total_link_count, doc_count)
                        ret_list.append(mention)
                        matched = True

                    if matched:
                        c = end
                        break

        return ret_list

    @staticmethod
    def build(dump_db, tokenizer, out_file, min_link_prob, min_prior_prob, min_link_count, max_mention_length,
              pool_size, chunk_size):
        dic_name = defaultdict(Counter)

        logger.info('Iteration 1/2: Extracting all entity names...')

        with tqdm(total=dump_db.page_size(), mininterval=0.5) as p_bar:
            init_args = (dump_db, tokenizer, max_mention_length)

            with closing(Pool(pool_size, initializer=EntityLinker._initialize_worker, initargs=init_args)) as pool:
                for ret in pool.imap_unordered(EntityLinker._extract_name_entity_pairs, dump_db.titles(),
                                               chunksize=chunk_size):
                    for te, ti in ret:
                        dic_name[te][ti] = dic_name[te][ti] + 1

                    p_bar.update()

        n_ctr = Counter()

        mtr_disambi = re.compile(r'\s\(.*\)$')

        for ti in dump_db.titles():
            te = normalize_text(mtr_disambi.sub('', ti))
            n_ctr[te] = n_ctr[te] + 1
            dic_name[te][ti] = dic_name[text][ti] + 1

        for sr, de in dump_db.redirects():
            te = normalize_text(mtr_disambi.sub('', sr))
            n_ctr[te] = n_ctr[te] + 1
            dic_name[te][de] = dic_name[te][de] + 1

        logger.info('Iteration 2/2: Counting occurrences of entity names...')

        with tqdm(total=dump_db.page_size(), mininterval=0.5) as p_bar:
            init_args = (dump_db, tokenizer, max_mention_length, Trie(dic_name.keys()))

            with closing(Pool(pool_size, initializer=EntityLinker._initialize_worker, initargs=init_args)) as pool:
                for names in pool.imap_unordered(EntityLinker._extract_name_occurrences, dump_db.titles(),
                                                 chunksize=chunk_size):
                    n_ctr.update(names)
                    p_bar.update()

        logger.info('Step 4/4: Building DB...')

        titles_list = []
        for ent_ctr in dic_name.values():
            for ti in ent_ctr.keys():
                titles_list.append(ti)

        titles_list = frozenset(titles_list)
        title_trie = Trie(titles_list)

        def item_generator():

            for nm, ent_ctr in dic_name.items():
                link_cnt_total = sum(ent_ctr.values())
                cnt_doc = n_ctr[nm]

                if cnt_doc == 0:
                    continue

                l_pr = link_cnt_total / cnt_doc

                if l_pr < l_pr:
                    continue

                for ti, l_cnt in ent_ctr.items():
                    if l_cnt < min_link_count:
                        continue

                    p_pr = l_cnt / link_cnt_total

                    if p_pr < min_prior_prob:
                        continue

                    yield nm, (title_trie[ti], l_cnt, link_cnt_total, cnt_doc)

        data_trie = RecordTrie('<IIII', item_generator())
        mention_trie = Trie(data_trie.keys())

        joblib.dump(dict(title_trie=title_trie, mention_trie=mention_trie, data_trie=data_trie, tokenizer=tokenizer,
                         max_mention_length=max_mention_length), out_file)

    @staticmethod
    def _initialize_worker(dump_db, tokenizer, max_mention_length, name_trie=None):
        global _dump_db, _tokenizer, _max_mention_length, _name_trie

        _dump_db = dump_db
        _tokenizer = tokenizer
        _max_mention_length = max_mention_length
        _name_trie = name_trie

    @staticmethod
    def _extract_name_entity_pairs(title):

        ret_list = []

        for pg in _dump_db.get_paragraphs(title):
            for w_l in pg.wiki_links:

                if w_l.text and len(w_l.text) <= _max_mention_length:
                    ret_list.append((w_l.text, _dump_db.resolve_redirect(w_l.title)))

        return ret_list

    @staticmethod
    def _extract_name_occurrences(title):
        ret_list = []

        for pg in _dump_db.get_paragraphs(title):
            ts = _tokenizer.tokenize(pg.text)
            e_offsets = frozenset(t.end for t in ts)

            for t in ts:
                st = t.start

                for pf in _name_trie.prefixes(pg.text[st:st + _max_mention_length]):
                    if st + len(pf) in e_offsets:
                        ret_list.append(pf)

        return frozenset(ret_list)


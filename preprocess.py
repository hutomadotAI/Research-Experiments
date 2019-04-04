import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os
import csv
import pickle
import argparse
import requests


'''
taken from https://github.com/NLPLearn/QANet/blob/master/prepro.py
and previously taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


nlp = spacy.blank("en")


def get_w2v(words, use='glove', lang='en', use_random_unk=False, ip_addr="http://10.8.0.22:8088"):
    # predict model
    if use == 'w2v':
        url = ip_addr + "/w2v/get"
    elif use == 'glove':
        url = ip_addr + "/glove/get"
    elif use == 'glove6B':
        url = ip_addr + "/glove6B/get"
    elif use == 'fasttext':
        url = ip_addr + "/fasttext/get"
    else:
        NotImplementedError('only glove, glove6B and w2v embeddings exist at the moment')
    data = {'words': words, 'lang': lang, 'random_unk': use_random_unk}
    headers = {'content-type': 'application/json'}
    result = requests.post(url, data=json.dumps(data), headers=headers)
    out = result.json()['embeddings']
    return out


def init_embedding(vocab_processor, embedding_dim, use='glove', ip_addr="10.8.0.22:8088"):
    vocab_len = len(vocab_processor)
    words = list(vocab_processor.keys())

    # initial matrix with random uniform
    initW = np.random.uniform(-0.25, 0.25, (vocab_len, embedding_dim))
    print("vocab: {}".format(vocab_len))
    w2v = get_w2v(words, use=use, ip_addr=ip_addr)
    for w, a in w2v.items():
        idx = vocab_processor[w]
        initW[idx] = np.array(a)
    print("initialised {}/{} words".format(len(w2v), vocab_len))
    return initW


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r", encoding="utf-8") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_word_dicts(counter, data_type, limit=-1, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    filtered_elements = [k for k, v in counter.items() if v > limit]

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        filtered_elements, 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    idx2token_dict = {idx: token for token, idx in token2idx_dict.items()}
    return token2idx_dict, idx2token_dict


def build_features(examples, data_type, out_files, out_files_char, word2idx_dict, char2idx_dict,
                   para_limit, ques_limit, char_limit, is_test=False):

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    meta = {}
    con_ids, quest_ids, ans_span, qa_ids = [], [], [], []
    con_char_ids, quest_char_ids = [], []
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0

        con_ids.append(context_idxs.tolist())
        quest_ids.append(ques_idxs.tolist())
        ans_span.append([np.argmax(y1), np.argmax(y2)])
        con_char_ids.append(context_char_idxs)
        quest_char_ids.append(ques_char_idxs)
        qa_ids.append([example["id"]])

    save_csv(out_files[0], con_ids, message='context_ids')
    save_csv(out_files[1], quest_ids, message='question_ids')
    save_csv(out_files[2], ans_span, message='answer_span')
    save_csv(out_files[3], qa_ids, message='qa_ids')

    pickle.dump(con_char_ids, open(out_files_char[0], 'wb'))
    pickle.dump(quest_char_ids, open(out_files_char[1], 'wb'))

    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return meta


def save_csv(filename, obj, message=None):
    print('Saving {} ...'.format(message))
    assert isinstance(obj, list), 'obj must be list'
    with open(filename, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(obj)


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def save_initW(vocab_proc, data_dir, embedding_dim=300, use='w2v', ip_addr="10.8.0.22:8088"):
    initW = init_embedding(vocab_proc, embedding_dim, use=use, ip_addr=ip_addr)
    print("shape of initW: {}".format(initW.shape))
    initW.tofile(os.path.join(data_dir, 'initW_' + use + '.dat'))


def prepro():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='/storage/datasets/squad-phrase/data3/')
    parser.add_argument("--in_dir", default='/storage/datasets/squad-phrase/download')
    parser.add_argument("--w2v", default='glove')
    parser.add_argument("--max_para_len", default=400)
    parser.add_argument("--max_para_len_test", default=1000)
    parser.add_argument("--max_quest_len", default=50)
    parser.add_argument("--max_quest_len_test", default=100)
    parser.add_argument("--max_char_len", default=16)
    parser.add_argument("--emb_dim", default=300)
    args = parser.parse_args()
    print(args)

    # set random seed
    random.seed(1234)

    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(
        os.path.join(args.in_dir, 'train-v1.1.json'), "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(
        os.path.join(args.in_dir, 'dev-v1.1.json'), "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(
        os.path.join(args.in_dir, 'dev-v1.1.json'), "test", word_counter, char_counter)

    word2idx_dict = None
    word2idx_file = os.path.join(args.data_dir, "word2idx.json")
    idx2word_file = os.path.join(args.data_dir, "idx2word.json")
    if os.path.isfile(word2idx_file):
        with open(word2idx_file, "r") as fh:
            word2idx_dict = json.load(fh)
            print("word2idx loaded")
    word2idx_dict, idx2word_dict = get_word_dicts(
        word_counter, "word", token2idx_dict=word2idx_dict)
    save(word2idx_file, word2idx_dict, message="word2idx")
    save(idx2word_file, idx2word_dict, message="idx2word")

    save_initW(word2idx_dict, args.data_dir, use=args.w2v,
               ip_addr=args.ip_addr, embedding_dim=args.emb_dim)

    char2idx_dict = None
    char2idx_file = os.path.join(args.data_dir, "char2idx.json")
    idx2char_file = os.path.join(args.data_dir, "idx2char.json")
    if os.path.isfile(char2idx_file):
        with open(char2idx_file, "r") as fh:
            print("char2idx loaded")
            char2idx_dict = json.load(fh)
    char2idx_dict, idx2char_dict = get_word_dicts(
        char_counter, "char", token2idx_dict=char2idx_dict)
    save(char2idx_file, char2idx_dict, message="char2idx")
    save(idx2char_file, idx2char_dict, message="idx2char")
    save(os.path.join(args.data_dir, "train_eval.json"), train_eval, message="train eval")
    save(os.path.join(args.data_dir, "dev_eval.json"), dev_eval, message="dev eval")
    save(os.path.join(args.data_dir, "test_eval.json"), test_eval, message="test eval")

    build_features(train_examples, "train",
                   [os.path.join(args.data_dir, 'train.ids.context'),
                    os.path.join(args.data_dir, 'train.ids.question'),
                    os.path.join(args.data_dir, 'train.span'),
                    os.path.join(args.data_dir, 'train.qa_ids')],
                   [os.path.join(args.data_dir, 'train.char_ids.context'),
                    os.path.join(args.data_dir, 'train.char_ids.question')],
                   word2idx_dict, char2idx_dict, args.max_para_len, args.max_quest_len, args.max_char_len)
    dev_meta = build_features(dev_examples, "dev",
                   [os.path.join(args.data_dir, 'dev.ids.context'),
                    os.path.join(args.data_dir, 'dev.ids.question'),
                    os.path.join(args.data_dir, 'dev.span'),
                    os.path.join(args.data_dir, 'dev.qa_ids')],
                   [os.path.join(args.data_dir, 'dev.char_ids.context'),
                    os.path.join(args.data_dir, 'dev.char_ids.question')],
                   word2idx_dict, char2idx_dict, args.max_para_len, args.max_quest_len, args.max_char_len)
    test_meta = build_features(test_examples, "test",
                   [os.path.join(args.data_dir, 'test.ids.context'),
                    os.path.join(args.data_dir, 'test.ids.question'),
                    os.path.join(args.data_dir, 'test.span'),
                    os.path.join(args.data_dir, 'test.qa_ids')],
                   [os.path.join(args.data_dir, 'test.char_ids.context'),
                    os.path.join(args.data_dir, 'test.char_ids.question')],
                   word2idx_dict, char2idx_dict, args.max_para_len_test, args.max_quest_len_test,
                   args.max_char_len, is_test=True)

    save(os.path.join(args.data_dir, "dev_meta.json"), dev_meta, message="dev meta")
    save(os.path.join(args.data_dir, "test_meta.json"), test_meta, message="test meta")


if __name__ == '__main__':
    prepro()

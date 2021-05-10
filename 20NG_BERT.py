# import re
import sys
import os
import tensorflow as tf
import torch
import numpy as np
# from official.nlp import optimization  # to create AdamW optimizer
from sklearn.model_selection import train_test_split
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
# import unicodedata
# from bs4 import BeautifulSoup

tf.get_logger().setLevel('ERROR')
CUDA_LAUNCH_BLOCKING = 1  # To check error log if there are any CUDA errors.

# Assign the cuda if available. GPU is better suited for calculations using Tensors.
if torch.cuda.is_available():
    device = torch.device('cuda')
    devicename = '[' + torch.cuda.get_device_name(0) + ']'
else:
    device = torch.device('cpu')
    devicename = ""


def get_dataset():
    data_dir = tf.keras.utils.get_file('20news', origin='http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz',
                                       extract=True)
    dataset_dir = os.path.join(os.path.dirname(data_dir), '20news-18828')
    return dataset_dir


DATA_DIRECTORY = get_dataset()
CACHE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'transformers-cache')
BERTMODEL = 'bert-base-uncased'


def generate_text_label():
    sentences = []  # To store the text sentences for classification
    int_labels = {}  # dictionary label name to numeric id
    labels = []  # list of label ids
    dataset_dir = get_dataset()
    for name in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path):
            label_id = len(int_labels)
            int_labels[name] = label_id
            for file_name in sorted(os.listdir(path)):
                if file_name.isdigit():
                    fpath = os.path.join(path, file_name)
                    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        sentences.append(t)
                    labels.append(label_id)

    return sentences, labels, int_labels


def tokenise_dataset(sentences, labels):
    TEST_SET = 3500
    (sentences_train, sentences_test,
     labels_train, labels_test) = train_test_split(sentences, labels,
                                                   test_size=TEST_SET, shuffle=True, random_state=42)

    # Adding the CLS token for beginning of the sentence.

    sentences_train = ["[CLS] " + s for s in sentences_train]
    sentences_test = ["[CLS] " + s for s in sentences_test]

    tokenizer = BertTokenizer.from_pretrained(BERTMODEL, cache_dir=CACHE_DIRECTORY,
                                              do_lower_case=True)

    tokenized_train = [tokenizer.tokenize(s) for s in sentences_train]
    tokenized_test = [tokenizer.tokenize(s) for s in sentences_test]

    MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512

    tokenized_train = [t[:(MAX_LEN_TRAIN - 1)] + ['SEP'] for t in tokenized_train]
    tokenized_test = [t[:(MAX_LEN_TEST - 1)] + ['SEP'] for t in tokenized_test]

    print("The truncated tokenized first training sentence:")
    print(tokenized_train[0])
    # Next we use the BERT tokenizer to convert each token into an integer
    # index in the BERT vocabulary. We also pad any shorter sequences to
    # `MAX_LEN_TRAIN` or `MAX_LEN_TEST` indices with trailing zeros.

    ids_train = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_train]
    ids_train = np.array([np.pad(i, (0, MAX_LEN_TRAIN - len(i)),
                                 mode='constant') for i in ids_train])

    ids_test = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_test]
    ids_test = np.array([np.pad(i, (0, MAX_LEN_TEST - len(i)),
                                mode='constant') for i in ids_test])

    amasks_train, amasks_test = [], []

    for seq in ids_train:
        seq_mask = [float(i > 0) for i in seq]
        amasks_train.append(seq_mask)

    for seq in ids_test:
        seq_mask = [float(i > 0) for i in seq]
        amasks_test.append(seq_mask)

    # We use again scikit-learn's train_test_split to use 10% of our
    # training data as a validation set, and then convert all data into
    # torch.tensors.
    return ids_train, ids_test, labels_train, labels_test, amasks_train, amasks_test


def generate_data_loaders():
    sentences, labels, label_int = generate_text_label()
    ids_train, ids_test, labels_train, labels_test, amasks_train, amasks_test = tokenise_dataset(sentences, labels)

    (train_inputs, validation_inputs, train_labels, validation_labels) = train_test_split(ids_train, labels_train,
                                                                                          random_state=42,
                                                                                          test_size=0.1)
    (train_masks, validation_masks, _, _) = train_test_split(amasks_train, ids_train,
                                                             random_state=42, test_size=0.1)

    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)
    test_inputs = torch.tensor(ids_test)
    test_labels = torch.tensor(labels_test)
    test_masks = torch.tensor(amasks_test)

    # Next we create PyTorch DataLoaders for all data sets.
    #
    # For fine-tuning BERT on a specific task, the authors recommend a
    # batch size of 16 or 32.

    BATCH_SIZE = 16

    train_data = TensorDataset(train_inputs, train_masks,
                               train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=BATCH_SIZE)
    validation_data = TensorDataset(validation_inputs, validation_masks,
                                    validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_loader = DataLoader(validation_data,
                                   sampler=validation_sampler,
                                   batch_size=BATCH_SIZE)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler,
                             batch_size=BATCH_SIZE)

    return train_dataloader, test_loader, validation_loader


train_dataloader, test_dataloader, validation_dataloader = generate_data_loaders()
model = BertForSequenceClassification.from_pretrained(BERTMODEL,
                                                      cache_dir=CACHE_DIRECTORY,
                                                      num_labels=20)
model.cuda()
# We also need to grab the training parameters from the pretrained model.

EPOCHS = 34
WEIGHT_DECAY = 0.01
LRATE = 2e-5
WARMUP_STEPS = int(0.2 * len(train_dataloader))

no_decay = ['bias', 'LayerNorm.weight']
optimizer_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_parameters, lr=LRATE, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=WARMUP_STEPS,
                                            num_training_steps=len(train_dataloader) * EPOCHS)


def train(epoch, loss_vector=None, log_interval=200):
    model.train()
    # Loop over each batch from the training set
    for step, batch in enumerate(train_dataloader):

        # Copy data to GPU if needed
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)

        loss = outputs[0]
        if loss_vector is not None:
            loss_vector.append(loss.item())

        # Zero gradient buffers
        optimizer.zero_grad()
        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(b_input_ids),
                len(train_dataloader.dataset),
                       100. * step / len(train_dataloader), loss))


def evaluate(loader):
    # set model to eval mode
    model.eval()
    pred_correct, total_samples = 0, 0
    original_labels = []
    predicted_labels = []
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
            out_logits = outputs[0]

        out_logits = out_logits.detach().cpu().numpy()
        predictions = np.argmax(out_logits, axis=1)

        labels = b_labels.to('cpu').numpy()
        pred_correct += np.sum(predictions == labels)
        total_samples += len(labels)
        original_labels.extend(labels)
        predicted_labels.extend(predictions)

    print(f'Accuracy: [{pred_correct}/{total_samples}] {pred_correct / total_samples:.4f}\n')
    print(f'f1_score:  {f1_score(original_labels, predicted_labels, average="macro"):.4f}\n')


train_loss = []
for epoch in range(1, EPOCHS + 1):
    train(epoch, train_loss)
    print(' For Validation set:')
    evaluate(validation_dataloader)

print('For Test set:')
evaluate(test_dataloader)





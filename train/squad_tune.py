import random
random.seed(1995)

import argparse
import glob
import logging
import os
import random
import timeit
import pdb
import collections
import json
import tensorflow as tf

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)

from transformers.data.processors.squad import SquadV2Processor, SquadExample

import json
from pathlib import Path

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

train_contexts, train_questions, train_answers = read_squad('train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad('dev-v2.0.json')

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        if context[start_idx:end_idx].lower() == gold_text.lower():
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1].lower() == gold_text.lower():
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2].lower() == gold_text.lower():
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

from transformers import BertTokenizerFast, BertModel

tokenizer = BertTokenizerFast.from_pretrained("dmis-lab/biobert-base-cased-v1.1", padding = True,
                                             truncation=True,add_special_tokens = True,
                                              model_max_length = 1000000000)

#model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True,max_length=500)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True,max_length=500)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)


import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        #print(self.encodings['start_positions'][idx])
         #{key: torch.tensor(val[idx], dtype = torch.long) for key, val in self.encodings.items()}
        return {'input_ids':torch.tensor(self.encodings['input_ids'][idx],dtype = torch.long),
         'attention_mask':torch.tensor(self.encodings['attention_mask'][idx],dtype = torch.long),
         'start_positions':torch.tensor(self.encodings['start_positions'][idx],dtype = torch.long),
         'end_positions':torch.tensor(self.encodings['end_positions'][idx],dtype = torch.long)}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)


from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

from barbar import Bar

for epoch in range(3):
  for i,batch in enumerate(Bar(train_loader)):
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device, dtype = torch.long)
    #print(input_ids)
    attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
    #print(attention_mask)
    start_positions = batch['start_positions'].to(device, dtype = torch.long)
    #print(start_positions)
    end_positions = batch['end_positions'].to(device, dtype = torch.long)
    #print(end_positions)
    outputs = model(input_ids, 
                    attention_mask=attention_mask, 
                    start_positions=start_positions, 
                    end_positions=end_positions)
    loss = outputs[0]
    loss.backward()
    optim.step()
model.eval()

torch.save({
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            },'checkpoint_squad_3epochs_1995.pt')


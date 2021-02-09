#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import torch
import json
import jsonlines
from pathlib import Path
from barbar import Bar
import random
from torch import nn

random.seed(1995)
# In[30]:


with open('preproc_datasets/BioASQ-train-yesno-8b-snippet.json', 'rb') as f:
    bio_yn_raw = json.load(f)['data'][0]['paragraphs']
bio_yn = [q['qas'][0] for q in bio_yn_raw]
for i in range(len(bio_yn)):
    bio_yn[i]['context'] = bio_yn_raw[i]['context']
bio_yn_df = pd.DataFrame.from_dict(bio_yn)
bio_yn_df.head()


# In[38]:


train_a = list(bio_yn_df_filtered.question)
train_b = list(bio_yn_df_filtered.context)
train_labels = [int(answer == 'yes') for answer in bio_yn_df.answers]


# In[3]:


from transformers import BertTokenizer
# Load the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1', 
                                          do_lower_case=True)


# In[39]:


train_tokens = tokenizer(train_a,train_b, 
                       add_special_tokens=True,
                       max_length=500,
                       truncation=True, padding=True)
train_tokens['labels'] = train_labels


# In[40]:


from torch.utils.data import Dataset, DataLoader

class MnliDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        #print(self.encodings['start_positions'][idx])
        #{key: torch.tensor(val[idx], dtype = torch.long) for key, val in self.encodings.items()}
        return {'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype = torch.long),
                'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype = torch.long),
                'token_type_ids': torch.tensor(self.encodings['token_type_ids'][idx], dtype = torch.long),
                'labels': torch.tensor(self.encodings['labels'][idx], dtype = torch.long)
               }

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = MnliDataset(train_tokens)


# In[5]:


# freeze all the parameters
for param in model.parameters():
    param.requires_grad = False


# In[4]:


from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels = 4)
checkpoint = torch.load('checkpoint_mnli_3epochs_seed.pt',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])


# In[73]:


class BERT_Arch(nn.Module):

    def __init__(self, model):
      
        super(BERT_Arch, self).__init__()

        self.model = model
        
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(4,512)
        
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, input_ids,
            attention_mask,
            token_type_ids,labels):

        #pass the inputs to the model  
        outputs = self.model(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,labels = labels)
        
        cls_hs = outputs.logits
        
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        
        # apply softmax activation
        x = self.softmax(x)

        return x


# In[74]:


model_full = BERT_Arch(model)


# In[81]:


from torch.utils.data import DataLoader
from transformers import AdamW
from torch.nn import DataParallel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_full.to(device)
model_full.train()

model_full = DataParallel(model_full, device_ids = [0,1,2,3])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)


# In[83]:


cross_entropy  = nn.NLLLoss() 
for epoch in range(3):
    for i,batch in enumerate(Bar(train_loader)):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device, dtype = torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)
        outputs = model_full(input_ids, 
                        attention_mask=attention_mask, 
                        token_type_ids = token_type_ids,
                        labels = labels)
        #loss = outputs.loss
        loss = cross_entropy(outputs, labels)
        loss.backward()
        optim.step()
model_full.eval()


# In[ ]:

torch.save({
            'epoch': 3,
            'model_state_dict': model_full.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            },'checkpoint_bio_yn_3epochs_seed.pt')




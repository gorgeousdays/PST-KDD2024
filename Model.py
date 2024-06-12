import os
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, BertModel
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import logging


class Model(nn.Module):
    def __init__(self, bert_model, oagbert):
        super().__init__()
        self.bert_model = bert_model
        self.oagbert = oagbert
        self.mlp1 = nn.Linear(2 * 768, 768) 
        self.mlp2 = nn.Linear(2 * 768, 100) 
        # self.mlp3 = nn.Linear(768, 100)
        self.mlp = nn.Linear(100, 2)
        # self.relu = nn.PReLU() # 2
        # self.relu = nn.PReLU() # 2-1
        self.relu = nn.LeakyReLU() #2
        self.sigmoid = nn.Sigmoid()
        self.init_embedding = torch.ones((1, 768), requires_grad=False)
        # self.init_embedding = torch.zeros((1, 768), requires_grad=False)
    
    def forward(self, input_ids, attention_mask, token_type_ids, oag_token, oag_token_paper):
        output = self.bert_model(input_ids, attention_mask, token_type_ids)[1] # [B, 768]
        # input = self.process(oag_token, output.device)

        # print(output.device, input['input_ids'].device)
        # _, output2 = self.oagbert.bert.forward(**input) # [B, 768]
        # print(output2.device)
        output2 = self.process_oag_token(oag_token, output.device)
        output3 = self.process_oag_token(oag_token_paper, output.device)
        out1 = self.mlp1(torch.concat([output2, output3], dim=1))
        out1 = self.relu(out1)
        out = torch.concat([output, out1], dim=1)
        out = self.relu(self.mlp2(out))
        # out = self.relu(self.mlp3(out))
        return self.sigmoid(self.mlp(out))
        
    def process(self, oag_token, device='cpu'):
        input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, \
            position_ids_second, masked_positions, num_spans = self.process_oag_token(oag_token)
            
        print(type(input_ids), len(input_ids), input_ids)
        inc = input_ids
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        print(input_ids.shape, input_ids)
        token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device)
        attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device)
        position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device)
        position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).to(device)

        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'token_type_ids':token_type_ids,
                'output_all_encoded_layers':False, 'checkpoint_activations':False,
                'position_ids':position_ids, 'position_ids_second':position_ids_second}

    def process_oag_token(self, oag_token, device='cpu'):
        output = []
        # print(oag_token, oag_token.keys())
        # title_all, abstract, venue, concepts, authors, affiliations = oag_token.values():
        batch_size = len(oag_token['title'])
        oag_token['concepts'] = oag_token['authors'] = oag_token['affiliations'] = [[] for i in range(batch_size)]
        # print(oag_token['venue'], *oag_token.values())
        # print(oag_token)
        data = zip(*oag_token.values())
        # print(oag_token.values())
        for title, abstract, authors, venue, concepts, affiliations in data:
            input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, \
                position_ids_second, masked_positions, num_spans = self.oagbert.build_inputs(
                title=title, abstract=abstract, venue=venue, authors=authors, \
                    concepts=concepts, affiliations=affiliations)
            if len(input_ids) == 0:
                output.append(self.init_embedding.to(device))
                continue
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device)
            attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device)
            position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device)
            position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).to(device)
            input = {'input_ids':input_ids, 'attention_mask':attention_mask, 'token_type_ids':token_type_ids,
                'output_all_encoded_layers':False, 'checkpoint_activations':False,
                'position_ids':position_ids, 'position_ids_second':position_ids_second}
            _, output2 = self.oagbert.bert.forward(**input) # [B, 768]
            output.append(output2)
        # for k in output:
        #     print(k.shape, k.device)
        # print(len(output), output)
        return torch.concat(output, dim=0).to(device)
        # return input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, \
        #     position_ids_second, masked_positions, num_spans

class Model_OnlySim(nn.Module):
    def __init__(self, oagbert):
        super().__init__()
        self.oagbert = oagbert
        self.mlp = nn.Linear(2 * 768, 384)
        self.mlp1 = nn.Linear(384, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_embedding = torch.zeros((1, 768), requires_grad=False)
    
    def forward(self, oag_token, oag_token_paper, device):
        output = self.process_oag_token(oag_token, device)
        output1 = self.process_oag_token(oag_token_paper, device)
        out1 = self.mlp(torch.concat([output, output1], dim=1))
        out1 = self.relu(out1)
        return self.sigmoid(self.mlp1(out1))
        

    def process_oag_token(self, oag_token, device='cpu'):
        output = []
        batch_size = len(oag_token['title'])
        oag_token['concepts'] = oag_token['authors'] = oag_token['affiliations'] = [[] for i in range(batch_size)]
        data = zip(*oag_token.values())
        for title, abstract, authors, venue, concepts, affiliations in data:
            input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, \
                position_ids_second, masked_positions, num_spans = self.oagbert.build_inputs(
                title=title, abstract=abstract, venue=venue, authors=authors, \
                    concepts=concepts, affiliations=affiliations)
            if len(input_ids) == 0:
                output.append(self.init_embedding.to(device))
                continue
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device)
            attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device)
            position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device)
            position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).to(device)
            input = {'input_ids':input_ids, 'attention_mask':attention_mask, 'token_type_ids':token_type_ids,
                'output_all_encoded_layers':False, 'checkpoint_activations':False,
                'position_ids':position_ids, 'position_ids_second':position_ids_second}
            _, output2 = self.oagbert.bert.forward(**input) # [B, 768]
            output.append(output2)
        return torch.concat(output, dim=0).to(device)
    
class Model_Feature(nn.Module):
    def __init__(self, bert_model, oagbert, feature_dim=12):
        super().__init__()
        self.bert_model = bert_model
        self.oagbert = oagbert
        self.mlp1 = nn.Linear(2 * 768, 768) 
        self.mlp = nn.Linear(2 * 768, feature_dim)

        self.mlp2 = nn.Linear(feature_dim, feature_dim)
        self.mlp3 = nn.Linear(feature_dim * 2, 2) 

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_embedding = torch.zeros((1, 768), requires_grad=False)
    
    def forward(self, input_ids, attention_mask, token_type_ids, oag_token, oag_token_paper, feature):
        output = self.bert_model(input_ids, attention_mask, token_type_ids)[1] # [B, 768]

        output2 = self.process_oag_token(oag_token, output.device)
        output3 = self.process_oag_token(oag_token_paper, output.device)
        out1 = self.mlp1(torch.concat([output2, output3], dim=1))
        out1 = self.relu(out1)
        out = torch.concat([output, out1], dim=1)
        out = self.relu(self.mlp(out))  # 这儿考虑要不要加dropout

        out_f = self.relu(self.mlp2(feature))
        # print(out.shape, out_f.shape)
        return self.sigmoid(self.mlp3(torch.concat((out, out_f), dim=1)))
        

    def process_oag_token(self, oag_token, device='cpu'):
        output = []
        batch_size = len(oag_token['title'])
        oag_token['concepts'] = oag_token['authors'] = oag_token['affiliations'] = [[] for i in range(batch_size)]

        data = zip(*oag_token.values())
        # print(oag_token.values())
        for title, abstract, authors, venue, concepts, affiliations in data:
            input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, \
                position_ids_second, masked_positions, num_spans = self.oagbert.build_inputs(
                title=title, abstract=abstract, venue=venue, authors=authors, \
                    concepts=concepts, affiliations=affiliations)
            if len(input_ids) == 0:
                output.append(self.init_embedding.to(device))
                continue
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).to(device)
            attention_mask=torch.LongTensor(input_masks).unsqueeze(0).to(device)
            position_ids=torch.LongTensor(position_ids).unsqueeze(0).to(device)
            position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0).to(device)
            input = {'input_ids':input_ids, 'attention_mask':attention_mask, 'token_type_ids':token_type_ids,
                'output_all_encoded_layers':False, 'checkpoint_activations':False,
                'position_ids':position_ids, 'position_ids_second':position_ids_second}
            _, output2 = self.oagbert.bert.forward(**input) # [B, 768]
            output.append(output2)

        return torch.concat(output, dim=0).to(device)

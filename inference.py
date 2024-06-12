import os
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import torch
import json
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, BertModel
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import logging

from cogdl.oag import oagbert
from Model import Model, Model_OnlySim
from data_processor import convert_examples_to_inputs, get_data_loader
import utils
import settings


# 设置随机数种子
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(seed)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MAX_SEQ_LENGTH=512


def gen_kddcup_test_submission_bert(model_name="scibert", save_path='kddcup'):
    print("model name", model_name)
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    data_dir2 = join(data_dir, 'OAG')
    papers = utils.load_json(data_dir2, "paper_source_trace_test_wo_ans.json")
    
    with open('data/PST/id_to_title602.json', 'r', encoding='utf-8') as f:
        pid_to_title = json.load(f)
    with open('data/data/test_pid_to_bid.json', 'r', encoding='utf-8') as f:
        pid_to_bid = json.load(f)        
    
    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "scibert":
        BERT_MODEL = "allenai/scibert_scivocab_uncased"
    elif model_name == "scincl":
        BERT_MODEL = "malteos/scincl"
    elif model_name == "citebert":
        BERT_MODEL = "copenlu/citebert"
    elif model_name == "specter2":
        BERT_MODEL = "allenai/specter2_base"
    else:
        raise NotImplementedError
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    # model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    # model.load_state_dict(torch.load(join(settings.OUT_DIR, "kddcup", model_name, "pytorch_model.bin"))) # 加载保存在out上的模型

    model = BertModel.from_pretrained(BERT_MODEL)
    _, model2 = oagbert("oagbert-v2-sim")
    model = Model(model, model2)
    
    model.load_state_dict(torch.load(join(settings.OUT_DIR, save_path, model_name, "pytorch_model.bin")))

    model.to(device)
    model.eval()
    
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(f'Total number of parameters: {total_params}')

    BATCH_SIZE = 16

    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx
        bib_to_contexts = utils.find_bib_context(xml)
        # bib_sorted = sorted(bib_to_contexts.keys())
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]
        
        y_score_dict = {cur_pid: [0] * n_refs}
        y_score = [0] * n_refs
        assert len(sub_example_dict[cur_pid]) == n_refs
        # continue
        # [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        contexts_sorted = {cur_pid:[]}
        for bib in bib_sorted:
            pid1 = None
            try:
                title = bid_to_title[bib]
                for k in pid_to_title:
                    if fuzz.ratio(title, pid_to_title[k]) >= 80:
                        pid1 = k
                if pid1 is None:
                    for k, v in pid_to_bid[cur_pid].items():
                        if bib == v:
                            pid1 = k
                            break       
            except:
                pid1 = None
                if pid1 is None:
                    for k, v in pid_to_bid[cur_pid].items():
                        if bib == v:
                            pid1 = k
                            break   
                
            content = ' '.join(bib_to_contexts[bib])
            contexts_sorted[cur_pid].append({'pid': pid1, 'content': content})
            

        test_features = convert_examples_to_inputs(contexts_sorted, y_score_dict, MAX_SEQ_LENGTH, tokenizer)
        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

        predicted_scores = []

        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids, label_ids, oag_token, oag_token_paper = batch 
            with torch.no_grad():
                r = model(input_ids, attention_mask=input_mask,
                                            token_type_ids=segment_ids, 
                                            oag_token=oag_token, oag_token_paper=oag_token_paper)
                # r = model(oag_token=oag_token, oag_token_paper=oag_token_paper, device=device)
                # tmp_eval_loss = r[0]
                # logits = r[1]
                logits = r # [B, 2]

            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)
        
        for ii in range(len(predicted_scores)):
            bib_idx = int(bib_sorted[ii][1:])
            # print("bib_idx", bib_idx)
            y_score[bib_idx] = float(utils.sigmoid(predicted_scores[ii]))
        
        sub_dict[cur_pid] = y_score
    
    utils.dump_json(sub_dict, join(settings.OUT_DIR, save_path, model_name), "test_submission_scibert.json")


if __name__ == "__main__":
    model_name = "specter2"
    gen_kddcup_test_submission_bert(model_name=model_name, save_path=model_name) #
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

def prepare_oag_bert_input():
    x_train = dd(list)
    y_train = dd(list)
    x_valid = dd(list)
    y_valid = dd(list)

    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    data_dir2 = join(data_dir, 'OAG')
    papers = utils.load_json(data_dir2, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    # n_train = int(n_papers * 2 / 3)
    n_train = int(n_papers * 6 / 7)
    # n_valid = n_papers - n_train

    papers_train = papers[:n_train]
    papers_valid = papers[n_train:]

    pids_train = {p["_id"] for p in papers_train}
    pids_valid = {p["_id"] for p in papers_valid}

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    with open('data/PST/id_to_title.json', 'r', encoding='utf-8') as f:
        pid_to_title = json.load(f)
    with open('data/data/train_pid_to_bid.json', 'r', encoding='utf-8') as f:
        pid_to_bid = json.load(f)    
        
    pid_to_source_titles = dd(list)
    for paper in tqdm(papers):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    # files = sorted(files)
    # for file in tqdm(files):
    for cur_pid in tqdm(pids_train | pids_valid):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
            # continue
        f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

        source_titles = pid_to_source_titles[cur_pid]  # json文件中的 ref 参考文献题目 1-2个
        if len(source_titles) == 0:
            continue

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
            bid_to_title[bid] = ref.analytic.title.text.lower() # xml文件里的参考文献
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx
        
        flag = False

        cur_pos_bib = set()

        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    cur_pos_bib.add(bid)
        
        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        
        if not flag:
            continue
    
        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue
    
        bib_to_contexts = utils.find_bib_context(xml)

        n_pos = len(cur_pos_bib)
        n_neg = n_pos * 10
        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)

        if cur_pid in pids_train:
            cur_x = x_train
            cur_y = y_train
        elif cur_pid in pids_valid:
            cur_x = x_valid
            cur_y = y_valid
        else:
            continue
            # raise Exception("cur_pid not in train/valid/test")
        
        for bib in cur_pos_bib:
            cur_context = " ".join(bib_to_contexts[bib]) # 有的文献会在论文中引用多次
            title = bid_to_title[bib]
            pid1 = None 
            for k in pid_to_title:
                if fuzz.ratio(title, pid_to_title[k]) >= 80:
                    pid1 = k
            if pid1 is None:
                for k, v in pid_to_bid[cur_pid].items():
                    if bib == v:
                        pid1 = k
                        break  
            cur_x[cur_pid].append({'pid': pid1, 'content': cur_context, })
            cur_y[cur_pid].append(1)
    
        for bib in cur_neg_bib_sample:
            cur_context = " ".join(bib_to_contexts[bib])
            title = bid_to_title[bib]
            pid1 = None
            for k in pid_to_title:
                if fuzz.ratio(title, pid_to_title[k]) >= 80:
                    pid1 = k
            if pid1 is None:
                for k, v in pid_to_bid[cur_pid].items():
                    if bib == v:
                        pid1 = k
                        break     
            cur_x[cur_pid].append({'pid': pid1, 'content': cur_context, })
            cur_y[cur_pid].append(0)
    
    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))

    with open(join(data_dir2, "bib_context_train_new_67.json"), "w", encoding="utf-8") as f:
        json.dump(x_train, f, indent=4)

    
    with open(join(data_dir2, "bib_context_valid_new_67.json"), "w", encoding="utf-8") as f:
        json.dump(x_valid, f, indent=4)

    
    with open(join(data_dir2, "bib_context_train_label_new_67.json"), "w", encoding="utf-8") as f:
        json.dump(y_train, f, indent=4)

    
    with open(join(data_dir2, "bib_context_valid_label_new_67.json"), "w", encoding="utf-8") as f:
        json.dump(y_valid, f, indent=4)

def evaluate(model, dataloader, device, criterion):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

            # logits = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, oag_token=oag_token, oag_token_paper=oag_token_paper)
    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, input_mask, segment_ids, label_ids, oag_token, oag_token_paper = batch

        with torch.no_grad():
            r = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, oag_token=oag_token, oag_token_paper=oag_token_paper)
            # r = model(oag_token=oag_token, oag_token_paper=oag_token_paper, device=device)
            # tmp_eval_loss = r[0]
            # logits = r[1]
            logits = r
            # print("logits", logits)
            tmp_eval_loss = criterion(logits, label_ids)

        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels


def train(year=2023, model_name="scibert", save_path='kddcup'):
    print("model name", model_name)
    train_texts = []
    dev_texts = []
    train_labels = []
    dev_labels = []
    data_year_dir = join(settings.DATA_TRACE_DIR, "PST", 'OAG')
    print("data_year_dir", data_year_dir)
    
    with open(join(data_year_dir, "bib_context_train_new_67.json"), "r", encoding="utf-8") as f:
        train_texts = json.load(f)
    with open(join(data_year_dir, "bib_context_valid_new_67.json"), "r", encoding="utf-8") as f:
        dev_texts = json.load(f)
    with open(join(data_year_dir, "bib_context_train_label_new_67.json"), "r", encoding="utf-8") as f:
        train_labels = json.load(f)
    with open(join(data_year_dir, "bib_context_valid_label_new_67.json"), "r", encoding="utf-8") as f:
        dev_labels = json.load(f)

    print("Train size:", sum(len(k) for k in train_texts.values()))
    print("Dev size:", sum(len(k) for k in dev_texts.values()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_label = [a for k in train_labels.values() for a in k]
    class_weight = len(train_label) / (2 * np.bincount(train_label)) # np.bincount用于统计类别标签的出现次数
    class_weight = torch.Tensor(class_weight).to(device)
    print("Class weight:", class_weight)

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

    # model2 = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)

    model = BertModel.from_pretrained(BERT_MODEL)
    _, model2 = oagbert("oagbert-v2-sim")
    model = Model(model, model2)
    # model = Model_OnlySim(model2)
        
    model.to(device)
    # model2.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    train_features = convert_examples_to_inputs(train_texts, train_labels, MAX_SEQ_LENGTH, tokenizer, verbose=0)
    dev_features = convert_examples_to_inputs(dev_texts, dev_labels, MAX_SEQ_LENGTH, tokenizer)

    BATCH_SIZE = 8
    train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)
    dev_dataloader = get_data_loader(dev_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 100
    # NUM_TRAIN_EPOCHS = 20
    # LEARNING_RATE = 5e-5
    LEARNING_RATE = 1e-6
    WARMUP_PROPORTION = 0.1
    MAX_GRAD_NORM = 5

    num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    OUTPUT_DIR = join(settings.OUT_DIR, save_path, model_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_FILE_NAME = "pytorch_model.bin"
    PATIENCE = 5

    loss_history = []
    no_improvement = 0
    for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            # print(batch, type(batch))
            # batch = tuple(t.to(device) for t in batch)
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, input_mask, segment_ids, label_ids, oag_token, oag_token_paper = batch # torch.Size([8, 512]) * 3, torch.Size([8])

            # outputs2 组成： loss, logits, hidden_states=None, attentions=None
            # tensor(0.6957, device='cuda:0', grad_fn=<NllLossBackward0>)   torch.Size([8, 2])
            # outputs2 = model2(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)

            # outputs 组成：  last_hidden_state, pooler_output, hidden_states=None, past_key_values=None, attentions=None, cross_attentions
            # torch.Size([8, 512, 768])   torch.Size([8, 768])
            # outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

            logits = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, oag_token=oag_token, oag_token_paper=oag_token_paper)
            # logits = model(oag_token=oag_token, oag_token_paper=oag_token_paper, device=device) # model_nolysim
            # loss = outputs[0]
            # logits = outputs[1]

            loss = criterion(logits, label_ids)

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
        dev_loss, _, _ = evaluate(model, dev_dataloader, device, criterion)
        
        print("Loss history:", loss_history)
        print("Dev loss:", dev_loss)
        
        if len(loss_history) == 0 or dev_loss < min(loss_history):
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            no_improvement += 1
        
        if no_improvement >= PATIENCE or dev_loss < 0.51: 
            print("No improvement on development set. Finish training.")
            print(f'min loss = {min(loss_history)}')
            break
            
        loss_history.append(dev_loss)


def eval_test_papers_bert(year=2023, model_name="scibert"):
    print("model name", model_name)
    data_year_dir = join(settings.DATA_TRACE_DIR, str(year))
    papers_test = utils.load_json(data_year_dir, "paper_source_trace_test.json")
    pids_test = {p["_id"] for p in papers_test}

    in_dir = join(settings.DATA_TRACE_DIR, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        cur_pid = f.split(".")[0]
        if f.endswith(".xml") and cur_pid in pids_test:
            files.append(f)

    truths = papers_test
    pid_to_source_titles = dd(list)
    for paper in tqdm(truths):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    # model.load_state_dict(torch.load(join(settings.OUT_DIR, model_name, "pytorch_model.bin")))
    # model.load_state_dict(torch.load(join(settings.OUT_DIR, "bert", "pytorch_model.bin")))
    model.to(device)
    model.eval()

    BATCH_SIZE = 16
    metrics = []
    f_idx = 0

    xml_dir = join(settings.DATA_TRACE_DIR, "paper-xml")

    for paper in tqdm(papers_test):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".tei.xml")
        f = open(file, encoding='utf-8')

        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue

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
        bib_sorted = sorted(bib_to_contexts.keys())

        for bib in bib_sorted:
            cur_bib_idx = int(bib[1:])
            if cur_bib_idx + 1 > n_refs:
                n_refs = cur_bib_idx + 1
        
        y_true = [0] * n_refs
        y_score = [0] * n_refs

        flag = False
        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    b_idx = int(bid[1:])
                    y_true[b_idx] = 1
        
        if not flag:
            continue

        contexts_sorted = [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]

        test_features = convert_examples_to_inputs(contexts_sorted, y_score, MAX_SEQ_LENGTH, tokenizer)
        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                r = model(input_ids, attention_mask=input_mask,
                                            token_type_ids=segment_ids, labels=label_ids)
                tmp_eval_loss = r[0]
                logits = r[1]

            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)
        
        try:
            for ii in range(len(predicted_scores)):
                bib_idx = int(bib_sorted[ii][1:])
                # print("bib_idx", bib_idx)
                y_score[bib_idx] = predicted_scores[ii]
        except IndexError as e:
            metrics.append(0)
            continue
        
        cur_map = average_precision_score(y_true, y_score)
        metrics.append(cur_map)
        f_idx += 1
        if f_idx % 20 == 0:
            print("map until now", np.mean(metrics), len(metrics), cur_map)

    print("bert average map", np.mean(metrics), len(metrics))


def gen_kddcup_valid_submission_bert(model_name="scibert", save_path='kddcup'):
    print("model name", model_name)
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    data_dir2 = join(data_dir, 'OAG')
    papers = utils.load_json(data_dir2, "paper_source_trace_valid_wo_ans.json")
    
    with open('data/PST/id_to_title.json', 'r', encoding='utf-8') as f:
        pid_to_title = json.load(f)
    with open('data/data/valid_pid_to_bid.json', 'r', encoding='utf-8') as f:
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

    sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    # model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    # model.load_state_dict(torch.load(join(settings.OUT_DIR, "kddcup", model_name, "pytorch_model.bin"))) # 加载保存在out上的模型

    model = BertModel.from_pretrained(BERT_MODEL)
    _, model2 = oagbert("oagbert-v2-sim")
    model = Model(model, model2)
    # model = Model_OnlySim(model2)
    model.load_state_dict(torch.load(join(settings.OUT_DIR, save_path, model_name, "pytorch_model.bin")))

    model.to(device)
    model.eval()

    BATCH_SIZE = 16
    # metrics = []
    # f_idx = 0

    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    # papers = papers[350:]
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
    
    utils.dump_json(sub_dict, join(settings.OUT_DIR, save_path, model_name), "valid_submission_scibert.json")


if __name__ == "__main__":
    # prepare_oag_bert_input() # generate data
    
    save_path = 'kdd-sim-new-23'
    save_path = 'kdd-onlysim'
    model_name = "specter2"
    train(model_name=model_name, save_path=model_name) # 2
    # eval_test_papers_bert(model_name="scibert")

    # gen_kddcup_valid_submission_bert(model_name=model_name, save_path=model_name) # 3

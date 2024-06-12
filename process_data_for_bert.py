
import settings
import numpy as np
import random
seed = 42
np.random.seed(seed)
random.seed(seed)
import time
import utils
from os.path import join
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from tqdm import tqdm
from lxml import etree
import re
import csv
import json

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

class Data_Porcesser(object):
    def __init__(self, paper_id, ref_id, paper_info_more):
        self.paper_id = paper_id
        self.ref_id = ref_id
        self.query_result = paper_info_more.get(ref_id, {})
        self.paper_result = paper_info_more.get(paper_id, {})
        
        # 通过xml获取tree和listBibl
        try:
            path = f'data/PST/paper-xml/{paper_id}.xml'
            self.tree = etree.parse(path)
            root = self.tree.getroot()
            listBibl = root.xpath("//*[local-name()='listBibl']")[0]
            self.biblStruct = listBibl.getchildren()
            self.num_ref = len(self.biblStruct)
        except OSError:
            self.tree = None
            self.num_ref = 0
            print('not exits xml ' + paper_id)
        
        self.feature = []
        reference_place_list = self.get_referenced_place_num()
        if len(reference_place_list) == 0:
            self.feature = []
            return # TODO： 外面需要确认feature非空  如果返回长度为0说明文章标题在xml的reference中没有找到自己的序号，这里暂时先不管它。


        self.feature.append(self.get_referenced_num())
        self.feature.append(self.get_common_authors())
        self.feature.append(self.get_reciprocal_of_reference_num())
        self.feature.append(self.key_words())
        self.feature += reference_place_list

    # ONE 被引用次数
    def get_referenced_num(self):
        return self.query_result.get('n_citation', 0)
    # TWO,SIX,EIGHT 引用位置, 是否出现在图表中, 引用次数/引用总数
    # 0 abstract
    # 1 introduction
    # 2 related work
    # 3 method
    # 4 graph and figure
    # 5 result
    # 6 others
    def get_referenced_place_num(self):
        title = self.query_result.get("title", "")
        
        if self.tree is None:
            # TODO: 存在问题
            return []
            # return [0 * 8]

        paper_number = -1
        for i, item in enumerate(self.biblStruct):
            this_test = item.xpath('.//*[local-name()="title"]')
            this_text = this_test[0].text
            if this_text is None:
                try:
                    this_text = this_test[1].text
                except IndexError:
                    this_text = ''
            try:
                score = fuzz.partial_ratio(title, this_text)
            except ValueError:
                score = 0
            if score >= 80:
                paper_number = i + 1
                break
            
        place_num = [0 for i in range(8)]
        self.paper_number = paper_number
        if paper_number == -1:
            return place_num
        # 使用序号，在xml文件中检索位置
        nodes = self.tree.xpath(f"//*[contains(text(), '[{paper_number}]')]")
        reference_times = len(nodes)
        
        for item in nodes:
            found_text = ''
            this_node = item
            while found_text == '':
                this_node = this_node.getparent()
                if this_node is None:
                    break
                if this_node.xpath("local-name()") == 'figure':
                    place_num[4] = 1
                it_children = this_node.iterchildren()
                for jtem in it_children:
                    node = this_node
                    if jtem.xpath("local-name()") == 'head':
                        found_text = node.text
                        n_num = jtem.attrib.get('n')
                        node = this_node
                        if n_num is None:
                            break
                        while not n_num.isdigit():
                            node = node.getprevious()
                            if node is None:
                                break
                            node_children = node.iterchildren()
                            for ktem in node_children:
                                if ktem.xpath("local-name()") == 'head':
                                    n = ktem.attrib.get('n')
                                    if n is not None and n.isdigit():
                                        n_num = ktem.attrib.get('n')
                                        found_text = ktem.text
                                        break
                                    break
            
            if this_node is None or found_text == '':
                place_num[6] = 1
                continue
            if found_text is not None:
                found_text = found_text.lower()
            if fuzz.partial_ratio('abstract', found_text) >= 60:
                place_num[0] = 1
            elif fuzz.partial_ratio('introduction', found_text) >= 60:
                place_num[1] = 1
            elif fuzz.partial_ratio('related work', found_text) >= 60:
                place_num[2] = 1
            elif fuzz.partial_ratio('method', found_text) >= 60:
                place_num[3] = 1
            elif fuzz.partial_ratio('result', found_text) >= 60 or fuzz.partial_ratio('experiment', found_text) >= 60:
                place_num[5] = 1
            else:
                place_num[6] = 1
        pattern = re.compile(r'[\d+]')
        nodes = self.tree.xpath("//*[re:match(text(), $pattern)]",
                                namespaces={"re": "http://exslt.org/regular-expressions"},
                                pattern=pattern.pattern)
        total_ref_num = len(nodes)
        if not total_ref_num == 0:
            place_num[7] = reference_times / total_ref_num
        return place_num
    
    def get_common_authors(self):
        ref_authors_set = set(self.query_result.get('authors', []))
        paper_authors_set = set(self.paper_result.get('authors',[]))
        # TODO: 检查其中有1
        if not len(paper_authors_set & ref_authors_set) == 0:
            return 1
        else:
            return 0
        
    def get_reciprocal_of_reference_num(self):
        if self.num_ref == 0:
            return 0
        else:
            return 1 / self.num_ref
    def key_words(self):
        if self.paper_number == -1:
            return 0
        pattern = re.compile(r'[\d+]')
        nodes = self.tree.xpath("//*[re:match(text(), $pattern)]",
                                namespaces={"re": "http://exslt.org/regular-expressions"},
                                pattern=pattern.pattern)
        # TODO: 增加key words
        key_words_list = ['motivated by', 'inspired by']
        for item in nodes:
            if item.xpath('local-name()') == 'ref':
                node_text = item.getparent().text
            else:
                node_text = item.text
            if node_text is None:
                return 0
            node_text = node_text.lower()
            for jtem in key_words_list:
                pattern = re.compile(fr"{jtem}")
                match = pattern.search(node_text)
                if match is not None:
                    return 1
        return 0
        
    
def process_data_for_mybert():
    ####################################################
    pid_to_bid_dict = {}
    ####################################################
    x_train, y_train = [], []
    x_valid, y_valid = [], []
    features_train, features_valid = [], []
    oagbert_data_train, oagbert_data_valid = [], []
    
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    # papers = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    papers = sorted(papers, key=lambda x: x["_id"])
    # TODO: 加入生成代码到该文件  可以往json中写入更多信息
    paper_info_more = utils.load_json(data_dir, "paper_info_hit_from_all.json")
    n_paper = len(papers)
    
    # TODO: 调节训练集和验证集的比例
    n_train = int(n_paper *  2 / 3)
    # n_train = int(n_paper *  99 / 100)
    papers_trian = papers[:n_train]
    papers_valid = papers[n_train:]
    
    pids_train = {p["_id"] for p in papers_trian}
    pids_valid = {p["_id"] for p in papers_valid}

    xml_dir = join(data_dir, "paper-xml")
    
    for paper in tqdm(papers):
        ####################################################
        pid_to_bid_dict[paper["_id"]] = {}
        ####################################################
        # 对应每篇paper 存在 
        #   pid： paper id唯一标识 
        #   bid: 被引用id [1],[2],.......
        #   title: 论文标题小写
        # 需要对三者建立联系 三者任意为空的数据不予采用   
        cur_pid = paper["_id"]
        # TODO: delete the data not in paper_info_more
        if cur_pid not in paper_info_more:
            continue
        ref_ids = paper.get("references", [])
        refs_trace = paper.get("refs_trace", [])
        cur_title_to_pid = {}
        for ref_id in ref_ids:
            if ref_id in paper_info_more:
                cur_title_to_pid[paper_info_more[ref_id]["title"].lower()] = ref_id # {title: pid, title: pid,...}
        
        # pass掉没有正样本的
        if len(refs_trace) == 0:
            continue
        
        f = open(join(xml_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()
        
        
        # 对齐xml中的论文和正负样本
        references = bs.find_all("biblStruct")
        bid_to_title = {} # 当前论文的bid:title
        n_refs = 0
        cur_title_to_b_idx = {} # 当前论文的title：bid
        for ref in references:
            # 排除前面的无用biblStruct标签
            # TODO: 有点title信息存在monogr里面
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            cur_title_to_b_idx[ref.analytic.title.text.lower()] = bid
            b_idx = int(bid[1:]) + 1
            # cur_title_to_b_idx[ref.analytic.title.text.lower()] = b_idx - 1
            if b_idx > n_refs:
                n_refs = b_idx
        
        
        # flag用于表示xml中的参考文献是否出现于refs_trace
        flag = False
        
        cur_pos_bib = {} # 存储当前pos样本的pid: bid
        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for ref_trace in refs_trace:
                ref_id = ref_trace["_id"]
                #TODO: delte positive smaple not in paper_info_more
                if ref_id not in paper_info_more:
                    continue 
                ref_title = ref_trace["title"].lower()
                if fuzz.ratio(cur_ref_title, ref_title) >= 80:
                    flag = True
                    cur_pos_bib[ref_id] = bid

        cur_neg_bib = {} # 存储当前neg样本的pid: bid
        for r_idx, ref_id in enumerate(ref_ids):
            # 如果是正样本 则丢弃
            if ref_id in cur_pos_bib:
                continue
            
            if ref_id not in paper_info_more:
                continue
            
            cur_title = paper_info_more[ref_id].get("title", "").lower()
            if len(cur_title) == 0:
                continue
            #TODO: 读特征 如果特征为null 也continue
            cur_bid = None
            for b_title in cur_title_to_b_idx:
                if fuzz.ratio(cur_title, b_title) >= 80:
                    cur_bid = cur_title_to_b_idx[b_title]
                    cur_neg_bib[ref_id] = cur_bid
                    break
            if cur_bid is None:
                continue
            
        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue
        ###########################################################
        pid_to_bid_dict[paper["_id"]] = {**cur_pos_bib, **cur_neg_bib}
        ###########################################################
        n_pos = len(cur_pos_bib)
        # n_neg = n_pos * 10
        
        n_neg = min(len(cur_neg_bib), n_pos * 10)
        # TODO: 调节replace参数
        # cur_neg_bib_sample_pid_list = np.random.choice(list(cur_neg_bib), n_neg, replace=True)
        cur_neg_bib_sample_pid_list = np.random.choice(list(cur_neg_bib), n_neg, replace=False)
        cur_neg_bib_sample_bid_list = [cur_neg_bib[pid] for pid in cur_neg_bib_sample_pid_list]
        cur_pos_bib_sample_pid_list = list(cur_pos_bib.keys())
        cur_pos_bib_sample_bid_list = list(cur_pos_bib.values())
        
        
         # TODO: 调节获取的上下文长度
        bib_to_contexts = utils.find_bib_context(xml)
        if cur_pid in pids_train:
            cur_x = x_train
            cur_y = y_train
            cur_features = features_train
            cur_oagbert_data = oagbert_data_train
        elif cur_pid in pids_valid:
            cur_x = x_valid
            cur_y = y_valid
            cur_features = features_valid
            cur_oagbert_data = oagbert_data_valid
        else:
            continue
        
        for bid, pid in zip(cur_pos_bib_sample_bid_list, cur_pos_bib_sample_pid_list):
            processed_data = Data_Porcesser(cur_pid, pid, paper_info_more)
            feature = processed_data.feature
            if len(feature) == 0:
                print("null feature")
                continue
            cur_features.append(feature)
            cur_context = " ".join(bib_to_contexts[bid])
            cur_oagbert_data.append([paper_info_more[cur_pid], paper_info_more[pid]])
            cur_x.append(cur_context)
            cur_y.append(1)
        for bid, pid in zip(cur_neg_bib_sample_bid_list, cur_neg_bib_sample_pid_list):
            processed_data = Data_Porcesser(cur_pid, pid, paper_info_more)
            feature = processed_data.feature
            if len(feature) == 0:
                print("null feature")
                continue
            cur_features.append(feature)
            cur_context = " ".join(bib_to_contexts[bid])
            cur_oagbert_data.append([paper_info_more[cur_pid], paper_info_more[pid]])
            cur_x.append(cur_context)
            cur_y.append(0)
    
    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))

    with open(join(data_dir, "bib_context_train.txt"), "w", encoding="utf-8") as f: 
        for line in x_train:
            f.write(line + "\n")
    
    with open(join(data_dir, "bib_context_valid.txt"), "w", encoding="utf-8") as f:
        for line in x_valid:
            f.write(line + "\n")
    
    with open(join(data_dir, "bib_context_train_label.txt"), "w", encoding="utf-8") as f:
        for line in y_train:
            f.write(str(line) + "\n")
    
    with open(join(data_dir, "bib_context_valid_label.txt"), "w", encoding="utf-8") as f:
        for line in y_valid:
            f.write(str(line) + "\n")
    
    with open(join(data_dir, "bib_context_train_features.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(features_train)

    with open(join(data_dir, "bib_context_valid_features.csv"), 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(features_valid)
        
    utils.dump_json(oagbert_data_train ,data_dir,  "bib_context_train_oagbert_data.json")
    utils.dump_json(oagbert_data_valid, data_dir, "bib_context_valid_oagbert_data.json")
        
    ####################################################
    utils.dump_json(pid_to_bid_dict,data_dir,"train_pid_to_bid.json")
    ####################################################
    
    
    


def extract_paper_info_from_oag():
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers_train = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    papers_valid = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    papers_test = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    clicked_num_oag, clicked_num_dblp, dblp_hit = 0, 0, 0
    all_num = 0
    hit_list = []
    paper_dict_hit = dd(dict)
    
    for paper in tqdm(papers_train + papers_valid + papers_test):
        cur_pid = paper["_id"]
        ref_ids = paper.get("references", [])
        pids = [cur_pid] + ref_ids
        all_num += len(pids)
        
    print(all_num)
    
    for i in range(1,15):
        temp_hit = 0
        paper_dict_open = {}
        dblp_fname = "v3.1_oag_publication_" + str(i) + ".json"
        with open(join(data_dir, "OAG", dblp_fname), "r", encoding="utf-8") as myFile:
            for i, line in enumerate(myFile):
                if len(line) <= 2:
                    continue
                if i % 10000 == 0: 
                    logger.info("reading papers %d", i)
                paper_tmp = json.loads(line.strip())
                paper_dict_open[paper_tmp["id"]] = paper_tmp
        logger.info(dblp_fname + "ok")

        for paper in tqdm(papers_train + papers_valid + papers_test):
            cur_pid = paper["_id"]
            ref_ids = paper.get("references", [])
            pids = [cur_pid] + ref_ids
            for pid in pids:
                if pid not in paper_dict_open:
                    continue
                clicked_num_oag += 1
                temp_hit += 1
                cur_paper_info = paper_dict_open[pid]
                cur_authors = [a.get("name", "") for a in cur_paper_info.get("authors", [])]
                n_citation = cur_paper_info.get("n_citation", 0)
                title = cur_paper_info.get("title", "")
                abstract = cur_paper_info.get("abstract", "")
                venue = cur_paper_info.get("venue", "")
                keywords = cur_paper_info.get("keywords", [])
                year = cur_paper_info.get("year", -1)
            
                paper_dict_hit[pid] = {"authors": cur_authors, "n_citation": n_citation, "title": title, "abstract": abstract, "venue": venue, "keywords": keywords, "year": year}
                
        hit_list.append(temp_hit)
                
             
    dblp_fname = "DBLP-Citation-network-V15.1.json"
    paper_dict_open = {}
    temp_hit = 0
    with open(join(data_dir, dblp_fname), "r", encoding="utf-8") as myFile:
        for i, line in enumerate(myFile):
            if len(line) <= 2:
                continue
            if i % 10000 == 0: 
                logger.info("reading papers %d", i)
            paper_tmp = json.loads(line.strip())
            paper_dict_open[paper_tmp["id"]] = paper_tmp
    for paper in tqdm(papers_train + papers_valid + papers_test):
        cur_pid = paper["_id"]
        ref_ids = paper.get("references", [])
        pids = [cur_pid] + ref_ids
        for pid in pids:
            if pid not in paper_dict_open:
                continue
            if pid in paper_dict_hit:
                dblp_hit += 1
                continue
            clicked_num_dblp += 1
            temp_hit += 1
            cur_paper_info = paper_dict_open[pid]
            cur_authors = [a.get("name", "") for a in cur_paper_info.get("authors", [])]
            n_citation = cur_paper_info.get("n_citation", 0)
            title = cur_paper_info.get("title", "")
            abstract = cur_paper_info.get("abstract", "")
            venue = cur_paper_info.get("venue", "")
            keywords = cur_paper_info.get("keywords", [])
            year = cur_paper_info.get("year", -1)
            
            paper_dict_hit[pid] = {"authors": cur_authors, "n_citation": n_citation, "title": title, "abstract": abstract, "venue": venue, "keywords": keywords, "year": year}
            
    hit_list.append(temp_hit)
    # number of papers before filtering: 21002
    print("number of papers after filtering", len(paper_dict_hit)) # 16170
    print(hit_list)
    utils.dump_json(paper_dict_hit, data_dir, "paper_info_hit_from_all.json")
    print(dblp_hit, clicked_num_dblp, clicked_num_oag, all_num) # 27790 682 29708 40399
    logger.info(str(dblp_hit) + " " + str(clicked_num_dblp) + " " + str(clicked_num_oag) + " " + str(all_num))


def process_feature_and_save():
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    train_papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    valid_papers = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    paper_info_more = utils.load_json(data_dir, "paper_info_hit_from_all.json")
    
    train_num, valid_num = 0, 0
    train_not_hit, valid_not_hit = 0, 0
    train_feature = {}
    for paper in tqdm(train_papers):
        cur_pid = paper["_id"]
        train_feature[cur_pid] = {}
        ref_ids = paper.get("references", [])
        for ref_id in ref_ids:
            processed_data = Data_Porcesser(cur_pid, ref_id, paper_info_more)
            feature = processed_data.feature
            train_num += 1
            if len(feature) == 0:
                train_not_hit += 1
                continue
            train_feature[cur_pid][ref_id] = feature
    
    utils.dump_json(train_feature, data_dir, "train_rf_features.json")
    
    valid_feature = {}
    
    for paper in tqdm(valid_papers):
        cur_pid = paper["_id"]
        valid_feature[cur_pid] = {}
        ref_ids = paper.get("references", [])
        for ref_id in ref_ids:
            processed_data = Data_Porcesser(cur_pid, ref_id, paper_info_more)
            feature = processed_data.feature
            valid_num += 1
            if len(feature) == 0:
                valid_not_hit += 1
                continue
            valid_feature[cur_pid][ref_id] = feature
    
    utils.dump_json(valid_feature, data_dir, "valid_rf_features.json")
    print(train_num, train_not_hit, valid_num, valid_not_hit)


def process_pid_to_bid():
    pid_to_bid_dict = {}
    
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    # papers = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    papers = sorted(papers, key=lambda x: x["_id"])
    # TODO: 加入生成代码到该文件  可以往json中写入更多信息
    paper_info_more = utils.load_json(data_dir, "paper_info_hit_from_all.json")
    n_paper = len(papers)
    
    xml_dir = join(data_dir, "paper-xml")
    
    for paper in tqdm(papers):
        ####################################################
        pid_to_bid_dict[paper["_id"]] = {}
        ####################################################
        # 对应每篇paper 存在 
        #   pid： paper id唯一标识 
        #   bid: 被引用id [1],[2],.......
        #   title: 论文标题小写
        # 需要对三者建立联系 三者任意为空的数据不予采用   
        cur_pid = paper["_id"]
        # TODO: delete the data not in paper_info_more
        # if cur_pid not in paper_info_more:
        #     continue
        ref_ids = paper.get("references", [])
        refs_trace = paper.get("refs_trace", [])
        cur_title_to_pid = {}
        for ref_id in ref_ids:
            if ref_id in paper_info_more:
                cur_title_to_pid[paper_info_more[ref_id]["title"].lower()] = ref_id # {title: pid, title: pid,...}
        
        # pass掉没有正样本的
        # if len(refs_trace) == 0:
        #    continue
        
        f = open(join(xml_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()
        
        
        # 对齐xml中的论文和正负样本
        references = bs.find_all("biblStruct")
        bid_to_title = {} # 当前论文的bid:title
        n_refs = 0
        cur_title_to_b_idx = {} # 当前论文的title：bid
        for ref in references:
            # 排除前面的无用biblStruct标签
            # TODO: 有点title信息存在monogr里面
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            cur_title_to_b_idx[ref.analytic.title.text.lower()] = bid
            b_idx = int(bid[1:]) + 1
            # cur_title_to_b_idx[ref.analytic.title.text.lower()] = b_idx - 1
            if b_idx > n_refs:
                n_refs = b_idx
        
        
        # flag用于表示xml中的参考文献是否出现于refs_trace
        
        cur_pos_bib = {} # 存储当前pos样本的pid: bid
        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for ref_trace in refs_trace:
                ref_id = ref_trace["_id"]
                ref_title = ref_trace["title"].lower()
                if fuzz.ratio(cur_ref_title, ref_title) >= 80:
                    flag = True
                    cur_pos_bib[ref_id] = bid

        cur_neg_bib = {} # 存储当前neg样本的pid: bid
        for r_idx, ref_id in enumerate(ref_ids):
            # 如果是正样本 则丢弃
            # if ref_id in cur_pos_bib:
            #     continue
            
            if ref_id not in paper_info_more:
                continue
            
            cur_title = paper_info_more[ref_id].get("title", "").lower()
            if len(cur_title) == 0:
                continue
            #TODO: 读特征 如果特征为null 也continue
            cur_bid = None
            for b_title in cur_title_to_b_idx:
                if fuzz.ratio(cur_title, b_title) >= 80:
                    cur_bid = cur_title_to_b_idx[b_title]
                    cur_neg_bib[ref_id] = cur_bid
                    break
            if cur_bid is None:
                continue
            
        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue
        ###########################################################
        pid_to_bid_dict[paper["_id"]] = {**cur_pos_bib, **cur_neg_bib}
        # pid_to_bid_dict[paper["_id"]] = {**cur_neg_bib}
        ###########################################################
    utils.dump_json(pid_to_bid_dict,data_dir,"train_pid_to_bid.json")

def process_pid_to_bid_test():
    pid_to_bid_dict = {}
    
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    papers = sorted(papers, key=lambda x: x["_id"])
    # TODO: 加入生成代码到该文件  可以往json中写入更多信息
    paper_info_more = utils.load_json(data_dir, "paper_info_hit_from_all.json")
    n_paper = len(papers)
    
    xml_dir = join(data_dir, "paper-xml")
    
    for paper in tqdm(papers):
        ####################################################
        pid_to_bid_dict[paper["_id"]] = {}
        ####################################################
        # 对应每篇paper 存在 
        #   pid： paper id唯一标识 
        #   bid: 被引用id [1],[2],.......
        #   title: 论文标题小写
        # 需要对三者建立联系 三者任意为空的数据不予采用   
        cur_pid = paper["_id"]
        # TODO: delete the data not in paper_info_more
        # if cur_pid not in paper_info_more:
        #     continue
        ref_ids = paper.get("references", [])
        cur_title_to_pid = {}
        for ref_id in ref_ids:
            if ref_id in paper_info_more:
                cur_title_to_pid[paper_info_more[ref_id]["title"].lower()] = ref_id # {title: pid, title: pid,...}
        
        f = open(join(xml_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()
        
        
        # 对齐xml中的论文和正负样本
        references = bs.find_all("biblStruct")
        bid_to_title = {} # 当前论文的bid:title
        n_refs = 0
        cur_title_to_b_idx = {} # 当前论文的title：bid
        for ref in references:
            # 排除前面的无用biblStruct标签
            # TODO: 有点title信息存在monogr里面
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            cur_title_to_b_idx[ref.analytic.title.text.lower()] = bid
            b_idx = int(bid[1:]) + 1
            # cur_title_to_b_idx[ref.analytic.title.text.lower()] = b_idx - 1
            if b_idx > n_refs:
                n_refs = b_idx
        
        

        cur_neg_bib = {} # 存储当前neg样本的pid: bid
        for r_idx, ref_id in enumerate(ref_ids):
            # 如果是正样本 则丢弃
            # if ref_id in cur_pos_bib:
            #     continue
            
            if ref_id not in paper_info_more:
                continue
            
            cur_title = paper_info_more[ref_id].get("title", "").lower()
            if len(cur_title) == 0:
                continue
            #TODO: 读特征 如果特征为null 也continue
            cur_bid = None
            for b_title in cur_title_to_b_idx:
                if fuzz.ratio(cur_title, b_title) >= 80:
                    cur_bid = cur_title_to_b_idx[b_title]
                    cur_neg_bib[ref_id] = cur_bid
                    break
            if cur_bid is None:
                continue
            
        if len(cur_neg_bib) == 0:
            continue
        ###########################################################
        pid_to_bid_dict[paper["_id"]] = {**cur_neg_bib}
        ###########################################################
    utils.dump_json(pid_to_bid_dict,data_dir,"test_pid_to_bid.json")



if __name__ == "__main__":
    # extract_paper_info_from_oag()
    # process_feature_and_save()
    # process_data_for_mybert()
    # process_pid_to_bid()
    # process_pid_to_bid_test()
    pass

    # data_dir = join(settings.DATA_TRACE_DIR, "PST")
    # papers_train = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    # papers_valid = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    # all_papers = set()
    # for paper in tqdm(papers_train + papers_valid):
    #     cur_pid = paper["_id"]
    #     ref_ids = paper.get("references", [])
    #     pids = [cur_pid] + ref_ids
    #     all_papers = set(pids) | all_papers
    # print(len(all_papers))

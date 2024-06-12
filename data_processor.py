import json
from collections import defaultdict as dd
from dataset import OagBertDataset
from torch.utils.data import DataLoader

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id, oag_token, oag_token_paper):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.oag_token = oag_token
        self.oag_token_paper = oag_token_paper

def capture_oag_token(pid, data_xml, data_dblp):
    oag_token = dd(str)
    oag_token['title'] = ''
    oag_token['abstract'] = ''
    oag_token['authors'] = []
    oag_token['venue'] = ''
    oag_token['concepts'] = []
    oag_token['affiliations'] = []

    # 只用title和abstract TODO
    try:
        data = data_xml.get(pid + '.xml', None)
        data_dblp = data_dblp.get(pid, None)
    except:
        return oag_token # 出现pid为空的情况 不能为None，Dataloader不支持
    if data is not None:
        oag_token['title'] = data['title']
        oag_token['abstract'] = data['abstract']
        # oag_token['abstract'] = data['contribution'] if len(data['contribution']) else data['conclusion']
    if data_dblp is not None:
        oag_token['title'] = data_dblp['title']
        # oag_token['authors'] = data_dblp['authors']
        oag_token['venue'] = data_dblp['venue']
        oag_token['abstract'] = data_dblp['abstract']

    return oag_token

def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    with open('data/PST/contribution_new.json', 'r', encoding='utf-8') as f:
        data_xml = json.load(f)
    with open('data/data/paper_info_hit_from_all.json', 'r', encoding='utf-8') as f:
        data_dblp = json.load(f)
        
    input_items = []
    for k in example_texts:
        oag_token_paper = capture_oag_token(k, data_xml, data_dblp)
        examples = zip(example_texts[k], example_labels[k])
        for (ex_index, (text1, label)) in enumerate(examples):
            # abstract=abstract, venue=venue, authors=authors, concepts=concepts, affiliations=affiliations

            pid, text = text1.values()

            # Create a list of token ids
            input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]

            # All our tokens are in the first input segment (id 0).
            segment_ids = [0] * len(input_ids)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label

            input_items.append(
                BertInputItem(text=text,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            oag_token=capture_oag_token(pid, data_xml, data_dblp),
                            oag_token_paper=oag_token_paper))
        
    return input_items


def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    data = OagBertDataset(features)
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader

import os
import torch
from torch.utils.data import DataLoader, Dataset

class OagBertDataset_Feature(Dataset):
    def __init__(self, features):
        super().__init__()
        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long) # torch.Size([7645, 512])
        self.all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        self.oag_token = [f.oag_token for f in features]
        self.oag_token_paper = [f.oag_token_paper for f in features]
        self.features = torch.tensor([f.feature for f in features], dtype=torch.float)
        self.data = features
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (self.all_input_ids[idx], self.all_input_mask[idx], self.all_segment_ids[idx], self.all_label_ids[idx], self.oag_token[idx], self.oag_token_paper[idx], self.features[idx])
    

class OagBertDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long) # torch.Size([7645, 512])
        self.all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        self.oag_token = [f.oag_token for f in features]
        self.oag_token_paper = [f.oag_token_paper for f in features]
        self.data = features
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (self.all_input_ids[idx], self.all_input_mask[idx], self.all_segment_ids[idx], self.all_label_ids[idx], self.oag_token[idx], self.oag_token_paper[idx])
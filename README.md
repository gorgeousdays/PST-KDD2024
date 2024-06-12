# PST-KDD2024

This repo is our solution for [KDDCUP2024 PST Track](https://www.biendata.xyz/competition/pst_kdd_2024/)

## Method

We obtained the metadata of the hited papers from the [DBLP](https://open.aminer.cn/open/article?id=655db2202ab17a072284bc0c) and [OAG](https://open.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf) datasets, learned the embeddings of the paper abstracts using [Spercter2](https://huggingface.co/allenai/specter2_base), and learned the embeddings of two papers using [oagbert-v2-sim](https://github.com/THUDM/OAG-BERT). We then constructed a prediction model to perform correlation analysis between the papers.


## Getting Started
### Prerequisites
* Linux
* Python 3
* NVIDIA GPU + CUDA CuDNN
### Enviroment
Install PyTorch and dependencies by:
```
pip install -r requirements.txt
```

### Data Prepare
The data processing code file is `process_data_for_bert.py`, and the processed data is located in `data/data`, which includes the following data:
* paper reference numbers corresponding to paper ids.
* metadata (including titles, abstracts, etc.) for papers in the train, valid, and test sets that match papers in the DBLP and OAG datasets.
### Train
```
python main.py
```

### Inference
Download the model from and place the model in `out/`
```
python inference.py
```

### Results on Test Set
0.38159

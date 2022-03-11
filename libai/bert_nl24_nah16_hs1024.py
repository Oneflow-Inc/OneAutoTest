from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from .common.models.bert import pretrain_model as model
from .common.models.graph import graph
from .common.train import train
from .common.optim import optim
from .common.data.bert_dataset import dataloader, tokenization

vocab_file = "/home/ylkj/dataset/bert-base-chinese-vocab.txt"
data_prefix = "/home/ylkj/dataset/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

# Bert-large model config
model.cfg.hidden_layers = 24
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 1024

train.dist.pipeline_num_layers = model.cfg.hidden_layers
train.test_micro_batch_size = 4

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.evaluation.enabled = True
train.evaluation.eval_iter = 30

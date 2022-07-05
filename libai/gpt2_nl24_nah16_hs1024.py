from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.config import LazyCall
from .common.models.gpt import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.gpt_dataset import dataloader, tokenization

from .common.models.graph import graph

vocab_file = "/path/to/gpt2-vocab.json"
merges_file = "/path/to/gpt2-merges.txt"
data_prefix = "/path/to/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merges_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix
# dataloader.train.num_workers = 4

# GPT-2 model config
model.cfg.embedding_dropout_prob = 0.1
model.cfg.attention_dropout_prob = 0.1
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 1024
model.cfg.ffn_hidden_size = 4096
#model.cfg.num_layers = 24
model.cfg.max_seq_length = 1024
#model.cfg.initializer_range = 0.006

# model.cfg.bias_dropout_fusion = True
# model.cfg.bias_gelu_fusion = True
# model.cfg.scale_mask_softmax_fusion = True


train.input_placement_device = "cpu"


for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length

optim.lr = 1.5e-4

#train.dist.pipeline_num_layers = model.cfg.num_layers

train.test_micro_batch_size = 4
train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.evaluation.enabled = False
train.evaluation.eval_iter = 30

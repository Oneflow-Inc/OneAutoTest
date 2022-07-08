from libai.config import LazyCall
from libai.evaluation import PPLEvaluator
from libai.config import LazyCall, get_config

from .common.models.t5 import pretrain_model as model
from .common.train import train
from .common.optim import optim
from .common.data.t5_dataset import dataloader, tokenization

# from projects.idea_t5.configs.t5_dataset import dataloader, tokenization

from .common.models.graph import graph

vocab_file = "/path/to/bert-base-chinese-vocab.txt"
data_prefix = "/path/to/loss_compara_content_sentence"

tokenization.tokenizer.vocab_file = vocab_file
dataloader.train.dataset[0].data_prefix = data_prefix
dataloader.train.dataset[0].indexed_dataset.data_prefix = data_prefix

train.eval_iter = 10

# Set all dropout to 0.
model.cfg.hidden_dropout_prob = 0.1
model.cfg.attention_probs_dropout_prob = 0.1
model.cfg.embedding_dropout_prob = 0.1
model.cfg.bias_gelu_fusion = True
model.cfg.bias_dropout_fusion = True

# Set matched model arguments
#model.cfg.hidden_layers = 24
model.cfg.hidden_size = 2304
model.cfg.intermediate_size = 3072
model.cfg.num_attention_heads = 16
model.cfg.max_position_embeddings = 512


train.warmup_ratio = 0.01
train.test_micro_batch_size = 4


#train.dist.pipeline_num_layers = 2 * model.cfg.hidden_layers

train.input_placement_device = "cpu"

# Set a constant lr scheduler after warmup
optim.lr = 0.0001

train.evaluation.evaluator = LazyCall(PPLEvaluator)()

train.evaluation.enabled = False
train.evaluation.eval_iter = 30

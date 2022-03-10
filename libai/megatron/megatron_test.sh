
# T5模型

# 1n1g batch size 16  12654MiB

bash examples/megatron_args_pretrain_t5.sh 1 1 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 32 32

pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_t5.sh 1 1 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 24 24

pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_t5.sh 1 1 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 16 16

# 1n8g batch size 64  12654MiB
bash examples/megatron_args_pretrain_t5.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 16 128

pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_t5.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 24 192

pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_t5.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 24 192

pkill python
pkill python
pkill python


# 1n8g batch size 64  12654MiB mp2
bash examples/megatron_args_pretrain_t5.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 2 1 16 128


pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_t5.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 2 1 32 256

pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_t5.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 2 1 32 128


pkill python
pkill python
pkill python



# BERT模型

# 1n1g batch size 4 g 8  12654MiB

bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 4 8

pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 8 8

pkill python
pkill python
pkill python



# 1n8g batch size 4  12654MiB
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 4 32

pkill python
pkill python
pkill python

# 1n8g batch size 4  12654MiB
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 1 1 8 64

pkill python
pkill python
pkill python


# 1n8g batch size 64  12654MiB mp2
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 2 2 8 64

bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/bert-base-chinese-vocab.txt 2 2 16 128



pkill python
pkill python
pkill python



# GPT模型

# 1n1g batch size 4 g 8  12654MiB

bash examples/megatron_args_pretrain_gpt2.sh 1 1 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 1 1 4 8



pkill python
pkill python
pkill python


bash examples/megatron_args_pretrain_gpt2.sh 1 1 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 1 1 8 8


pkill python
pkill python
pkill python


bash examples/megatron_args_pretrain_gpt2.sh 1 1 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 1 1 16 16

pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 1 1 16 128



pkill python
pkill python
pkill python

bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 1 1 8 64



pkill python
pkill python
pkill python


bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 2 2 16 32

pkill python
pkill python
pkill python


bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 2 2 16 128

pkill python
pkill python
pkill python


bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 2 2 32 64

pkill python
pkill python
pkill python


bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 /home/ylkj/dataset/loss_compara_content_sentence /home/ylkj/dataset/gpt2-vocab.json /home/ylkj/dataset/gpt2-merges.txt 2 2 32 128

pkill python
pkill python
pkill python



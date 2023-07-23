# 

###  Project Overview
seq2seq model that translates German into English. The dataset used [Multi30k](https://www.statmt.org/wmt16/multimodal-task.html#task1).

Metric used for evaluation: Bleu Score

This implementation focuses on the 2 approaches: seq2seq and tranformers architecture.

### seq2seq
- Encoder&Decoder: RNN(LSTM)


### seq2seq transformers
The transformer architecture was first introducted in "Attention is all you need" and was used as a sequence to sequence model for machine translation tasks. This network consists of three parts:
 - Embedding layer: Layer that converts tensors of input indices into corresponding tensor of input embeddings. These embedding are further augmented with positional encodings to provide position information of input tokens to the model.
 - Transformer: Network consisting of an encoder and decoder, each consisting of stacked self-attention and point-wise fully connected layers.
 - Linear layer: Output of transformer is passed through a linear layer that gives un-normalized probabilities (logits) for each token in the target language.


### Setup
torchtext=0.6

Spacy languages used:
- python -m spacy download en 
- python -m spacy download de
 






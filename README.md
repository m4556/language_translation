# 

###  Project Overview
seq2seq model that translates German into English. 

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

#### 1. Download Spacy Languages:
python -m spacy download en  
python -m spacy download de

#### 2. Conda environment:
If you're using a Conda environment, create and activate the environment using the following commands:  
conda create --n lang_translation_env  
conda activate lang_translation_env

#### 3. Install required Libraries
Install the required libraries using the following command:  
pip install -r requirements.txt

#### 4. Run file
Position to desired methods folder, and run file to train the model:
python run train.py 



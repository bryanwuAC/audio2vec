# Audio2Vec: An exploration of embedding methods to represent an arbitrary segment of audio in a fixed length vector

## Abstract
Recent years, with the popularity of large digital music
libraries, the ability to retrieve similar or even
duplicate songs has become more important.
Therefore, it is important to embed music into feature
vector so that it is discriminative between different
songs, but still robust to songs with noise and
themselves. As text and audio are both well studied
sequential data, we implement two unsupervised audio embedding method inspired by text
embedding. We first conduct (MFCC and Chroma) feature extraction on raw waveform,
then we embed each sequence of feature vectors into a fixed length vector using
Bag-of-Audio-Words Model and Sequence-to-sequence Autoencoder. The
embedding models are evaluated by retrieval and recommendation tasks and present good performance.

## Dataset

We used Free Music Archive (FMA) as our dataset. Related data and code can be found here: https://github.com/mdeff/fma

## Methods
### Bag of audio words
1. Feature dimension reduction: Perform chroma feature extraction and feature dimension reduction (PCA)
2. Vector quantization: Cluster all feature vectors into 1000 bins as "audio words"/
3. Histogram construction: Given a segment of audio, contruct a histogram (or a document vector) based on its feature vectors and previously generated bins.
### Sequence to sequence model
1. Encoder is a bidirectional LSTM and a fully connct layer.
2. Decoder is a unidirectional LSTM and a shared fully connect layer.
3. Train the encoder and decoder from end to end, where the encoder is encoding a segment of audio represented by a sequence of chroma feature vectors into a encoding vector, and the decoder is decoding the encoding vector into its original sequence chroma feature vectors. Use the reconstruction loss to optimize the entire model

## Paper
Algorithm details and experiment results see this paper:
https://1drv.ms/b/s!AkpHFm7pqfBnhbJyUiDSL9qEqJV9Tw

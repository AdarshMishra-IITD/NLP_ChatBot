# NLP_ChatBot
**Data:** Working on Babi Data Set from Facebook Research.Corpus contains lists of stories, questions, answers in a tuple. 

**Methodology:** Started with stop word removal, Stemming, Lemmatization and created Vocabulary. Used Tokenization, Vectorization(TfIdf,Bag of words, Word2Vec), and Padding for equal sequence length. 

**Model:** Using sequential, embedding, and dropout layers, created Input encoder M, Input encoder C and Question
encoder. Computed the match between internal state and each memory vector of encoded sequence in M by inner
product followed by a SoftMax layer. Created response vector from the memory vectors which is the sum over inputs
by encoder C, weight by probability vector from the input. Applied the LSTM layer and SoftMax activation at the end.

**Result:** Trained RNN model for 120 epochs and got the test accuracy of 91%. Deployed the model using Flask.

![This is an image]/assets/images/]

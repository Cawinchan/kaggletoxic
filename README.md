# git lfs

you must run 
`git lfs install`

and

`git lfs pull`

to get all the large data files! 

# kaggletoxic

Problem statement : Given a sentence, decide if it is toxic (several variants)


## Big Questions 

### 1. How do I turn words into vectors?



#### Solution 1 : Semantics 
-W2V GloVe
https://nlp.stanford.edu/projects/glove/

- fastText

#### Solution 2 : Character level (turn each letter into a point)  

See https://arxiv.org/pdf/1508.06615.pdf

Tweettokenizer

 

#### Solution 3 : Learn from scratch
Explore build our own w2v/character level model

 

### 2. How do i turn vectors into decisions?

#### Solution 1 : RNN 

http://karpathy.github.io/assets/rnn/diags.jpeg

Sequence classification with LSTM:

Idea - read each word one at a time into a neural network, put this out at the end
 
In Keras Code :
```
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
```
          
          
#### Solution 2 : CNN (Experimental) 

http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

#### Solution 3: LDA & LSI (With Tf-IDF Vectors)

LDA(Latent Dirichlet Allocation)
https://radimrehurek.com/gensim/models/ldamodel.html

LSI(Latent Sematic Indexing)
https://radimrehurek.com/gensim/models/lsimodel.html

#### Solution 4: LSTM (Experimental)

more info below

#### Solution 5: With already trained models
- GloVe 

- fastText 

#### Solution 6: Gazette(Bag of bad words)

With some words that can be easily spotted as vulgar or offensive, root out these from the get go.

(However can be used sarcastically or ironically as such a binary/ zero-sum stance should not be taken)







# Extra Materials

raw data at 

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#evaluation

reading material
http://karpathy.github.io/2015/05/21/rnn-effectiveness/#recurrent-neural-networks

# Experimental Model 1

Sequence classification with LSTM:

https://keras.io/getting-started/sequential-model-guide/



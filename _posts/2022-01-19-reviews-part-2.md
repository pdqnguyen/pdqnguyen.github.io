---
layout: post
date: 2022-01-19
title: "Reviews of Heavy Metal Albums - Part 2: Review Score Prediction"
categories: jekyll update
permalink: /projects/heavy-metal-analysis/reviews-part-2
summary: |
  Predicting review scores from text using a convolution neural network and GloVe word embeddings. 
---

This article is a part of my [heavy metal lyrics project](/projects/heavy-metal-analysis.html).
If you're interested in seeing the code, check out the
[original notebook](https://github.com/pdqnguyen/metallyrics/blob/main/analyses/reviews/reviews2.ipynb).

## Imports

```python
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
```

```python
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split
```

## Data pre-processing

```python
df = pd.read_csv('reviews.csv')
```

```python
df['review_title'], df['review_score'] = df['review_title'].str.extract('(.*) - (\d+)%').values.T
df['review_score'] = df['review_score'].astype(int)
```

```python
hist = df['review_score'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(hist.index, hist.values, width=1)
plt.xlabel("Review score")
plt.ylabel("Reviews")
plt.show()
```

![histogram](/assets/images/heavy-metal-lyrics/reviews-ml/histogram.png)

#### Split data into training and test sets

```python
X = df['review_content']
y = df['review_score'] / 100
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

#### Sample weights

```python
train_hist = y_train.value_counts().sort_index()
intervals = pd.cut(train_hist.index, np.linspace(y.min(), y.max(), 11), include_lowest=True).categories
bin_counts = np.zeros(len(intervals))
for i, interval in enumerate(intervals):
    for j in train_hist.index:
        if j in interval:
            bin_counts[i] += train_hist[j]
sample_bins = np.zeros(len(y_train), dtype=int)
for i, y in enumerate(y_train):
    for j, interval in enumerate(intervals):
        if y in interval:
            sample_bins[i] = j
            break
sample_weights = 1.0 / bin_counts[sample_bins]
sample_weights /= sample_weights.sum()
```

```python
pd.DataFrame(np.column_stack([y_train, sample_bins, sample_weights]), columns=["y_train", "bin", "weight"]).convert_dtypes().sort_values('y_train').plot('y_train', 'weight')
plt.show()
```

![binning](/assets/images/heavy-metal-lyrics/reviews-ml/binning.png)

#### Convert text to padded sequences of tokens

```python
tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\'')
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.index_word) + 1
print(f"vocabulary size: {vocab_size}")
```

<pre class="code-output">vocabulary size: 166596</pre>

```python
def texts_to_padded(texts, maxlen=None):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = keras.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=maxlen)
    return padded
```

```python
padded_train = texts_to_padded(X_train)
padded_test = texts_to_padded(X_test)
```

```python
pd.DataFrame(np.sum(padded_train > 0, axis=1), columns={"Sequence length"}).describe()
```

<div style="width: 200px">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sequence length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>38981.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>627.327365</td>
    </tr>
    <tr>
      <th>std</th>
      <td>304.061929</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>424.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>759.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5769.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
print(padded_train.shape, y_train.shape, padded_test.shape, y_test.shape)
```

<pre class="code-output">(38981, 5769) (38981,) (9746, 5065) (9746,)</pre>

## Benchmark model

This benchmark model "predicts" scores by sampling from the distribution of scores in the training data, so it represents the outcome of informed random guessing.

```python
train_pdf = y_train.value_counts().sort_index()
train_cdf = train_pdf.cumsum() / train_pdf.sum()


def benchmark_predict(n_samples):
    pred = np.zeros(n_samples)
    r = np.random.rand(n_samples)
    for i in range(n_samples):
        pred[i] = train_cdf.index[np.argmax((train_cdf - r[i]) > 0)]
    return pred

def evaluate_prediction(pred, true, benchmark=False):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.3)
    bins = np.linspace(0, 1, 20)
    hist, bins = np.histogram(np.abs(pred - true), bins=bins)
    bin_centers = bins[:-1] + np.diff(bins)[0] / 2
    ax1.plot(bin_centers, hist, label="model", zorder=1)
    ax2.plot(true, pred, '.', zorder=1)
    ax3.plot(true, pred - true, '.', zorder=1)
    if benchmark:
        y_bench = benchmark_predict(len(true))
        hist_bm, _ = np.histogram(np.abs(y_bench - true), bins=bins)
        ax1.plot(bin_centers, hist_bm, label="benchmark", zorder=0)
        ax2.plot(true, y_bench, '.', zorder=0)
        ax3.plot(true, y_bench - true, '.', zorder=0)
    ax1.set_xlabel("Absolute error")
    ax1.set_ylabel("Samples")
    ax1.legend()
    ax2.set_xlabel("True values")
    ax2.set_ylabel("Predicted values")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xticks(np.linspace(0, 1, 6))
    ax2.set_yticks(np.linspace(0, 1, 6))
    ax2.set_aspect('equal')
    ax3.set_xlabel("True values")
    ax3.set_ylabel("Residual")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-1.02, 1.02)
    ax3.set_xticks(np.linspace(0, 1, 6))
    ax3.set_yticks(np.linspace(-1, 1, 5))
    plt.show()

evaluate_prediction(benchmark_predict(len(y_test)), y_test, benchmark=True)
```

![benchmark](/assets/images/heavy-metal-lyrics/reviews-ml/benchmark.png)


## GloVe word embedding

Adapted from [a Keras tutorial](https://keras.io/examples/nlp/pretrained_word_embeddings/).

Here I create a word embedding layer in order to convert each token in each sequence into a word vector.
I use a pre-trained word embedding, the 6-billion-token, 100-dimensional
Wikipedia+Gigaword 5 word embedding from [GloVe](https://nlp.stanford.edu/projects/glove/).
This transforms each token into a 100-dimensional vector whose location in the word vector space represents
its association to nearby word vectors.
The full dataset will therefore be represented as a matrix of shape (number of samples, sequence length, 100).


```python
path_to_glove_file = "E:/Projects/metallyrics/data/glove.6B.100d.txt"

embedding_vectors = {}
with open(path_to_glove_file, encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embedding_vectors[word] = coefs
print(len(embedding_vectors))
print(len(list(embedding_vectors.values())[0]))
```

<pre class="code-output">
400001
100
</pre>    


```python
embedding_dim = len(list(embedding_vectors.values())[0])
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_vectors.get(word)
    if embedding_vector is not None:
        if len(embedding_vector) > 0:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
            continue
    misses += 1
print("Converted %d words (%d misses)" % (hits, misses))
```

<pre class="code-output">Converted 66882 words (99713 misses)</pre>
    

As an example, we can see look at the 10 nearest words to "fire", based on cosine distance.


```python
vector = embedding_vectors['fire']
cos_dist = np.dot(embedding_matrix, vector) / (np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(vector))
cos_dist = np.nan_to_num(cos_dist, 0)
print([tokenizer.index_word.get(i, 0) for i in cos_dist.argsort()][:-11:-1])
```

<pre class="code-output">
['fire', 'fires', 'fired', 'firing', 'attack', 'explosion', 'blast', 'blaze', 'police', 'ground']
</pre>


```python
embedding_layer = layers.Embedding(
    vocab_size,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)
```




## Convolutional neural network

After a little manual hyperparameter tuning (tweaking the number of filters, dense layer size, learning rate, and regularization methods),
I found that this model was unable to learn at all using mean squared error (MSE) for the loss function.
Mean absolute error (MAE) worked instantly when implemented.
This is probably because MSE does well at punishing outliers, but there the review score range is bounded,
so there are no huge outliers in the data.

I also tested it with and without sample weighting since the test samples are heavily distributed in favor of high-scoring reviews.
I found that with sample weighting, the model was less likely to overestimate the scores of negative reviews.
However, the residual plots show that there is still a decent amount of bias towards overestimating scores,
although it does perform much better than the random sampling benchmark.

Any further tuning should probably be done with cross-validation just to robust,
but I'm pretty happy with the model as is so I'm leaving it as is.

Also, I tried training a recurrent neural network on the data and it miserably overfit,
and tuning took too long because of the very slow training time. Oh well, I'm happy with the ConvNet!


```python
cnn_model = Sequential()
cnn_model.add(embedding_layer)
cnn_model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.GlobalMaxPooling1D())
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64))
cnn_model.add(layers.Dropout(0.2))
cnn_model.add(layers.Dense(1, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=0.001)
cnn_model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mae'])
print(cnn_model.summary())
```

<details>
<summary>Show output</summary>
<pre class="code-output">
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 100)         16659600  
                                                                     
     conv1d (Conv1D)             (None, None, 128)         64128     
                                                                     
     batch_normalization (BatchN  (None, None, 128)        512       
     ormalization)                                                   
                                                                     
     global_max_pooling1d (Globa  (None, 128)              0         
     lMaxPooling1D)                                                  
                                                                     
     flatten (Flatten)           (None, 128)               0         
                                                                     
     dense (Dense)               (None, 64)                8256      
                                                                     
     dropout (Dropout)           (None, 64)                0         
                                                                     
     dense_1 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 16,732,561
    Trainable params: 72,705
    Non-trainable params: 16,659,856
    _________________________________________________________________
    None
</pre>
</details>
<br>

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
```


```python
cnn_history = cnn_model.fit(
    padded_train[::10],
    y_train[::10],
    batch_size=32,
    callbacks=[early_stopping],
    epochs=64,
    sample_weight=sample_weights,
    validation_split=0.2,
    verbose=1
)
```

<details>
<summary>Show output</summary>
<pre class="code-output">
    Epoch 1/64
    98/98 [==============================] - 47s 479ms/step - loss: 3.8145e-05 - mae: 1.5172 - val_loss: 1.3022e-05 - val_mae: 0.5182
    Epoch 2/64
    98/98 [==============================] - 46s 473ms/step - loss: 2.1434e-05 - mae: 0.8327 - val_loss: 1.3523e-05 - val_mae: 0.5150
    Epoch 3/64
    98/98 [==============================] - 48s 487ms/step - loss: 1.6014e-05 - mae: 0.6235 - val_loss: 9.9559e-06 - val_mae: 0.3845
    Epoch 4/64
    98/98 [==============================] - 47s 475ms/step - loss: 1.4144e-05 - mae: 0.5445 - val_loss: 6.8736e-06 - val_mae: 0.2715
    Epoch 5/64
    98/98 [==============================] - 46s 473ms/step - loss: 1.2105e-05 - mae: 0.4791 - val_loss: 1.1832e-05 - val_mae: 0.4729
    Epoch 6/64
    98/98 [==============================] - 47s 475ms/step - loss: 1.0517e-05 - mae: 0.4107 - val_loss: 9.3738e-06 - val_mae: 0.3629
    Epoch 7/64
    98/98 [==============================] - 46s 473ms/step - loss: 1.0134e-05 - mae: 0.3802 - val_loss: 1.4776e-05 - val_mae: 0.5858
    Epoch 8/64
    98/98 [==============================] - 50s 506ms/step - loss: 8.2274e-06 - mae: 0.3308 - val_loss: 5.3127e-06 - val_mae: 0.2089
    Epoch 9/64
    98/98 [==============================] - 48s 495ms/step - loss: 6.6443e-06 - mae: 0.2718 - val_loss: 4.7160e-06 - val_mae: 0.1823
    Epoch 10/64
    98/98 [==============================] - 46s 474ms/step - loss: 6.6578e-06 - mae: 0.2750 - val_loss: 6.7310e-06 - val_mae: 0.2637
    Epoch 11/64
    98/98 [==============================] - 46s 475ms/step - loss: 5.6197e-06 - mae: 0.2327 - val_loss: 4.4481e-06 - val_mae: 0.1656
    Epoch 12/64
    98/98 [==============================] - 47s 478ms/step - loss: 5.9548e-06 - mae: 0.2491 - val_loss: 1.1675e-05 - val_mae: 0.4584
    Epoch 13/64
    98/98 [==============================] - 46s 473ms/step - loss: 6.4460e-06 - mae: 0.2631 - val_loss: 7.6773e-06 - val_mae: 0.3003
    Epoch 14/64
    98/98 [==============================] - 46s 473ms/step - loss: 6.0742e-06 - mae: 0.2480 - val_loss: 1.0781e-05 - val_mae: 0.4244
    Epoch 15/64
    98/98 [==============================] - 47s 476ms/step - loss: 5.6677e-06 - mae: 0.2267 - val_loss: 4.0738e-06 - val_mae: 0.1541
    Epoch 16/64
    98/98 [==============================] - 48s 486ms/step - loss: 4.8291e-06 - mae: 0.2054 - val_loss: 4.0348e-06 - val_mae: 0.1527
    Epoch 17/64
    98/98 [==============================] - 50s 513ms/step - loss: 4.7887e-06 - mae: 0.2035 - val_loss: 4.5678e-06 - val_mae: 0.1723
    Epoch 18/64
    98/98 [==============================] - 52s 535ms/step - loss: 3.6422e-06 - mae: 0.1622 - val_loss: 3.9583e-06 - val_mae: 0.1485
    Epoch 19/64
    98/98 [==============================] - 54s 554ms/step - loss: 4.0679e-06 - mae: 0.1798 - val_loss: 3.8226e-06 - val_mae: 0.1458
    Epoch 20/64
    98/98 [==============================] - 50s 513ms/step - loss: 3.7386e-06 - mae: 0.1657 - val_loss: 4.5246e-06 - val_mae: 0.1741
    Epoch 21/64
    98/98 [==============================] - 46s 474ms/step - loss: 3.8084e-06 - mae: 0.1681 - val_loss: 7.0378e-06 - val_mae: 0.2758
    Epoch 22/64
    98/98 [==============================] - 47s 476ms/step - loss: 3.8768e-06 - mae: 0.1695 - val_loss: 3.8681e-06 - val_mae: 0.1484
    Epoch 23/64
    98/98 [==============================] - 47s 475ms/step - loss: 3.3501e-06 - mae: 0.1529 - val_loss: 3.7756e-06 - val_mae: 0.1439
    Epoch 24/64
    98/98 [==============================] - 46s 471ms/step - loss: 3.5623e-06 - mae: 0.1598 - val_loss: 3.7723e-06 - val_mae: 0.1429
    Epoch 25/64
    98/98 [==============================] - 46s 470ms/step - loss: 3.0570e-06 - mae: 0.1423 - val_loss: 4.0576e-06 - val_mae: 0.1543
    Epoch 26/64
    98/98 [==============================] - 47s 475ms/step - loss: 3.1351e-06 - mae: 0.1482 - val_loss: 4.6272e-06 - val_mae: 0.1774
    Epoch 27/64
    98/98 [==============================] - 46s 470ms/step - loss: 3.1796e-06 - mae: 0.1449 - val_loss: 3.9686e-06 - val_mae: 0.1522
    Epoch 28/64
    98/98 [==============================] - 48s 494ms/step - loss: 3.0467e-06 - mae: 0.1374 - val_loss: 3.6614e-06 - val_mae: 0.1398
    Epoch 29/64
    98/98 [==============================] - 47s 478ms/step - loss: 2.9457e-06 - mae: 0.1390 - val_loss: 4.3404e-06 - val_mae: 0.1656
    Epoch 30/64
    98/98 [==============================] - 47s 477ms/step - loss: 3.0364e-06 - mae: 0.1400 - val_loss: 5.8656e-06 - val_mae: 0.2286
    Epoch 31/64
    98/98 [==============================] - 47s 484ms/step - loss: 3.1033e-06 - mae: 0.1421 - val_loss: 3.6623e-06 - val_mae: 0.1404
    Epoch 32/64
    98/98 [==============================] - 47s 477ms/step - loss: 2.9275e-06 - mae: 0.1364 - val_loss: 4.1055e-06 - val_mae: 0.1573
    Epoch 33/64
    98/98 [==============================] - 48s 490ms/step - loss: 3.0637e-06 - mae: 0.1395 - val_loss: 4.4914e-06 - val_mae: 0.1731
    Epoch 34/64
    98/98 [==============================] - 47s 482ms/step - loss: 3.1534e-06 - mae: 0.1419 - val_loss: 5.2991e-06 - val_mae: 0.2044
    Epoch 35/64
    98/98 [==============================] - 47s 480ms/step - loss: 3.3693e-06 - mae: 0.1523 - val_loss: 3.5920e-06 - val_mae: 0.1378
    Epoch 36/64
    98/98 [==============================] - 47s 476ms/step - loss: 2.8618e-06 - mae: 0.1360 - val_loss: 3.5860e-06 - val_mae: 0.1374
    Epoch 37/64
    98/98 [==============================] - 47s 476ms/step - loss: 3.1351e-06 - mae: 0.1414 - val_loss: 4.3046e-06 - val_mae: 0.1640
    Epoch 38/64
    98/98 [==============================] - 47s 477ms/step - loss: 2.7852e-06 - mae: 0.1338 - val_loss: 4.0075e-06 - val_mae: 0.1526
    Epoch 39/64
    98/98 [==============================] - 47s 477ms/step - loss: 3.5332e-06 - mae: 0.1558 - val_loss: 4.6540e-06 - val_mae: 0.1793
    Epoch 40/64
    98/98 [==============================] - 47s 476ms/step - loss: 2.8457e-06 - mae: 0.1322 - val_loss: 3.9495e-06 - val_mae: 0.1513
    Epoch 41/64
    98/98 [==============================] - 47s 475ms/step - loss: 2.6968e-06 - mae: 0.1292 - val_loss: 4.1121e-06 - val_mae: 0.1571
    Epoch 42/64
    98/98 [==============================] - 47s 476ms/step - loss: 2.6937e-06 - mae: 0.1261 - val_loss: 3.9207e-06 - val_mae: 0.1498
    Epoch 43/64
    98/98 [==============================] - 47s 482ms/step - loss: 2.9295e-06 - mae: 0.1367 - val_loss: 4.1338e-06 - val_mae: 0.1587
    Epoch 44/64
    98/98 [==============================] - 47s 480ms/step - loss: 2.6640e-06 - mae: 0.1266 - val_loss: 3.5145e-06 - val_mae: 0.1349
    Epoch 45/64
    98/98 [==============================] - 47s 480ms/step - loss: 3.0938e-06 - mae: 0.1384 - val_loss: 4.0262e-06 - val_mae: 0.1547
    Epoch 46/64
    98/98 [==============================] - 48s 488ms/step - loss: 3.2265e-06 - mae: 0.1453 - val_loss: 4.8560e-06 - val_mae: 0.1887
    Epoch 47/64
    98/98 [==============================] - 47s 476ms/step - loss: 2.5910e-06 - mae: 0.1222 - val_loss: 3.5102e-06 - val_mae: 0.1350
    Epoch 48/64
    98/98 [==============================] - 47s 478ms/step - loss: 2.7827e-06 - mae: 0.1310 - val_loss: 3.4579e-06 - val_mae: 0.1330
    Epoch 49/64
    98/98 [==============================] - 47s 478ms/step - loss: 2.8334e-06 - mae: 0.1303 - val_loss: 3.5284e-06 - val_mae: 0.1348
    Epoch 50/64
    98/98 [==============================] - 47s 477ms/step - loss: 2.7222e-06 - mae: 0.1272 - val_loss: 3.4825e-06 - val_mae: 0.1324
    Epoch 51/64
    98/98 [==============================] - 47s 481ms/step - loss: 2.5269e-06 - mae: 0.1211 - val_loss: 3.7103e-06 - val_mae: 0.1398
    Epoch 52/64
    98/98 [==============================] - 48s 491ms/step - loss: 2.6162e-06 - mae: 0.1238 - val_loss: 3.4978e-06 - val_mae: 0.1323
    Epoch 53/64
    98/98 [==============================] - 47s 483ms/step - loss: 2.9474e-06 - mae: 0.1338 - val_loss: 4.5738e-06 - val_mae: 0.1778
    Epoch 54/64
    98/98 [==============================] - 47s 481ms/step - loss: 2.7510e-06 - mae: 0.1285 - val_loss: 3.4152e-06 - val_mae: 0.1309
    Epoch 55/64
    98/98 [==============================] - 47s 478ms/step - loss: 2.7366e-06 - mae: 0.1272 - val_loss: 4.0273e-06 - val_mae: 0.1555
    Epoch 56/64
    98/98 [==============================] - 47s 481ms/step - loss: 2.4605e-06 - mae: 0.1137 - val_loss: 3.4660e-06 - val_mae: 0.1327
    Epoch 57/64
    98/98 [==============================] - 47s 480ms/step - loss: 2.5482e-06 - mae: 0.1206 - val_loss: 3.5785e-06 - val_mae: 0.1365
    Epoch 58/64
    98/98 [==============================] - 47s 480ms/step - loss: 2.3459e-06 - mae: 0.1149 - val_loss: 3.5496e-06 - val_mae: 0.1341
    Epoch 59/64
    98/98 [==============================] - 47s 483ms/step - loss: 2.2755e-06 - mae: 0.1077 - val_loss: 3.4191e-06 - val_mae: 0.1300
    Epoch 60/64
    98/98 [==============================] - 48s 486ms/step - loss: 2.6701e-06 - mae: 0.1238 - val_loss: 4.0818e-06 - val_mae: 0.1564
    Epoch 61/64
    98/98 [==============================] - 48s 487ms/step - loss: 2.4703e-06 - mae: 0.1170 - val_loss: 3.5177e-06 - val_mae: 0.1338
    Epoch 62/64
    98/98 [==============================] - 47s 485ms/step - loss: 2.1340e-06 - mae: 0.1054 - val_loss: 3.7672e-06 - val_mae: 0.1422
    Epoch 63/64
    98/98 [==============================] - 48s 486ms/step - loss: 2.5771e-06 - mae: 0.1231 - val_loss: 3.4108e-06 - val_mae: 0.1291
    Epoch 64/64
    98/98 [==============================] - 48s 487ms/step - loss: 2.3585e-06 - mae: 0.1110 - val_loss: 3.4937e-06 - val_mae: 0.1327
</pre>
</details>
<br>

```python
train_metrics = cnn_model.evaluate(padded_train, y_train, verbose=0)
test_metrics = cnn_model.evaluate(padded_test, y_test, verbose=0)
train_metrics = {cnn_model.metrics_names[i]: value for i, value in enumerate(train_metrics)}
test_metrics = {cnn_model.metrics_names[i]: value for i, value in enumerate(test_metrics)}
```

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.subplots_adjust(hspace=0.4)
ax1.set_title('Loss')
ax1.plot(cnn_history.history['loss'], label='train')
ax1.plot(cnn_history.history['val_loss'], label='test')
ax1.set_yscale('log')
ax1.legend()
ax2.set_title('Mean Absolute Error')
ax2.plot(cnn_history.history['mae'], label='train')
ax2.plot(cnn_history.history['val_mae'], label='test')
ax2.set_yscale('log')
ax2.legend()
plt.show()
for metric in cnn_model.metrics_names:
    print(f"Train {metric}: {train_metrics[metric]:.2f}")
    print(f"Test  {metric}: {test_metrics[metric]:.2f}")
```

![training](/assets/images/heavy-metal-lyrics/reviews-ml/training.png)
    

<pre class="code-output">
    Train loss: 0.13
    Test  loss: 0.13
    Train mae: 0.13
    Test  mae: 0.13
</pre>

```python
y_pred = cnn_model.predict(padded_test)[:, 0]
y_pred = np.maximum(0, np.minimum(1, y_pred))
```


```python
evaluate_prediction(y_pred, y_test, benchmark=True)
```

![comparison](/assets/images/heavy-metal-lyrics/reviews-ml/comparison.png)


```python
texts = ["This album is bad", "This album is okay", "This album is good", "This album is awesome"]
pred = cnn_model.predict(texts_to_padded(texts, maxlen=padded_train.shape[0]))[:, 0]
plt.barh(range(len(pred)), pred[::-1])
plt.yticks(range(len(pred)), texts[::-1])
plt.xlabel("Predicted score")
plt.show()
```

![example](/assets/images/heavy-metal-lyrics/reviews-ml/example.png)

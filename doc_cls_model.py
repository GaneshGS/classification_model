from __future__ import absolute_import, division, print_function, unicode_literals
def doc_cls_model(): 
  import tensorflow as tf
  tf.enable_eager_execution()
  import pandas as pd
  import numpy as np
  from numpy import genfromtxt  
  import io
  import os
  import tarfile
  from google.colab import drive
  from tensorflow import keras
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences  

  drive.mount('/content/drive/') 
  %cd '/content/drive/My Drive'
  !tar -xvf yahoo_answers_csv.tar.gz
  %cd '/content/drive/My Drive/yahoo_answers_csv'
  column_defaults = [tf.int32, tf.constant('', tf.string), tf.constant('', tf.string), tf.constant('', tf.string)]
  column_names = ['content', 'qtype', 'qcontent', 'answer']
  feature_names = column_names[:0]
  label_name = column_names[0]
  batch_size = 1

  train_data = tf.data.experimental.make_csv_dataset("train.csv",
                                                      batch_size,
                                                      column_names,
                                                      column_defaults,
                                                      label_name,
                                                      field_delim = ',',
                                                      num_epochs=1)
  test_data = tf.data.experimental.make_csv_dataset("test.csv",
                                                      batch_size,
                                                      column_names,
                                                      column_defaults,
                                                      label_name,
                                                      field_delim = ',',
                                                      num_epochs=1)  
  features, labels = next(iter(train_data))

  def features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

  train_data = train_data.map(features_vector)
  test_data = test_data.map(features_vector)

  train_sentences = []
  train_labels = []
  test_sentences = []
  test_labels = []

  for i,j in train_data:
    train_sentences.append(str(i.numpy()))
    train_labels.append(int(j.numpy()))

  for i,j in test_data:
    test_sentences.append(str(i.numpy()))
    test_labels.append(int(j.numpy()))
       
  train_labels_f = []
  test_labels_f = []

  for i in train_labels:
    train_labels_f.append(i-1)
  for i in test_labels:
    test_labels_f.append(i-1)
  
  train_labels_fn = np.array(train_labels_f)
  test_labels_fn = np.array(test_labels_f)

  #return train_sentences, test_sentences, train_labels_f, test_labels_f
    
#def build_model():    
  vocab_size = 50000
  embedding_dim = 128
  max_length = 200
  trunc_type='post'
  oov_tok = "<OOV>"

  tokenizer = Tokenizer(num_words =vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(train_sentences)
  word_index = tokenizer.word_index

  train_sequences = tokenizer.texts_to_sequences(train_sentences)
  train_padded = pad_sequences(train_sequences,maxlen=max_length, truncating=trunc_type, padding='post')

  test_sequences = tokenizer.texts_to_sequences(test_sentences)
  test_padded = pad_sequences(test_sequences,maxlen=max_length, truncating=trunc_type, padding='post')

  train_sent_par = train_padded[:1050000]
  train_label_par = train_labels_fn[:1050000]

  validation_sentence = train_padded[1050000:]
  validation_label = train_labels_fn[1050000:]

  model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
  tf.keras.layers.Dropout(0.1),             
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax')
  ])
    
  model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 
  model_summary = model.summary()  
  output = model.fit(train_sent_par,
                    train_label_par,
                    epochs = 20,
                    batch_size = 256,
                    validation_data=(validation_sentence, validation_label),
                    verbose=1
                    )   
  model_results =  model.evaluate(test_padded, test_labels_f)
  model_save = model.save('model.h5')
  return output, model_summary, model_results, model_save  

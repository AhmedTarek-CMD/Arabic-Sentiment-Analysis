from tensorflow.keras import layers, models
import pickle
from nltk.stem.isri import ISRIStemmer
import regex
from langdetect import detect
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import csv
import emoji
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, LearningRateScheduler

###############
# preprocessing
###############

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

stop_words_arabic = set(stopwords.words('arabic'))

english_words = set(nltk.corpus.words.words())

arabic_numbers = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']


def doesnt_contain(word):
    for char in arabic_numbers:
        if char in word:
            return False
    return True


def preprocess_text(text):
    # remove emojis
    text = emoji.replace_emoji(text, replace='')

    # text = replace_emojis(emoji_dict, text)

    # remove non-letters
    text = re.sub(r'[^a-zA-Z\s\u0621-\u064A]', '', text)

    # remove english alphabets
    text = re.sub(r'[a-zA-Z0-9]', '', text)

    # Replace consecutive repeated punctiation marks with one
    text = re.sub(r'(.؟!)\1+', r'\1', text)
    # remove repeated letters and replacing it by 2 letters like هارووون -> هاروون
    text = re.sub(r'(.)\1+', r'\1\1', text)

    tokens = word_tokenize(text)

    # check if not an english word and not an arabic stop word and doesn't contain arabic numbers
    non_english_words = [word for word in tokens if word.lower(
    ) not in english_words and word not in stop_words_arabic and doesnt_contain(word)]

    # Join the non-English words back into a text
    text = ' '.join(non_english_words)

    # text without double spaces
    text = re.sub(r'\s+', ' ', text)

    tokens = word_tokenize(text)

    # Remove consecutive words
    unique_tokens = []
    prev_token = None
    for token in tokens:
        if token != prev_token:
            unique_tokens.append(token)
            prev_token = token

    text = ' '.join(unique_tokens)

    return text


###################
# end preprocessing
###################

data = pd.read_excel('data/train.xlsx')

data['processed_review'] = data['review_description'].apply(preprocess_text)
X = data['processed_review']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index
num_of_classes = 3

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# hyper parameters
max_len = max(map(len, sequences))
print(max_len)

embedding_dim = 100

padded_sequences = pad_sequences(sequences, maxlen=max_len)  # , padding='post'
print(padded_sequences)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['rating'])
print(f"Encoded Labels:\n {y}")
le_name_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, y, test_size=0.2, random_state=42)


class RNN():
    def __init__(self, words_size, embedding_dim, input_size):
        self.words_size = words_size
        self.embedding_dim = embedding_dim
        self.input_size = input_size

    def __call__(self, lstm_layers, dense_layers, dense_activation, dropout, num_of_classes, output_activation, optimizer, loss):
        model = models.Sequential()
        model.add(layers.Embedding(self.words_size,
                  self.embedding_dim, input_length=self.input_size))
        lstm_layer_units = 100
        for l in range(lstm_layers):
            if (l == lstm_layers-1):
                model.add(layers.LSTM(units=lstm_layer_units))
            else:
                model.add(layers.LSTM(
                    units=lstm_layer_units, return_sequences=True))
        dense_layer_units = 100
        for l in range(dense_layers):
            model.add(layers.Dense(units=dense_layer_units,
                      activation='relu', kernel_regularizer=l2(0.001)))
        model.add(layers.Dropout(dropout))
        # output layer
        model.add(layers.Dense(units=num_of_classes,
                  activation=output_activation))
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model


print(len(word_index) + 1)

rnn = RNN(len(word_index) + 1,
          embedding_dim=embedding_dim,
          input_size=max_len)

optimizer = Adam(learning_rate=0.001)

model = rnn(lstm_layers=2,
            dense_layers=1,
            dense_activation='relu',
            dropout=0.3,
            num_of_classes=num_of_classes,
            output_activation='softmax',
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy')

epochs = 2

early_stopping = EarlyStopping(
    monitor='val_loss', patience=4, restore_best_weights=True)
lr_schedule = LearningRateScheduler(
    lambda epoch: max(1e-5, 0.001 * 0.9 ** epoch))

model.fit(X_train, y_train, epochs=epochs, batch_size=128,
          validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)

with open('model_sameh.pkl', 'wb') as file:
    pickle.dump(model, file)

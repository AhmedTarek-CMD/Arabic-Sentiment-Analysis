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

# hyper parameters
max_len = 291

# test
loaded_model = pickle.load(open('model_sameh.pkl', 'rb'))

data = pd.read_csv('data/test _no_label.csv')
data['processed_review'] = data['review_description'].apply(preprocess_text)
Xtest = data['processed_review']

with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

tokenizer = loaded_tokenizer
# tokenizer.fit_on_texts(Xtest)
sequences = tokenizer.texts_to_sequences(Xtest)

padded_sequences_test = pad_sequences(
    sequences, maxlen=max_len)  # padding='post'

predictions = loaded_model.predict(padded_sequences_test)


def approximate_number(sublist):
    pclass = np.argmax(sublist, axis=-1)

    return pclass - 1


flattened_predictions = [approximate_number(
    sublist) for sublist in predictions]
print(predictions)

data['rating'] = flattened_predictions
data = data.drop('review_description', axis=1)
data = data.drop('processed_review', axis=1)


# Save predictions to a CSV file
csv_file_path = 'predictions_large_latest_2.csv'
data.to_csv(csv_file_path, index=False)

print(f"Predictions saved to {csv_file_path}")

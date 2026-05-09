#import
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')


class TextPreprocessor:
    def __init__(self, tokenizer, max_len=100):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

#Cleaning the text
    def clean(self, text):
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'s", " is", text)

        # Remove emojis / non-ascii
        text = text.encode("ascii", "ignore").decode("utf-8")

        # Remove special chars
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text if len(text) > 0 else "empty_review"

#Lemmatization
    def lemmatize(self, text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)

        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]

        return " ".join(lemmatized)

    def preprocess(self, text):
        text = self.clean(text)
        text = self.lemmatize(text)
        return text

    def text_to_sequence(self, text):
        text = self.preprocess(text)

        seq = self.tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=self.max_len, padding="post")

        return seq, text
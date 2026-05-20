#import
import re
import nltk
import os

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
if os.path.exists("/opt/render/nltk_data"):
    nltk.data.path.append("/opt/render/nltk_data")


class TextPreprocessor:

    def __init__(self, word_index, max_len=100):

        self.word_index = word_index
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

    # Cleaning
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

    # Lemmatization
    def lemmatize(self, text):

        words = text.split()

        pos_tags = pos_tag(words)

        lemmatized = [
            self.lemmatizer.lemmatize(
                word,
                self.get_wordnet_pos(tag)
            )
            for word, tag in pos_tags
        ]

        return " ".join(lemmatized)

    def preprocess(self, text):

        text = self.clean(text)
        text = self.lemmatize(text)

        return text

    # Manual padding
    def pad_sequence_manual(self, seq):

        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))

        else:
            seq = seq[:self.max_len]

        return seq

    # Convert text → sequence
    def text_to_sequence(self, text):

        text = self.preprocess(text)

        words = text.split()

        seq = [
            self.word_index.get(word, 1)
            for word in words
        ]

        seq = self.pad_sequence_manual(seq)

        return [seq], text
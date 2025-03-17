Install dependencies:

pip install nltk gensim sumy

import nltk
import re
import heapq
import gensim
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from gensim.summarization import summarize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")

def nltk_summarizer(text, num_sentences=3):
    """
    Text Summarization using NLTK (Frequency-Based Approach)
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    word_frequencies = {}

    for word in words:
        if word.lower() not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    max_freq = max(word_frequencies.values())

    for word in word_frequencies:
        word_frequencies[word] /= max_freq

    sentence_list = sent_tokenize(text)
    sentence_scores = {}

    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return " ".join(summary_sentences)


def gensim_summarizer(text, ratio=0.3):
    """
    Text Summarization using Gensim's summarize() function
    """
    return gensim.summarization.summarize(text, ratio=ratio)


def sumy_summarizer(text, num_sentences=3):
    """
    Text Summarization using Sumy's LSA Summarizer
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)


# Sample text for summarization
sample_text = """
Artificial Intelligence (AI) is a rapidly growing field that has transformed various industries,
including healthcare, finance, and marketing. AI-powered applications use machine learning and deep learning
algorithms to analyze data, automate processes, and provide intelligent decision-making. With advancements in
natural language processing (NLP), AI can now understand and generate human-like text, making it useful for chatbots,
content generation, and language translation. As AI technology continues to evolve, it will play a crucial role in shaping
the future of automation and intelligence.
"""

print("\nOriginal Text:\n", sample_text)

print("\n🔹 NLTK Summarization:\n", nltk_summarizer(sample_text))
print("\n🔹 Gensim Summarization:\n", gensim_summarizer(sample_text))
print("\n🔹 Sumy Summarization:\n", sumy_summarizer(sample_text))

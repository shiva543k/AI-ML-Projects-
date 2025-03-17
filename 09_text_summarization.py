Install dependencies:

pip install nltk transformers torch

import nltk
from transformers import pipeline

# Download necessary NLP models
nltk.download('punkt')

# Load Hugging Face summarization model
summarizer = pipeline("summarization")

# Sample long text
long_text = """
Artificial Intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that 
typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, 
and language understanding. AI is categorized into two main types: narrow AI, which is designed to perform a specific 
task such as voice recognition, and general AI, which has the potential to perform any intellectual task that a human can do.
"""

# Generate summary
summary = summarizer(long_text, max_length=50, min_length=20, do_sample=False)

# Print summarized text
print("Original Text:\n", long_text)
print("\nSummarized Text:\n", summary[0]['summary_text'])

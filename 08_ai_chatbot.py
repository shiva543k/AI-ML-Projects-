pip install nltk - - install dependcies 
import nltk
import random
import string

from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize

# Define chatbot responses using pattern-response pairs
chat_pairs = [
    (r"hi|hello|hey", ["Hello!", "Hi there!", "Hey! How can I help you?"]),
    (r"how are you", ["I'm good! How about you?", "I'm fine, thanks for asking!"]),
    (r"what is your name", ["I'm an AI chatbot!", "You can call me ChatBot."]),
    (r"what can you do", ["I can answer basic questions. Try asking something!"]),
    (r"bye|exit", ["Goodbye!", "See you later!", "Take care!"]),
]

# Create chatbot instance
chatbot = Chat(chat_pairs, reflections)

# Simple chatbot loop
def chat():
    print("AI Chatbot: Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ").lower()
        if user_input == "exit":
            print("Chatbot: Goodbye!")
            break
        response = chatbot.respond(user_input)
        print("Chatbot:", response if response else "Sorry, I didn't understand that.")

# Run chatbot
if __name__ == "__main__":
    chat() 

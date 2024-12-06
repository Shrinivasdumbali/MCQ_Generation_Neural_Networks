#  MCQ Generation Through Input Document(txt file)
import random
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout

def extract_sentences(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def generate_mcq(sentences, num_questions):
    questions = []
    for _ in range(num_questions):
        sentence = random.choice(sentences)
        # Split sentence into words
        words = sentence.split()
        if not words:  # Check if words list is empty
            continue
        # Randomly choose a word to replace with a blank
        blank_index = random.randint(0, len(words) - 1)
        blank_word = words[blank_index]
        # Create options with a randomly shuffled subset of words from the sentence
        options = random.sample(words, min(4, len(words)))
        # Ensure correct answer is in the options
        if blank_word not in options:
            options[random.randint(0, min(3, len(words)))] = blank_word
        # Form question and options
        question = ' '.join(word if word != blank_word else '_____' for word in words)
        questions.append((question, options, blank_word))
    return questions

def prepare_data(sentences, max_seq_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    return padded_sequences, tokenizer

def create_model(vocab_size, max_seq_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_seq_length))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))  # Output layer with vocab_size units
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

file_path = '/content/The Crow and The Pitcher story.txt' 
num_questions = 5  

sentences = extract_sentences(file_path)
mcq_questions = generate_mcq(sentences, num_questions)
texts = [question[0] for question in mcq_questions]

max_seq_length = 35  # Maximum sequence length for padding

padded_sequences, tokenizer = prepare_data(texts, max_seq_length)
vocab_size = len(tokenizer.word_index) + 1

model = create_model(vocab_size, max_seq_length)
model.fit(padded_sequences, np.zeros((padded_sequences.shape[0], 1)), epochs=10, verbose=1)

# Generate questions using the trained model
generated_questions = []
for _ in range(num_questions):
    random_sentence = random.choice(sentences)
    words = random_sentence.split()
    if not words:  # Check if words list is empty
        continue
    blank_word = random.choice(words)
    options = [word for word in words if word != blank_word]
    options = random.sample(options, min(3, len(options)))  # Select at most 3 unique options
    options.append(blank_word)  # Add the blank word as an option
    random.shuffle(options)  # Shuffle the options
    generated_question = random_sentence.replace(blank_word, '____'), options, blank_word
    generated_questions.append(generated_question)

print("\nGenerated Questions:")
for i, (question, options, _) in enumerate(generated_questions):
    print(f"{i+1}. {question}")
    for j, option in enumerate(options):
        print(f"   {chr(ord('A')+j)}. {option}")
    print()






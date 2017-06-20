"""
RNN-based generator of new chapters of the Mahabaratha.
The artificial intelligence is based on LSTM networks.

All the code is written using Keras library:
https://keras.io/

The code is created using those two examples:
https://gist.github.com/karpathy/d4dee566867f8291f086
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py


author: Lorenzo Peppoloni
mail: l.peppoloni@gmail.com
"""

# Imports

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
from termcolor import colored
import argparse


def sample_output(preds, temperature=1.0):
    """
    Function to sample an index from a probability distribution
    :param preds: The probability distribution
    :param temperature: The temperature for the sampling
    :return: return the sampled index
    """

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


parser = argparse.ArgumentParser(
    description='Train RNN to generate new chapters of the Mahabaratha')
parser.add_argument('-i', '--input', type=str,
                    help="path to the mahabaratha txt file", required=True, dest="infile")
parser.add_argument('-l', '--layers', help="List of hidden layers for the net", nargs='+', type=int)
parser.add_argument('-s', '--sequence', help="Length of the sequence", type=int, default=25)
parser.add_argument('--learningrate', help="Learning rate for the net", type=float, default=0.01)
parser.add_argument('--dropout', help="Dropout between layers for the net", type=float, default=0.3)
parser.add_argument('--it', help="Training iteration", type=int, default=300)
parser.add_argument('-b', '--batch', help="Training batch size", type=int, default=256)


args = parser.parse_args()

# Load the Mahabaratha as a .txt file, passed as argument
print colored("Loading txt file %s" % args.infile, "cyan")

data = open(args.infile, 'r').read()

chars = list(set(data))

data_size, vocab_size = len(data), len(chars)

print 'Loaded book has %d characters and %d unique characters' % (data_size, vocab_size)

# Every character is coded with a unique integer
# The dict will be used to create the hot encoding vector later

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Prepare chunks of data to be fed to the network
chunks = []
next_chars = []


# The text is divided in chunks swiping the text from left to right with a unitary step
# Every chunk is a data point (X[i, :, :]), and the following character of every chunk is the output (Y[i, :])

for i in range(0, len(data) - args.sequence):
    chunks.append(data[i: i + args.sequence])
    next_chars.append(data[i + args.sequence])

print 'Generated %d chunks of text' % len(chunks)

X = np.zeros((len(chunks), args.sequence, len(chars)), dtype=np.bool)
Y = np.zeros((len(chunks), len(chars)), dtype=np.bool)

# Create input and output for the net
# X = [samples x sequence length x features]
# Y = [samples x features]
# Features are an hot encoding vector for encoding the integer
# associated to every unique character

for i, chunk in enumerate(chunks):
    for t, char in enumerate(chunk):
        X[i, t, char_to_ix[char]] = 1
    Y[i, char_to_ix[next_chars[i]]] = 1


# Build the model
print colored("Building the model: \n\t Number of layers %d" % len(args.layers), "cyan")
model = Sequential()
for i, lay in enumerate(args.layers):
    if i == 0:
        model.add(LSTM(lay, input_shape=(args.sequence, len(chars)), return_sequences=True))
    else:
        model.add(Dropout(args.dropout))
        model.add(LSTM(lay, return_sequences=True))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=args.learningrate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


for iteration in range(1, args.it):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, Y,
              batch_size=args.batchsize,
              epochs=1)

    start_index = random.randint(0, len(data) - args.sequence - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()






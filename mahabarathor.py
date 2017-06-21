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
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
from termcolor import colored
import argparse

parser = argparse.ArgumentParser(
    description='Train RNN to generate new chapters of the Mahabaratha')
parser.add_argument('-i', '--input', type=str,
                    help="path to the mahabaratha txt file", required=True, dest="infile")
parser.add_argument('-l', '--layers', help="List of hidden layers for the net", nargs='+', type=int)
parser.add_argument('-s', '--sequence', help="Length of the sequence", type=int, default=25)
parser.add_argument('--learningrate', help="Learning rate for the net", type=float, default=0.01)
parser.add_argument('--dropout', help="Dropout between layers for the net", type=float, default=0.2)
parser.add_argument('--it', help="Training iteration", type=int, default=300)
parser.add_argument('-b', '--batch', help="Training batch size", type=int, default=256)
parser.add_argument('-w', '--weight', type=str,
                    help="Path to the weight file", default="", dest="weights")

args = parser.parse_args()

# Load the Mahabaratha as a .txt file, passed as argument
print colored("Loading txt file %s" % args.infile, "cyan")

data = open(args.infile, 'r').read()

data = data[:700000]

chars = list(set(data))

print 'Loaded book has %d characters and %d unique characters' % (len(data), len(chars))

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
        if i == len(args.layers) - 1:
            model.add(LSTM(lay, input_shape=(args.sequence, len(chars))))
        else:
            model.add(LSTM(lay, input_shape=(args.sequence, len(chars)), return_sequences=True))
        model.add(Dropout(args.dropout))
    else:
        if i == len(args.layers) - 1:
            model.add(LSTM(lay))
        else:
            model.add(LSTM(lay, return_sequences=True))
        model.add(Dropout(args.dropout))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=args.learningrate)

if args.weights != '':
    print colored("Loading weights from file %s" % args.weights, "red")
    model.load_weights(args.weights)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)


for iteration in range(1, args.it):

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

    # Model checkpoints for weights
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, Y,
              batch_size=args.batch,
              epochs=1, callbacks=callbacks_list)
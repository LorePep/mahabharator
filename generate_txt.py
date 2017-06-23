from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
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
parser.add_argument('-w', '--weight', type=str,
                    help="Path to the weight file", required=True, dest="weights")
parser.add_argument('-l', '--layers', help="List of hidden layers for the net", nargs='+', type=int)
parser.add_argument('-t', '--temperature', help="Temperature for softmax", default=0.5, type=float)
parser.add_argument('-o', '--outfile', help="Output file for generated text", default="test.txt", type=str)
parser.add_argument('-s', '--sequence', help="Length of the sequence", type=int, default=25)
parser.add_argument('-c', '--chars', help="Length of the sequence to be generated", type=int, default=1000)


args = parser.parse_args()

# Load the Mahabaratha as a .txt file, passed as argument
print colored("Loading txt file %s" % args.infile, "cyan")

data = open(args.infile, 'r').read()

data = data[:700000]
chars = list(set(data))

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Create model
print colored("Building the model: \n\t Number of layers %d" % len(args.layers), "cyan")
model = Sequential()
for i, lay in enumerate(args.layers):
    if i == 0:
        if i == len(args.layers) - 1:
            model.add(LSTM(lay, input_shape=(args.sequence, len(chars))))
        else:
            model.add(LSTM(lay, input_shape=(args.sequence, len(chars)), return_sequences=True))
    else:
        if i == len(args.layers) - 1:
            model.add(LSTM(lay))
        else:
            model.add(LSTM(lay, return_sequences=True))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)

model.load_weights(args.weights)

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Sample seed text and generate new text
start_index = random.randint(0, len(data) - args.sequence - 1)

generated = ''

sentence = data[start_index: start_index + args.sequence]
generated += sentence
print colored("Generating with seed: %s" % sentence, "red")

sys.stdout.write(generated)

# Open file for output
out = open(args.outfile, 'w')

for i in range(args.chars):
    x = np.zeros((1, args.sequence, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_to_ix[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample_output(preds, args.temperature)
    next_char = ix_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    out.write(next_char)

out.close()

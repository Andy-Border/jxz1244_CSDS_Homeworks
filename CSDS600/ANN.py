import sys

import numpy as np
from Eval import Eval

import torch
import torch.nn as nn
import torch.optim as optim
import random

from imdb import IMDBdata


class ANN(nn.Module):
    def __init__(self, X, Y, VOCAB_SIZE, DIM_EMB=10, NUM_CLASSES=2):
        super(ANN, self).__init__()
        (self.VOCAB_SIZE, self.DIM_EMB, self.NUM_CLASSES) = (VOCAB_SIZE, DIM_EMB, NUM_CLASSES)
        # TODO: Initialize parameters.
        self.E = nn.Embedding(VOCAB_SIZE, DIM_EMB)
        self.R = nn.ReLU()
        self.W = nn.Linear(DIM_EMB, NUM_CLASSES)
        self.softmax = nn.Softmax(dim=0)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, X, train=True):
        # TODO: Implement forward computation.
        embedding = self.E(X)
        hidden_layer = torch.sum(self.R(embedding), dim=0)
        output = self.W(hidden_layer)

        return self.softmax(output)


def Eval_ANN(X, Y, mlp):
    num_correct = 0
    for i in range(len(X)):
        logProbs = mlp.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))


def TrainANN(X, Y, vocab_size, n_iter):
    print("Start Training!")
    mlp = ANN(X, Y, vocab_size)
    # TODO: initialize optimizer.
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    # batch_size = 32

    for epoch in range(n_iter):
        # shuffled_i = list(range(0,len(Y)))
        # random.shuffle(shuffled_i)
        total_loss = 0.0
        for i in range(len(Y)):
            x = X[i]
            y_onehot = torch.zeros(mlp.NUM_CLASSES)
            # y_onehot[int(Y[start:start + batch_size])] = 1.
            y_onehot[int(Y[i])] = 1.

            mlp.zero_grad()
            probs = mlp.forward(x)
            loss = torch.sum((probs - y_onehot) ** 2)
            total_loss += loss

            loss.backward()
            optimizer.step()
        print(f"loss on epoch {epoch} = {total_loss}")
    return mlp


if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    train.vocab.Lock()
    test = IMDBdata("%s/dev" % sys.argv[1], vocab=train.vocab)

    mlp = TrainANN(train.XwordList, (train.Y + 1.0) / 2.0, train.vocab.GetVocabSize(), int(sys.argv[2]))
    Eval_ANN(test.XwordList, (test.Y + 1.0) / 2.0, mlp)

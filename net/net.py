import math
import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def dsigmoid(a):
    siga = sigmoid(a)
    return siga * (1 - siga)

def relu(a):
    return np.maximum(0, a)

def drelu(a):
    if np.isscalar(a):
        return 0 if a <= 0 else 1

    d = np.copy(a)
    d[np.where(d <= 0)] = 0
    d[np.where(d > 0)] = 1
    return d

def lrelu(a):
    if np.isscalar(a):
        return 0.01 * a if a <= 0 else a

    d = np.copy(a)
    for i in range(len(d)):
        d[i] = 0.01 * d[i] if d[i] <= 0 else d[i]
    return d

def dlrelu(a):
    if np.isscalar(a):
        return 0.01 if a <= 0 else 1

    d = np.copy(a)
    d[np.where(d <= 0)] = 0.01
    d[np.where(d > 0)] = 1
    return d

class Net:
    def __init__(self, shape, mu, afuncs, dafuncs):
        self.shape = shape
        self.mu = mu
        self.afuncs = afuncs
        self.dafuncs = dafuncs
        self._cache = None

        weights = []
        # initialize weights to small positive values
        for i in range(1, len(shape)):
            weights.append(np.random.uniform(-0.1, 0.1, (shape[i], shape[i-1])))
        self.weights = weights

    def train(self, data, values):
        assert len(data) == len(values), 'Training data and values must be the same dimensions'
        for i in range(len(data)):
            self.feed_forward(data[i])
            self.backprop(values[i])

    def feed_forward(self, inp):
        assert len(inp) == self.shape[0], 'Input size must match input nodes'
        self._cache = [inp]

        z = []
        zi = np.array(inp)
        for i in range(len(self.shape)-1):
            ai = []
            weights = self.weights[i]
            for j in range(weights.shape[0]):
                ai.append(np.dot(zi, weights[j]))
            self._cache.append(ai)
            zi = self.afuncs[i](np.array(ai))
            z.append(zi)

        return z[-1]

    def backprop(self, outp):
        # calculate delta for each node
        delta = []
        for i in range(len(self.shape)-1, 0, -1):
            deltaj = []
            for j in range(self.shape[i]):
                aj = self._cache[i][j]
                if i == len(self.shape)-1:
                    deltaj.append(self.dafuncs[i-1](aj) * (self.afuncs[i-1](aj) - outp[j]))
                else:
                    weights = self.weights[i]
                    wkj = [weights[k][j] for k in range(len(weights))]
                    deltaj.append(self.dafuncs[i-1](aj) * np.dot(wkj, delta[0]))
            delta.insert(0, deltaj)

        # update weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if k == 0:
                        self.weights[i][j][k] -= self.mu * (delta[i][j] * self._cache[i][k])
                    else:
                        self.weights[i][j][k] -= self.mu * (delta[i][j] * self.afuncs[i](self._cache[i][k]))

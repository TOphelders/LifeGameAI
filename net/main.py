import numpy as np

import net as netutils

data_prefix = './data/'
label_prefix = './data/'

def read_data(f):
    lines = [[*map(int, line.rstrip('\n').split(','))]
             for line in open(data_prefix + f)]
    return lines

def read_labels(f):
    lines = [int(line.rstrip('\n'))
             for line in open(label_prefix + f)]
    labels = []
    for line in lines:
        if line == 5:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    return labels

if __name__ == '__main__':
    shape = [64, 32, 2]
    mu = 0.1
    afuncs = [netutils.sigmoid] * 2
    dafuncs = [netutils.dsigmoid] * 2
    net = netutils.Net(shape, mu, afuncs, dafuncs)

    train_data = read_data('data.csv')
    train_values = read_labels('labels.csv')

    test_data = read_data('testdata.csv')
    test_values = read_labels('testlabels.csv')

    for _ in range(20):
        net.train(train_data, train_values)

    correct = 0
    for i in range(len(test_data)):
        out = np.argmax(net.feed_forward(test_data[i]))
        if test_values[i][out] == 1:
            correct += 1
    print('correct: ' + str(correct/len(test_data)))

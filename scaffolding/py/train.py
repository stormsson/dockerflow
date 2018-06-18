#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# generic stuff
import math
import numpy as np
import random

# chart lib
import plotly
import plotly.graph_objs as go
from plotly import tools


# deep learning libs
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense



X = []
Y = []

def generateData(qty=1000):
    data = {
        'in': [],
        'out': []
    }
    for el in xrange(1, qty+1):
        rnd = 2 * random.random()
        #in: rads
        data['in'].append([rnd])
        X.append(rnd)

        data['out'].append([math.sin(math.pi * rnd)])
        Y.append(math.sin(math.pi * rnd))

    # convert in numpy array
    data['in'] = np.asarray(data['in'], dtype="float32")
    data['out'] = np.asarray(data['out'], dtype="float32")

    return data


def step1():
    """
    in this step, input data are generated
    """


    chart_data = [go.Scatter(x=X, y=Y, mode = 'markers')]
    layout = dict(title = 'Input data')

    figure_configuration = dict(data=chart_data, layout=layout)
    plotly.offline.plot(figure_configuration, filename="input-data.html")


def createModel():
    # we will pass to the training net an unknown number of elements (so None) of size 1 (the x coordinate)
    input_shape = (1, )

    # we will expect from the network a single output value (the y coordinate)
    output_size = 1

    model = Sequential()
    # 32 neuron layer fully interconnected
    model.add(Dense(32, activation='relu', input_shape=input_shape))

    # 32 neuron layer fully interconnected
    model.add(Dense(32, activation='relu'))

    # toward 1 single output (the sin(x) prediction)
    model.add(Dense(output_size, activation='linear'))

    model.summary()


    return model

def createOptimizer(learning_rate=False):

    if not learning_rate:
        learning_rate = 0.001
    sgd_optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.9)

    return sgd_optimizer

def createAndTrainModel(data, training_epochs=100, learning_rate=False):
    """
    in this step, we build the model and train it
    """

    model = createModel()

    model.compile(loss = keras.losses.mean_absolute_error,
              optimizer=createOptimizer(learning_rate),
              metrics=['accuracy'])


    # separate data in training set and validation set
    # we use 15% of data as validation data
    validation_percentage = .15
    validation_size = int(math.floor(validation_percentage * len(X)))

    x_train = data["in"][:-validation_size]
    y_train = data["out"][:-validation_size]

    x_validation = data["in"][-validation_size:]
    y_validation = data["out"][-validation_size:]

    # this is the real training
    batch_size = 32

    print("Starting training for %d epochs, of %d samples, of wich %d are used as validation, taken in batches of %d elements"
        % (training_epochs, len(X), validation_size, batch_size))
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=training_epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))



    return model


def drawResultChart(models, labels):
    rX = []
    rY = []

    predictions = []
    chart_data = []


    # generate the predictions
    for  index, model in enumerate(models):
        model_predictions = [[],[]]
        for x in xrange(1,150):
            rnd = random.random() * 2
            model_predictions[0].append(rnd)
            res = model.predict([[rnd]])
            model_predictions[1].append( res[0][0] )
        predictions.append(model_predictions)
        chart_data.append (go.Scatter(x=model_predictions[0], y=model_predictions[1], mode = 'markers', name= labels[index]))



    # create the chart layout
    layout = dict(title = 'Output data, 3 networks with different training epochs', showlegend=True)
    fig = tools.make_subplots(rows=len(models), cols=1)
    for index, model in enumerate(models):
        fig.append_trace(chart_data[index],1+index, 1)


    plotly.offline.plot(fig, filename="output-data.html")

    # figure_configuration = dict(data=chart_data[0], layout=layout)
    plotly.offline.plot(fig, filename="output-data.html")

    print("open output-data.html to check the results")

data = generateData(2000)
print("PLEASE READ THE CODE TO PROCEED")
step1()

"""
in the following steps we will train 3 times the same models, each time with different epochs.
"""
epochs = 20
model_20 = createAndTrainModel(data, epochs)

epochs = 50
model_50 = createAndTrainModel(data, epochs)

epochs = 100
model_100 = createAndTrainModel(data, epochs)

drawResultChart([model_20, model_50, model_100], ["20 epochs", "50 epochs", "100 epochs"])

savePath = "sinxApproximation100epochs.h5"
try:
    model_100.save(savePath)
    print("model saved to %s " % savePath)
except Exception as e:
    print("an error occurred while saving the model to %s " % savePath)
    raise e


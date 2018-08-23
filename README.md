

This image contains a pre made environment to work with tensorflow, tensorflow-js and keras.

Frontend
========

To check the frontend simply run the container with:

    docker run -d -p 8080:8080 ccamillo/dockerflow

and open your browser to http://localhost:8080/html/

The frontend area is a simple html page that loads a pre-elaborated model that approximates the value of sin(x)
The contents are in
    - /html
    - /js

See github page to check sources ( https://github.com/stormsson/dockerflow )


Backend
=======

The backend area is simply a training experiment.
The experiment creates a model and then trains it for 20,50,100 epochs and creates an html result page to show the accuracy difference due to the different training times.

The contents are in
    - /py

See github page to check sources ( https://github.com/stormsson/dockerflow )

To train the model run

    docker run --rm -v $(pwd):/workdir/ ccamillo/dockerflow python /static/py/train.py .

this will create a trained model in your work directory.

### Training the model

When the previous command is run, the model will be trained 3 times.
The first time the model is trained for just 20 epochs, so an insufficient amount of epochs to let the model correct its predictions.
By seeing the output-data.html it is possible to see that the generated approximation is extremely rough.


The second time the model is trained for a greater amount of epochs.
Still insufficient though to create an adequate rappresentation of the function we want to approximate.
Checking the results in output-data.html will allow you to see that the model is better than the first one, but still inadequate.

The third and last time, the model is trained for 100 epochs.
The results in output-data.html will show that the approximation is very similar to the original data, therefore the model is usable to approximate the function we wanted to emulate.

The model is the a copy of the one used in the frontend area and already included in this container.

# Converting the model from Keras to TensorFlow.js
In this training we used Keras to generate the model.
To use it in TensorFlow.js a conversion is needed.
Check https://js.tensorflow.org/tutorials/import-keras.html for details.



# Exposed Ports
* TensorBoard exposed on port 6006
* IPython exposed on port 8888
* Python webserver exposed on port 8080



Github
======
https://github.com/stormsson/dockerflow
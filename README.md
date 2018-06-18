

This image contains a pre made environment to work with tensorflow, tensorflow-js and keras.

Frontend
========

To check the frontend simply run the container with:

docker run -d -p 8080:8080 ccamillo/dockerflow

 and open your browser to http://localhost:8080/static/html/

The frontend area is a simple html page that loads a pre-elaborated model that approximates the value of sin(x)
The contents are in
    - /workdir/html
    - /workdir/js



Backend
=======

The backend area is simply a training experiment.
The experiment creates a model and then trains it for 20,50,100 epochs and creates an html result page to show the accuracy difference due to the different training times.

The contents are in
    - /workdir/py

To train the model run

docker run --rm ccamillo/dockerflow python static/py/train.py

# Exposed Ports
* TensorBoard exposed on port 6006
* IPython exposed on port 8888
* Python webserver exposed on port 8080
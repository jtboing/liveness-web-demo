# Liveness Web Demo

This web app processes video stream using OpenCV and determines the liveness; whether the frame represents a real person or a fake. Video is transmitted from browser to server using a Python library AIORTC. Liveness determination is done with a trained neural network using tensorflow.

## How to run

Download all the files and install all the dependencies using

```
pipenv install
```

Run the shell

```
pipenv shell
```

and run the `server.py` file with Python

```
python server.py
```

## Training the Neural Network

The neural network here is already trained. But, it is possible to retrain the neural network. Training should be done when new data (video) is added to the data set. The files to be run in order are:

1. `gather_examples.py`
2. `train.py`

## Implementation status

TO-DO:

- ~~Pass frame label to web page~~ Done. Used data channels
- ~~Optimize liveness determination (currently runs very slow)~~ Partially done. Removed stream display but still triggers @tf.function retracing
- Label only shows real on tests done with real or fake faces.
- Implement WSGI in server using gunicorn
- Create a better model by using web Tensorflow

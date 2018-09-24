# BioSimVis

This is an exploratory project looking to see how some biological phenomena can be incorporated into computer vision methodologies.

# Getting Started

Run training in normal TensorFlow input. Compare with results from modified input using golden ratio algorithm.
The training data set is extracting from the movie "Hannah and her sisters" directed by Woody Allen.
This code is developed in local Ubuntu machine with TensorFlow installed. See Wiki for all software packages used in this deployment.

# Prerequisites

Minimum hardware requirements:

   * Intel i3 processor.
   * 4 GB RAM.
   * For this project, we use Intel i5 core processor and 8 GB RAM.

Software packages:

   * Ubuntu 16.04 or later (64-bit)
   * TensorFlow with pip
   * Anaconda 5.2 For Linux Installer - 64 bits (Python 2.7)
   * PyCham 2018.2

## Installing

Due to the rapid updates in versions in software packages. There will not be step-by-step instructions on how to install the software packages.
For this project, I simply use the default settings for various software package installations.

## Running the initial tests

When the environment is ready. One should be able to run the following testing code.

import tensorflow as tf <br/>
mnist = tf.keras.datasets.mnist  <br/>

(x_train, y_train),(x_test, y_test) = mnist.load_data()  <br/>
x_train, x_test = x_train / 255.0, x_test / 255.0  <br/>

model = tf.keras.models.Sequential([  <br/>
  tf.keras.layers.Flatten(),  <br/>
  tf.keras.layers.Dense(512, activation=tf.nn.relu),  <br/>
  tf.keras.layers.Dropout(0.2),  <br/>
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)  <br/>
])  <br/>
model.compile(optimizer='adam', <br/>
              loss='sparse_categorical_crossentropy', <br/>
              metrics=['accuracy']) <br/>

model.fit(x_train, y_train, epochs=5) <br/>
model.evaluate(x_test, y_test) <br/>


## Sample results

Epoch 1/5
60000/60000 [==============================] - 16s 269us/step - loss: 0.2208 - acc: 0.9358
Epoch 2/5
60000/60000 [==============================] - 16s 268us/step - loss: 0.0977 - acc: 0.9698
Epoch 3/5
60000/60000 [==============================] - 16s 266us/step - loss: 0.0680 - acc: 0.9783
Epoch 4/5
60000/60000 [==============================] - 16s 266us/step - loss: 0.0552 - acc: 0.9821
Epoch 5/5
60000/60000 [==============================] - 16s 267us/step - loss: 0.0432 - acc: 0.9857
10000/10000 [==============================] - 0s 47us/step

[0.06318106037605903, 0.9811]

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.
Versioning

We use SemVer for versioning. For the versions available, see the tags on this repository.
## Authors

    Hoi Hon - Master in Computer Science - Monmouth University
    William Tepfenhart Ph.D - Advisor - Monmouth University

See also the list of contributors who participated in this project.
## License

This project is licensed under the MIT License - see the LICENSE.md file for details
## Acknowledgments

    tbd


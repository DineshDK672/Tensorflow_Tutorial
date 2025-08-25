import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28*28).astype("float32")/255.0

# Sequential API

modelSeq = keras.Sequential(
    [
        keras.Input(shape=(28*28,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

print(modelSeq.summary())
modelSeq.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

modelSeq.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
modelSeq.evaluate(x_test, y_test, batch_size=32, verbose=2)

# Functional API

inputs = keras.Input(shape=(784,))

x = layers.Dense(1024, activation='relu', name='First_layer')(inputs)
x = layers.Dense(512, activation='relu', name='Second_layer')(x)
x = layers.Dense(256, activation='relu', name='Third_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
modelFun = keras.Model(inputs=inputs, outputs=outputs)

print(modelFun.summary())
modelFun.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

modelFun.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
modelFun.evaluate(x_test, y_test, batch_size=32, verbose=2)

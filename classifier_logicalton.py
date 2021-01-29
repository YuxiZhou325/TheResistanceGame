"""
This is a classifier re-write from lab given "classifier_loggerbot.ipnb"

@Yuxi 2020-11-10 13:39:29
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

df0 = pd.read_csv("logs/Logicalton.log",
                  names=["Turn", "Try", "PlayerID", "PlayerName", "MissionsBeenOn", "FailedMissionsBeenOn", "VotedUp0",
                         "VotedUp1", "VotedUp2", "VotedUp3", "VotedUp4", "VotedUp5", "VotedDown0", "VotedDown1",
                         "VotedDown2", "VotedDown3", "VotedDown4", "VotedDown5", "Spy"])
# print(df0.head())
df = df0.query("PlayerName=='Logicalton'")  # filter a pandas data frame very nicely like an SQL query!
# print(df.head())  # print the filtered dataset
# print(df.describe())

x_train = df.values[:, 4:18].astype(np.float32)
# This filters out only the columns we want to use as input vector for our NN.
# Note [:,4:18] only includes columns 4 to 17 inclusive (it does not include column 18)
y_train = df.values[:, 18].astype(np.int32)  # This is our target column.
print(y_train.shape)  # This is just a rank 1 array.
print(y_train[0:6])  # first 6 entries of y.  Should be all 1s and zeros
print(x_train[0:6])  # first 6 rows of x, our input vectors.
num_inputs = x_train.shape[1]  # this works out how many columns there are in x, i.e. how many inputs our network needs.
num_outputs = 2  # Two outputs needed - for "spy" or "not spy".

# Split the dataset into a training data set and a validation dataset.
dataset_size = len(x_train)
train_set_size = int(dataset_size * 0.7)  # choose 70% of the data for training and 30% for validation
x_val, y_val = x_train[train_set_size:], y_train[train_set_size:]
x_train, y_train = x_train[:train_set_size], y_train[:train_set_size]

# Build a keras model:
# Define Sequential model with 3 layers
model = keras.Sequential(name="my_neural_network")

layer1 = layers.Dense(10, activation="tanh", input_shape=(num_inputs,))
model.add(layer1)
layer2 = layers.Dense(10, activation="tanh")
model.add(layer2)
# No activation function here,
# we'll use from_logits=True below which implicitly adds the softmax into the training loss function.
layer3 = layers.Dense(num_outputs)
model.add(layer3)
# just check the NN is the correct shape for our training data, and see what comes out of it:
# print(model(x_train[0:3]))

# Do the usual business for keras training
# It's a classification problem , so we need cross entropy here.
model.compile(
    optimizer=keras.optimizers.Adam(0.001),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']  # TODO @Yuxi 2020-11-11 20:55:07
)

# Do the usual business for keras training
history = model.fit(
    x_train,
    y_train,
    batch_size=50,
    epochs=500,
    validation_data=(x_val, y_val), verbose=1
)

# Plot our training curves. This is always important to see if we've started to overfit or whether
# we could benefit from more training cycles....
# print("\n")
# print("================================================================")
# print(history.history["accuracy"], history.history["val_accuracy"])
# print("\n")

plt.figure(1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Set Accuracy")
plt.legend()
plt.grid()
plt.savefig('training_curves_08.png')
# The following graph should show about 77% accuracy.

# Calculate base rate accuracy
print(y_train.mean())
# This shows us what accuracy we could get if we just guess the same thing all the time.
accuracy_by_turn = []
maximum_turn = df['Turn'].max()
accuracy_metric = tf.keras.metrics.Accuracy()
print("maximum_turn", maximum_turn)

for turn in range(1, maximum_turn + 1):
    df_restricted = df.query(
        'Turn>=' + str(turn))  # Pull out just those rows of the training data corresponding to later turns in the game

    x = df_restricted.values[:, 4:18].astype(np.float32)
    y = df_restricted.values[:, 18].astype(np.int32)
    y_guess = model(x)
    y_guess = tf.argmax(y_guess, axis=1)
    # accuracy=tf.reduce_mean(tf.cast(tf.equal(y,y_guess),tf.float32)) # This formula would also give us the accuracy
    # but this is hand-evaluated.
    accuracy = accuracy_metric(y_guess, y)  # This function calculates accuracy using an in-built keras function.
    accuracy_by_turn.append(accuracy.numpy())  # record the results so we can plot them.

print(tf.range(maximum_turn), accuracy_by_turn)

plt.figure(2)
plt.plot(tf.range(1, 1 + len(accuracy_by_turn)), accuracy_by_turn)
plt.title('Accuracy at identifying whether "Logicalton" is a spy as the game progresses')
plt.xlabel('Turn')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig('accuracy_08.png')

# Finally, save our model so we can build a resistance-bot that plays using this neural network.

# TODO I am making it my own classifier now by training the new log  @Yuxi 2020-11-12 12:03:04
model.save(filepath="C://Users/12826/Documents/Univerity of Essex/MSc/CE811 Game Artificial "
                    "Intelligence/CE811_Labs/Lab1/logicalton_classifier")
# model.save_weights(filepath="C://Users/12826/Documents/Univerity of Essex/MSc/CE811 Game Artificial
# "Intelligence/CE811_Labs/Lab1/loggerbot_classifier_01") # TODO @Yuxi 2020-11-11 20:54:27
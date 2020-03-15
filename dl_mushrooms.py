import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

mushrooms_df = pd.read_csv('mushrooms.csv')

# We have to values in the column 'class', eatable and poisonous denoted 'e','p'.

print(mushrooms_df.groupby (['class','odor']).count())

# We can see that we have only three odor types on eatable mushrooms.
# One overlapping class between the two is the 'n' odor class.
plt.scatter(mushrooms_df['odor'], mushrooms_df['class'])
plt.show()

#Preparing to feed the model with features, X as the input and labels y as the output.
labels = mushrooms_df['class']
features = mushrooms_df.drop(columns=['class'])
y = labels
X = features

# Poison will be equal to 0 and eatable will be equal to 1
y.replace('p', 0, inplace=True)
y.replace('e', 1, inplace=True)

# Use Pandas get_dummies to convert features to values between 0 & 1
X = pd.get_dummies(features)
print(y[0:5])
print(X[0:5])

X = X.values.astype('float32')
y = y.values.astype('float32')

# Dataset split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

# Creating the model.
model = keras.Sequential([keras.layers.Dense(32, input_shape=(117,)),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(2,activation=tf.nn.softmax)])

model.summary()

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])

history = model.fit(X_train, y_train, epochs=7, validation_data=(X_validation, y_validation))

# Checking models performance.
prediction_X = model.predict(X_test)
performance = model.evaluate(X_test, y_test)

print(performance)

# Visualization.
history_dict = history.history
history_dict.keys()

dict_keys=['loss', 'acc', 'val_loss', 'val_acc']

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

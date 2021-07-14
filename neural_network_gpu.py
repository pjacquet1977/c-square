import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

PATH="/scratch/pjacquet/"
Nb_Epochs = 100

def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def get_dataset():
    batch_size = 640
    num_val_samples = 10000
    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )



strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()

# Train the model on all available devices.
train_dataset, val_dataset, test_dataset = get_dataset()
history=model.fit(train_dataset, epochs=Nb_Epochs, validation_data=val_dataset)


#Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig(PATH+"loss_curves.pdf")

# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['sparse_categorical_accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_sparse_categorical_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig(PATH+"accuracy_curves.pdf")

print("The validation loss is ", history.history['val_loss'][Nb_Epochs-1])

# Test the model on all available devices.
scores=model.evaluate(test_dataset)
print("Test accuracy rate:: %.2f%%"%(scores[1]*100))
print("Test error rate: %.2f%%"%(100-scores[1]*100))
print('Test cross-entropy loss:',round(scores[0],2))

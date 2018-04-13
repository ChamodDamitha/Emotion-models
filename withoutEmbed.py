'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation, GRU
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import print_summary
import datetime
import cPickle as pickle
import keras.backend as K

vocab_size = 19851
max_features = 20000
maxSeqLength = 50
maxlen = 50  # cut texts after this number of words (among top max_features most common words)
batch_size = 16
print('Loading data...')

with open('preprocessed/x_train.p', 'rb') as f:
    x_train = pickle.load(f)

with open('preprocessed/y_train.p', 'rb') as f:
    y_train = pickle.load(f)

with open('preprocessed/x_test.p', 'rb') as f:
    x_test = pickle.load(f)

with open('preprocessed/y_test.p', 'rb') as f:
    y_test = pickle.load(f)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

def jaccard_distance(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (sum_ - intersection)
    return jac

logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.0001)
tbCallBack = TensorBoard(log_dir=logdir, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
checkpointer = ModelCheckpoint(filepath='models/weights.hdf5', verbose=1, save_best_only=True)
earlyStoper = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# define baseline model
def baseline_model():
    embedding_vecor_length = 1000

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vecor_length, input_length=maxSeqLength))
    model.add(LSTM(100))
    model.add(Dense(11, activation='relu'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', jaccard_distance])
    # create model
    # model = Sequential()
    # model.add(Dense(8, input_dim=50, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(11, activation='softmax'))
    # # Compile model
    # model.compile(loss='categorical_crossentropy', optimizer='adam',
    #               metrics=['accuracy', jaccard_distance])
    return model

model = baseline_model()
print_summary(model)
print('Train...')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=30,
#           validation_data=(x_test, y_test),
#           callbacks=[reduce_lr,tbCallBack,earlyStoper,checkpointer])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=30,
          validation_data=(x_test, y_test),
          callbacks=[earlyStoper])
model.save('test_seq1.h5')
score, acc, jac_dis = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print('Test Jaccard Distance:', jac_dis)

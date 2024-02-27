"""
Module for building and testing recommendation models
"""
from keras import Sequential
from keras.src.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


def kneighbours_model(train_dataset_x, test_dataset_x, train_dataset_y, test_dataset_y):
    neigh_class = KNeighborsClassifier(n_neighbors=3, weights="distance")
    neigh_class.fit(train_dataset_x, train_dataset_y)

    print(f"\nK-neighbours model accuracy is: {accuracy_score(test_dataset_y, neigh_class.predict(test_dataset_x))}\n")

    return neigh_class


def grad_boost(train_dataset_x, test_dataset_x, train_dataset_y, test_dataset_y):
    gbc = HistGradientBoostingClassifier(learning_rate=0.05, max_features=0.1)
    gbc.fit(train_dataset_x, train_dataset_y)

    print(f"\nGradient boost model accuracy is: {accuracy_score(test_dataset_y, gbc.predict(test_dataset_x))}\n")

    return gbc


def mnn_model(train_dataset_x, test_dataset_x, train_dataset_y, test_dataset_y):
    # Create model
    model = Sequential()
    # Init Input layer with 6 params
    # Init Dense layer with 6 params
    # Init Output(Dense) layer with 1 param
    model.add(Dense(units=6, activation='relu', input_shape=(1, 6)))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile mod
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    model.fit(train_dataset_x.values.reshape(40000, 1, 6), 1 / train_dataset_y.values.reshape(40000, 1, 1),
              epochs=10, batch_size=40000)

    print(model.evaluate(test_dataset_x.values.reshape(10000, 1, 6), 1 / test_dataset_y.values.reshape(10000, 1, 1),
                         batch_size=10000, use_multiprocessing=True))
    return model

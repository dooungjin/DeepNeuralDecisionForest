import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import pickle
import sys
import time

startTime = time.time()

# Hyperparameters
learning_rate = float(sys.argv[1])
batch_size = 1024
num_epochs = 150
num_trees = int(sys.argv[2])
depth = int(sys.argv[3])
used_features_rate = float(sys.argv[4])
# Hyperparameters end

data_path = '/domino/datasets/local/NN_Data/'

fea_list = open ("fea_list.pkl", "rb")
fea_list = pickle.load(fea_list)

p_in_time_train = pd.read_csv(data_path + 'training_set.csv')
p_in_time_valid = pd.read_csv(data_path + 'validation_set.csv')

mean_input = p_in_time_train[fea_list].mean()
p_in_time_train[fea_list] = p_in_time_train[fea_list].fillna(mean_input)
p_in_time_valid[fea_list] = p_in_time_valid[fea_list].fillna(mean_input)

X_train=p_in_time_train[fea_list]
X_valid=p_in_time_valid[fea_list]

y_train=1-p_in_time_train["label"]
y_valid=1-p_in_time_valid["label"]

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

X_train_scaled_df= pd.DataFrame(X_train_scaled, columns=list(X_train.columns))
train_data = X_train_scaled_df.copy()
train_data['ODEFINDPROXY1'] = list(y_train)
train_data.to_csv('train_data.csv', index=False, header=False)

X_valid_scaled_df = pd.DataFrame(X_valid_scaled, columns=list(X_valid.columns))
valid_data = X_valid_scaled_df.copy()
valid_data['ODEFINDPROXY1'] = list(y_valid)
valid_data.to_csv('valid_data.csv', index=False, header=False)

# Declare the necessary variables
train_data_file = "train_data.csv"
valid_data_file = "valid_data.csv"
test_data_file = "test_data.csv"
TARGET_FEATURE_NAME = 'ODEFINDPROXY1'
TARGET_LABELS = [0,1]

#Derive the necessary variables
#No. of classes, assuming last column is dependent variable
num_classes = int(train_data.iloc[:, -1].nunique()/2)
# [batch_size, num classes]
csv_header = list(train_data.columns)   #List of all the columns
feature_names = list(train_data.columns[:-1])  #List of only independent variables
COLUMN_DEFAULTS = [
    [0.0] if feature_name in csv_header  else ["NA"]
    for feature_name in csv_header
]

#Import the data from csv with all the relevant columns
def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=batch_size):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=csv_header,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        #na_value="?",
        shuffle=shuffle,
    ).map(lambda features, target: (features, target))
    return dataset.cache()

# Create one input layer for each feature
def create_model_inputs(FEATURE_NAMES):
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32)
    return inputs

# If there is any categorical feature, it can be encoded
# Since we had only numerical feature, we will skip this step
# We will also exdpand the dimension of the feature inputs
def encode_inputs(inputs):
    encoded_features = []
    for feature_name in inputs:
        encoded_feature = inputs[feature_name]
        #if inputs[feature_name].shape[-1] is "?":
        encoded_feature = tf.expand_dims(encoded_feature, -1)

        encoded_features.append(encoded_feature)

    encoded_features = layers.concatenate(encoded_features)
    return encoded_features

train_dataset = get_dataset_from_csv(
        train_data_file, shuffle=True, batch_size=batch_size
    )

valid_dataset = get_dataset_from_csv(
        valid_data_file, shuffle=True, batch_size=batch_size
    )

class NeuralDecisionTree(keras.Model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth   # Pre-defined depth
        self.num_leaves = 2 ** depth  # No of leaves in the tree
        self.num_classes = num_classes  # No of classes in the dependent variable

        # Create a mask for the randomly selected features.
        # Number of features to be selected for each tree
        # DL: type change from Dimension to int
        num_used_features = int(int(num_features) * used_features_rate)
        # Select "num_used_features" features from the total features
        # DL: type change from Dimension to int
        one_hot = np.eye(int(num_features), dtype="float32")
        sampled_feature_indicies = np.random.choice(
            np.arange(int(num_features)), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]
        #self.used_features_mask = tf.cast(self.used_features_mask, tf.float32)
        # Initialize the weights of the classes in leaves.

        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]
            ),
            dtype="float32",
            trainable=True,
        )

        # Initialize the stochastic routing layer.
        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    def call(self, features):
        batch_size = tf.shape(features)[0]

        # Apply the feature mask to the input features.
        features = tf.matmul(
            features, self.used_features_mask, transpose_b=True
        )
        # Compute the routing probabilities.
        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )
        # Concatenate the routing probabilities with their complements.
        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )
        # Initiate mu, the probablity of a sample reaching a leaf node
        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        # Update probabilities in each level and node.
        # Calculate total final output probability
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                :, begin_idx:end_idx, :
            ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)
        mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        # DL: Change from softmax to sigmoid
        probabilities = keras.activations.sigmoid(self.pi)  # [num_leaves, num_classes]
        outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
        return outputs

class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionForest, self).__init__()
        self.ensemble = []
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes
                )
            )

    def call(self, inputs):
        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, num_classes])

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)
        # Divide the outputs by the ensemble size to get the average.
        outputs /= len(self.ensemble)
        return outputs

def create_forest_model():
    inputs = create_model_inputs(feature_names)
    features = encode_inputs(inputs)
    #features = tf.cast(features, dtype='float32')
    features = layers.BatchNormalization()(features)
    #num_features = 200
    #num_features = features.shape[1].value
    num_features = features.shape[1]

    forest_model =  NeuralDecisionForest(
        num_trees, depth, num_features, used_features_rate, num_classes
    )

    #outputs = forest_model(tf.cast(features, dtype='float32'))
    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

forest_model = create_forest_model()

forest_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        #loss=keras.losses.SparseCategoricalCrossentropy(),
        # DL: Use keras.metrics.AUC(name='val_auc')
        # DL: Change num_classes from 2 to 1
        metrics=[keras.metrics.AUC(name='val_auc')]
        #metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

# DL: Changes from 'val_loss' to 'val_auc'
# DL: To use 'val_auc', the num_classes should be 1 and the nonlinear activation function should be sigmoid (not softmax)
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=5)

print("Fitting Start")
print("Learning Rate:", learning_rate)
print("Num Tree:", num_trees)
print("Depth:", depth)
print("Used Feature Rate:", used_features_rate)
print("\n")
forest_model.fit(train_dataset, validation_data=valid_dataset, epochs=num_epochs, callbacks=[es], verbose=2)
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

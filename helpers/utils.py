import json
import os
from math import sqrt, pow

import pandas as pd
from keras.models import load_model
from tensorflow.lite.python.interpreter import InterpreterWithCustomOps, Interpreter


def read_csv(path):
    data = []
    with open(path, "r") as file:
        for line in file:
            line = line.rstrip('\n')
            line = line.rstrip('\r')
            data.append(line.split(','))
    return data


def load_saved_model_pb(path=None, tf_function=False, load_specs=True):
    if path is None:
        print("Path cannot be None for loading the model")
    model = load_model(path)
    if tf_function:
        # Loading a models as a TF function allows a much faster execution. Warning: cannot modify a models
        # when loaded as tf_function: no conversion, pruning or quantization; just execution
        model = fast_execution_model(model)
    if load_specs:
        specs = json.load(open(os.path.join(os.path.dirname(os.path.dirname(path)), "specs.json")))
        return model, specs
    else:
        return model

def load_saved_model_h5(path=None, model_name="lightweight_baby"):
    if path is None:
        path = os.path.dirname(os.path.abspath(__file__)) + "/models"
    model = load_model(model_name + ".h5")
    specs = json.load(open(path + "/specs.json"))
    return model, specs


def load_saved_tf_lite_model(path=None):
    if path is not None:
        model, input_details, output_details = prepare_tflite_model(path)
        specs = json.load(open(os.path.dirname(path) + "/specs.json"))
        return model, input_details, output_details, specs
    else:
        print("Error: the path of the TF Lite model must be given.")
        exit(-1)


def fast_execution_model(model):
    import tensorflow as tf
    return tf.function(model, input_signature=(tf.TensorSpec(shape=model.input_shape, dtype=tf.float32),))


def prepare_tflite_model(path):
    model = Interpreter(model_path=path)
    # model = InterpreterWithCustomOps(model_path=path)
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    return model, input_details, output_details


def string_to_number(dataset):
    for i in range(len(dataset) - 1):  # skip headers
        for x in range(len(dataset[i + 1])):
            dataset[i + 1][x] = float(dataset[i + 1][x])
    return dataset


def transpose_array(array):
    transpose = []
    for i in zip(*array):
        transpose.append((list(i)))
    return transpose


def remove_unnecessary_data(dataset: list, features):
    headers = dataset[0]
    remove = []
    for h in headers:
        if h not in features:
            remove.append(h)

    for r in remove:
        header_id = headers.index(r)
        for i in range(len(dataset)):
            dataset[i].pop(header_id)

    # Put "exhausts_co2_concentration" in the last column
    transpose = transpose_array(dataset)
    transpose.append(transpose.pop(dataset[0].index("exhausts_co2_concentration")))
    dataset = transpose_array(transpose)

    return string_to_number(dataset)


def load_dataset(path, specs):
    input_features = [feature for feature in specs["input_features"]]
    output = [output for output in specs["output_features"]]
    dataset = remove_unnecessary_data(read_csv(path), input_features + output)
    return dataset

def load_dataset_by_input_list(path, input_features):
    dataframe = pd.read_csv(path)
    input_features.append("exhausts_co2_concentration")
    headers = list(dataframe.columns)
    remove = []
    for h in headers:
        if h not in input_features:
            remove.append(h)
    dataframe.drop(remove, axis=1, inplace=True)
    return dataframe


def mean_and_std(array):
    sum = 0
    for i in array:
        sum += i
    mean = sum / len(array)

    std = 0
    for i in array:
        std += pow((i - mean), 2)
    std = sqrt(std - len(array))

    return mean, std


def get_stats_from_specs(specs):
    mean_dict, std_dict = {}, {}
    min_dict, max_dict = {}, {}
    for input in specs["input_features"]:
        mean_dict[input] = specs["input_features"][input]["mean"]
        std_dict[input] = specs["input_features"][input]["std"]
        min_dict[input] = specs["input_features"][input]["min"]
        max_dict[input] = specs["input_features"][input]["max"]
    for output in specs["output_features"]:
        mean_dict[output] = specs["output_features"][output]["mean"]
        std_dict[output] = specs["output_features"][output]["std"]
        min_dict[output] = specs["output_features"][output]["min"]
        max_dict[output] = specs["output_features"][output]["max"]
    return mean_dict, std_dict, min_dict, max_dict


def preprocess_dataset(dataset, specs):
    mean_dict, std_dict, min_dict, max_dict = get_stats_from_specs(specs)

    # Convert values based on training dataset values
    conversion_type = specs["dataset_preprocessing"]
    if conversion_type not in ["normalise", "standarise"]:
        print("Conversion type not allowed")
        exit(-1)

    for feature in dataset.columns:
        if conversion_type == "normalise":
            dataset[feature] = (dataset[feature] - min_dict[feature]) / (max_dict[feature] - min_dict[feature])
        elif conversion_type == "standarise":
            dataset[feature] = (dataset[feature] - mean_dict[feature]) / std_dict[feature]

    escalation_dict = {
        "mean_dict": mean_dict,
        "std_dict": std_dict,
        "min_dict": min_dict,
        "max_dict": max_dict
    }

    return dataset, escalation_dict


def split(dataset, train_size=0.8):
    # split into train and test sets
    training_size = int(len(dataset) * train_size)
    train = dataset[:training_size]
    test = dataset[training_size:]
    return test, train


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        seq_x = []

        # check if we are beyond the dataset
        if i + n_steps >= len(sequences):
            break

        for v in range(n_steps, 0, -1):
            seq_x.append(sequences[i + v - 1][:-1])

        X.append(seq_x)
        y.append(sequences[i + (n_steps - 1)][-1])
    return X, y


def prepare_dataset_timeseries(dataset, timesteps):
    assert timesteps > 0

    test, train = split(dataset)

    # Split them and shape them in arrays like [samples, timesteps, features]
    X_train, y_train = split_sequences(train, timesteps)
    X_test, y_test = split_sequences(test, timesteps)

    return X_train, X_test, y_train, y_test


def prepare_dataset_supervised(dataset):
    test, train = split(dataset)

    X_train, y_train = [], []
    for x in range(len(train)):
        X_train.append(train[x][:-1])
        y_train.append(train[x][-1])
    X_test, y_test = [], []
    for x in range(len(test)):
        X_test.append(test[x][:-1])
        y_test.append(test[x][-1])

    return X_train, X_test, y_train, y_test


def get_data_original_scale(y_test, y_predict, escalation_dict, specs):
    if specs["dataset_preprocessing"] == "standarise":
        avg = escalation_dict["mean"]["exhausts_co2_concentration"]
        std = escalation_dict["std_dict"]["exhausts_co2_concentration"]
        y_test = (y_test * std) + avg
        y_predict = (y_predict * std) + avg
    elif specs["dataset_preprocessing"] == "normalise":
        max = escalation_dict["max_dict"]["exhausts_co2_concentration"]
        min = escalation_dict["min_dict"]["exhausts_co2_concentration"]
        y_test = y_test * (max - min) + min
        y_predict = y_predict * (max - min) + min
    return y_test, y_predict


def flatten_array(array):
    flat_array = []
    for i in range(len(array)):
        new_list = []
        for y in array[i]:
            for z in y:
                new_list.append(z)
        flat_array.append([new_list])

    return flat_array

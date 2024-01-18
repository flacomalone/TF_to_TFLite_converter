from math import sqrt

import numpy as np

from helpers.utils import get_data_original_scale, prepare_tflite_model
import tensorflow as tf


def verify_mae(y_test, y_predict, verbose):
    sum = 0
    for i in range(len(y_test)):
        sum += abs(y_test[i] - y_predict[i])
    testScore = sum / len(y_test)
    if verbose:
        print('MAE: %.4f' % testScore)
    return testScore


def verify_mse(y_test, y_predict, verbose):
    summation = 0
    n = len(y_test)
    for i in range(0, n):
        difference = y_test[i] - y_predict[i]
        squared_difference = difference ** 2
        summation = summation + squared_difference
    testScore = summation / n
    if verbose:
        print('MSE: %.4f' % testScore)
    return testScore


def verify_R2(y_test, y_predict, verbose):
    ss_t = 0  # total sum of squares
    ss_r = 0  # total sum of square of residuals
    mean_predict = calculateMean(y_predict)
    for i in range(len(y_test)):  # val_count represents the no.of input x values
        ss_t += (y_test[i] - mean_predict) ** 2
        ss_r += (y_test[i] - y_predict[i]) ** 2
    testScore = 1 - (ss_r / ss_t)
    if verbose:
        print('R2: %.4f' % testScore)
    return testScore


def verify_rmse(y_test, y_predict, verbose):
    testScore = sqrt(verify_mse(y_test, y_predict, False))
    if verbose:
        print('RMSE: %.4f' % testScore)
    return testScore


def calculateMean(array: list):
    sum = 0
    for i in range(len(array)):
        sum += array[i]
    return sum/len(array)


def getScores(y_test, y_predict, verbose=True):
    results = {"R2": verify_R2(y_test, y_predict, verbose),
               "MAE": verify_mae(y_test, y_predict, verbose),
               "MSE": verify_mse(y_test, y_predict, verbose),
               "RMSE": verify_rmse(y_test, y_predict, verbose)}
    return results


def validate_tf_model(model, X_test, y_test, escalation_dict, specs, scale_back=True):
    y_predict = np.zeros(shape=len(X_test))
    for i in range(len(X_test)):
        X = np.array(X_test[i]).reshape(1, specs["timesteps"], len(specs["input_features"]))
        prediction = model(X)
        y_predict[i] = (float(prediction[0]))

    if scale_back:
        y_test, y_predict = get_data_original_scale(y_test, y_predict, escalation_dict, specs)

    return y_test, y_predict


def validate_tf_lite_model(model_path, X_test, y_test, escalation_dict, specs, scale_back=True):
    model, input_details, output_details = prepare_tflite_model(model_path)
    input_scale, input_zero_point = input_details[0]["quantization"]

    if (input_scale, input_zero_point) == (0.0, 0):
        quantize_input = False
    else:
        quantize_input = True

    output_scale, output_zero_point = output_details[0]["quantization"]
    if (output_scale, output_zero_point) == (0.0, 0):
        quantize_output = False
    else:
        quantize_output = True

    # model_input_type = specs["data_type"]
    model_input_type = tf.float32

    y_predict = np.zeros(shape=len(X_test))
    for i in range(len(X_test)):
        X = np.array(X_test[i]).reshape(1, specs["timesteps"], len(specs["input_features"]))

        if quantize_input:
            X = X / input_scale + input_zero_point
            X = X.astype(model_input_type)

        model.set_tensor(input_details[0]["index"], X)
        model.invoke()
        output = model.get_tensor(output_details[0]["index"])[0][0]

        if quantize_output:
            output = (output - output_zero_point) * output_scale
            output = output.astype(np.float32)

        y_predict[i] = output

    if scale_back:
        y_test, y_predict = get_data_original_scale(y_test, y_predict, escalation_dict, specs)

    return y_test, y_predict

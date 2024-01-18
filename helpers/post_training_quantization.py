import numpy as np
import tensorflow as tf
from keras import Sequential

from helpers.evaluations import verify_R2, validate_tf_lite_model
from helpers.pruning import prune_model
from helpers.utils import (preprocess_dataset, load_saved_model_pb, prepare_dataset_timeseries,
                           load_dataset_by_input_list)
import tensorflow_model_optimization as tfmot

QUANTIZATION_OPTIONS = [
    "integer_with_float_fallback",
    "integer_with_8bits_weights_and_16bit_activations",
    "int8",
    "float16",
    "None"
]


def prepare_input_layer(model: Sequential, specs: dict):
    fixed_input = tf.keras.layers.Input(shape=[specs["timesteps"], len(specs["input_features"])],
                                        batch_size=1,
                                        dtype=model.inputs[0].dtype,
                                        name="fixed_input")
    fixed_output = model(fixed_input)
    run_model = tf.keras.models.Model(fixed_input, fixed_output)
    return run_model


def post_training_quantization(model, X, specs, pruning, quantization, save=False, export_dir=None):
    def representative_dataset_gen():
        for data in X:
            yield [tf.dtypes.cast(data.reshape(1, specs["timesteps"], len(specs["input_features"])), tf.float32)]

    # This is step is critical
    # It will worsen the accuracy when run on PC, but will allow to execute the model in an MCU
    model = prepare_input_layer(model, specs)

    if pruning:
        model = tfmot.sparsity.keras.strip_pruning(model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = representative_dataset_gen

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if pruning:
        converter.optimizations += [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]

    # In order to fully integer quantize a model, but use float operators when they don't have an integer
    # implementation (to ensure conversion occurs smoothly), we use a representative dataset to calibrate model
    # input, activations and model output
    # Warning: This method won't be compatible with integer only devices (such as 8-bit microcontrollers)
    # and accelerators (such as the Coral Edge TPU) because the input and output still remain float in
    # order to have the same interface as the original float only model.
    if quantization == "integer_with_float_fallback":
        converter.representative_dataset = representative_dataset_gen
    elif quantization == "integer_with_8bits_weights_and_16bit_activations":
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        # If 16x8 quantization is not supported for some operators in the model, then the model
        # still can be quantized, but unsupported operators kept in float. For that, uncomment line below
        # converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS)
    elif quantization == "int8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif quantization == "float16":
        converter.inference_input_type = tf.float16
        converter.inference_output_type = tf.float16
        # Many ops do not yet support 16-long float, so must use TF Lite built-ins
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    elif quantization == "None":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    # Troubleshooting: play with these three flags to see if it solves something

    # experimental_new_converter: Experimental flag, subject to change. Enables MLIR-based conversion. (default True)
    converter.experimental_new_converter = True
    converter._experimental_lower_tensor_list_ops = False
    # converter.allow_custom_ops: Boolean indicating whether to allow custom operations. When False, any unknown
    # operation is an error. When True, custom ops are created for any op that is unknown. The developer needs to
    # provide these to the TensorFlow Lite runtime with a custom resolver. (default False)
    converter.allow_custom_ops = False

    # Convert the model to TF Lite
    tf_model = converter.convert()

    # In order to run a TF Lite model it is necessary to save it first as such and then load it from storage
    if save:
        if export_dir is not None:
            print("Saving model in: " + export_dir)
            if pruning:
                open(export_dir + "/post_training_quantization_pruned.tflite", "wb").write(tf_model)
            else:
                open(export_dir + "/post_training_quantization.tflite", "wb").write(tf_model)
        else:
            print("Error: export directory argument must be passed when exporting the model.")
            exit(-1)

    return tf_model


def ptq():
    input_dir = "../models/lightweight_baby_tests"

    # load original models and models specs
    model, specs = load_saved_model_pb(path=input_dir + "/tf_model")

    # load dataset
    dataset = load_dataset_by_input_list(input_dir + "/dataset/combined_dataset.csv",
                                         [f for f in specs["input_features"].keys()])
    # dataset preprocessing
    dataset, escalation_dict = preprocess_dataset(dataset, specs)

    # Get timeseries-shaped dataset
    X_train, X_test, y_train, y_test = prepare_dataset_timeseries(dataset.values, specs["timesteps"])
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    export_path = "../models/lightweight_baby_tests/export"
    pruning = True
    if pruning:
        # prune model
        # model = prune_model(model, specs, X_train, y_train, save=True, show_sparsity=True, export_dir=export_path)
        model = load_saved_model_pb(path="../models/lightweight_baby_tests/export/pruned_pb", load_specs=False)

    # Quantize model
    # post_training_quantization(model, np.concatenate((X_train, X_test), axis=0), specs, pruning=pruning,
    #                            quantization="None", save=True, export_dir=export_path)
    post_training_quantization(model, X_test, specs, pruning=pruning,
                               quantization="None", save=True, export_dir=export_path)

    if pruning:
        tf_lite_model_path = export_path + "/post_training_quantization_pruned.tflite"
    else:
        tf_lite_model_path = export_path + "/post_training_quantization.tflite"
    y_test, y_predict = validate_tf_lite_model(model_path=tf_lite_model_path, X_test=X_test, y_test=y_test,
                                               escalation_dict=escalation_dict, specs=specs, scale_back=False)

    print("Model's accuracy (R2):", float(verify_R2(y_test, y_predict, verbose=False)))


if __name__ == "__main__":
    ptq()

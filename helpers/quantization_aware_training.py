import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import keras.src.layers.rnn
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import LastValueQuantizer, \
    MovingAverageQuantizer

from helpers.evaluations import validate_tf_lite_model, verify_R2
from helpers.pruning import prune_model
from helpers.utils import (load_saved_model_pb, preprocess_dataset, prepare_dataset_timeseries,
                           load_dataset_by_input_list)

QAT_EPOCH = 50
QAT_BATCH_SIZE = 32


class CustomLSTMQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [
            (layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


def get_q_aware_mode(model, specs):
    # Register the custom quantization configuration within a custom object scope
    tf.keras.utils.custom_object_scope({"CustomLSTMQuantizeConfig": CustomLSTMQuantizeConfig})

    def apply_custom_quantization(layer):
        """
        In this case, LSTM layers are quantized differently due to TF requirements
        """
        if layer.__class__ is keras.src.layers.rnn.lstm.LSTM:
            return tfmot.quantization.keras.quantize_annotate_layer(to_annotate=layer,
                                                                    quantize_config=CustomLSTMQuantizeConfig())
        else:
            return tfmot.quantization.keras.quantize_annotate_layer(to_annotate=layer)

    cloned_model = tf.keras.models.clone_model(model, clone_function=apply_custom_quantization)
    return cloned_model


def quantization_aware_training(model, X_train, y_train, specs, pruning=False, save=True,
                                export_dir=None):
    model = get_q_aware_mode(model, specs)

    if pruning:
        model = tfmot.sparsity.keras.strip_pruning(model)

    quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(model)
    model = tfmot.quantization.keras.quantize_apply(quant_aware_annotate_model,
                                                    tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())

    # quantize_model requires a recompile.
    opt = eval(f'keras.optimizers.{specs["optimiser"]["name"]}')()
    optimiser = opt.from_config({k: v for k, v in specs["optimiser"].items() if hasattr(specs["optimiser"], k)})
    model.compile(optimizer=optimiser, loss=specs["loss"])

    # Re-train quantized model
    model.fit(X_train, y_train, batch_size=QAT_BATCH_SIZE, epochs=QAT_EPOCH, validation_split=0.2)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]

    # # Troubleshooting: play with these three flags to see if it solves something
    #
    # # experimental_new_converter: Experimental flag, subject to change. Enables MLIR-based conversion. (default True)
    # converter.experimental_new_converter = True  # This must be True
    # converter._experimental_lower_tensor_list_ops = False
    # # converter.allow_custom_ops: Boolean indicating whether to allow custom operations. When False, any unknown
    # # operation is an error. When True, custom ops are created for any op that is unknown. The developer needs to
    # # provide these to the TensorFlow Lite runtime with a custom resolver. (default False)
    # converter.allow_custom_ops = True

    tf_model = converter.convert()

    # In order to run a TF Lite model it is necessary to save it first as such and then load it from storage
    if save:
        if export_dir is not None:
            print("Saving model in: " + export_dir)
            if pruning:
                open(export_dir + "/quantization_aware_training_pruned.tflite", "wb").write(tf_model)
            else:
                open(export_dir + "/quantization_aware_training.tflite", "wb").write(tf_model)
        else:
            print("Error: export directory argument must be passed when exporting the model.")
            exit(-1)

    return tf_model


def qat():
    input_dir = "../models/lightweight_baby_tests"

    # load original models and models specs
    model, specs = load_saved_model_pb(path=input_dir)

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

    # prune model
    model = prune_model(model, specs, X_train, y_train, save=False, show_sparsity=True)

    export_dir = "../models/lightweight_baby_tests/"
    quantization_aware_training(model, X_train, y_train, specs, pruning=True, quantization="None", save=True,
                                export_dir=export_dir)
    tf_lite_model_path = export_dir + "/quantization_aware_training_pruned.tflite"
    y_test, y_predict = validate_tf_lite_model(model_path=tf_lite_model_path, X_test=X_test, y_test=y_test,
                                               escalation_dict=escalation_dict, specs=specs, scale_back=False)

    print("Model's accuracy (R2):", float(verify_R2(y_test, y_predict, verbose=False)))


if __name__ == "__main__":
    qat()

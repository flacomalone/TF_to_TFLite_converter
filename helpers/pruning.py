import os
import numpy as np
import tensorflow_model_optimization as tfmot
from keras.models import save_model
import keras.optimizers  # needed to check optimisers in specs

PRUNING_INITIAL_SPARSITY = 0.0
PRUNING_FINAL_SPARSITY = 0.75
# PRUNING_EPOCHS = 12
PRUNING_EPOCHS = 1
PRUNING_BATCH_SIZE = 32


def print_model_weights_sparsity(model):
    print("Showing sparsity of each of the layers of the model:")
    for layer in model.layers:
        if isinstance(layer, keras.layers.Wrapper):
            weights = layer.trainable_weights
        else:
            weights = layer.weights
        for weight in weights:
            # ignore auxiliary quantization weights
            if "quantize_layer" in weight.name:
                continue
            weight_size = weight.numpy().size
            zero_num = np.count_nonzero(weight == 0)
            print(
                f"\t{weight.name}: {zero_num/weight_size:.2%} sparsity ",
                f"\t({zero_num}/{weight_size})",
            )


def prune_model(model, specs, X_train, y_train, save=True, export_dir=None, show_sparsity=False):
    if save and export_dir is None:
        print("If pruned models is to be saved, then an export directory should be given")
        exit(-1)

    num_iterations_per_epoch = len(X_train) / specs["batch_size"]
    end_step = np.ceil(num_iterations_per_epoch * PRUNING_EPOCHS)
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=PRUNING_INITIAL_SPARSITY,
                                                                 final_sparsity=PRUNING_FINAL_SPARSITY,
                                                                 begin_step=0,
                                                                 end_step=end_step),
    }
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    opt = eval(f'keras.optimizers.{specs["optimiser"]["name"]}')()
    optimiser = opt.from_config({k: v for k, v in specs["optimiser"].items() if hasattr(specs["optimiser"], k)})

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    model_for_pruning.compile(
        loss=specs["loss"],
        optimizer=optimiser
    )

    model_for_pruning.fit(
        X_train,
        y_train,
        callbacks=callbacks,
        epochs=PRUNING_EPOCHS,
        batch_size=PRUNING_BATCH_SIZE,
        validation_split=0.05,
        verbose=2
    )

    if show_sparsity:
        print_model_weights_sparsity(model)

    if save:
        if not os.path.exists(os.path.dirname(export_dir)):
            os.makedirs(os.path.dirname(export_dir))
        save_model(model_for_pruning, filepath=export_dir + "/pruned_pb", save_format="tf")  # .pb format
        # save_model(model_for_pruning, filepath=export_dir + "/pruned.h5")  # .h5 format

    return model

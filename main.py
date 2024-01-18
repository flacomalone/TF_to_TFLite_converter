import argparse
import os

import numpy as np

from helpers.add_quantization_details_specs import add_quantization_details
from helpers.evaluations import verify_R2, validate_tf_model, validate_tf_lite_model
from helpers.plot_results import plot_colormap
from helpers.post_training_quantization import post_training_quantization, QUANTIZATION_OPTIONS
from helpers.pruning import prune_model
from helpers.quantization_aware_training import quantization_aware_training
from helpers.utils import (load_saved_model_pb, preprocess_dataset, prepare_dataset_timeseries,
                           load_dataset_by_input_list)


def main(args):
    # load original models and models specs
    model, specs = load_saved_model_pb(path=args.input_dir)

    # load dataset
    dataset = load_dataset_by_input_list(os.path.dirname(os.path.dirname(args.input_dir)) +
                                         "/dataset/combined_dataset.csv", [f for f in specs["input_features"].keys()])
    # dataset preprocessing
    print("Loading and processing dataset...")
    dataset, escalation_dict = preprocess_dataset(dataset, specs)

    if args.export_dir is not None:
        export_path = args.export_dir
    else:
        export_path = os.path.dirname(os.path.dirname(args.input_dir)) + "/export"
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Get timeseries-shaped dataset
    X_train, X_test, y_train, y_test = prepare_dataset_timeseries(dataset.values, specs["timesteps"])
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    test_original = False
    if test_original:
        print("Testing original model...")
        _, y_predict_original_model = validate_tf_model(model, X_test, y_test, escalation_dict, specs, scale_back=False)
        print("Original model's accuracy (R2):", float(verify_R2(y_test, y_predict_original_model, verbose=False)))
        if args.plot_results:
            plot_colormap(y_test, y_predict_original_model)

    # You have to respect this order:
    #   1. pruning (optional)
    #   2. either one of these 2 (as conversion is part of quantization)
    #       2.1. quantization-aware training (QAT)
    #       2.2 post-training quantization (PTQ)

    # 1. prune models
    if args.prune:
        print("Pruning model...")
        model = prune_model(model, specs, X_train, y_train, save=True, export_dir=export_path, show_sparsity=True)
        test_pruned = False
        if test_pruned:
            print("Testing pruned model...")
            _, y_predict_pruned_model = validate_tf_model(model, X_test, y_test, escalation_dict, specs,
                                                          scale_back=False)
            print("Pruned model's accuracy (R2):", float(verify_R2(y_test, y_predict_pruned_model, verbose=False)))
            if args.plot_results:
                plot_colormap(y_test, y_predict_pruned_model)

    # 2.1 Quantization-aware training (+ conversion)
    if args.QAT:
        if not args.PTQ:
            print("Performing quantization-aware training and model conversion...")
            save_model = True
            quantization_aware_training(model, np.concatenate((X_train, X_test), axis=0), specs, args.prune,
                                        save=save_model, export_dir=export_path)

            # Add quantization details to specs
            if args.prune:
                tf_lite_model_path = export_path + "/quantization_aware_training_pruned.tflite"
            else:
                tf_lite_model_path = export_path + "/quantization_aware_training.tflite"
            add_quantization_details(model_path=tf_lite_model_path,
                                     specs_path=os.path.dirname(os.path.dirname(args.input_dir)) + "/specs.json")
            test_qat = True
            if test_qat:
                if save_model:
                    _, y_predict_qat = validate_tf_lite_model(model_path=tf_lite_model_path, X_test=X_test,
                                                              y_test=y_test, escalation_dict=escalation_dict,
                                                              specs=specs, scale_back=False)
                    print("Quantization-aware trained model's accuracy (R2):", float(verify_R2(y_test, y_predict_qat,
                                                                                  verbose=False)))
                    if args.plot_results:
                        plot_colormap(y_test, y_predict_qat)
                else:
                    print("You have to select option 'save'=True to save the TF Lite and load it from disk if you"
                          " want to evaluate it")
        else:
            print("Error: you have to choose either --PTQ or --QAT")
            exit(-1)

    # 2.2 Post-training model quantization (+ conversion)
    elif args.PTQ:
        if not args.QAT:
            print("Performing post-training quantization and model conversion...")
            save_model = True
            post_training_quantization(model, np.concatenate((X_train, X_test), axis=0), specs, args.prune,
                                       quantization=args.quantization_type, save=save_model, export_dir=export_path)

            # Add quantization details to specs
            if args.prune:
                tf_lite_model_path = export_path + "/post_training_quantization_pruned.tflite"
            else:
                tf_lite_model_path = export_path + "/post_training_quantization.tflite"
            add_quantization_details(model_path=tf_lite_model_path,
                                     specs_path=os.path.dirname(os.path.dirname(args.input_dir)) + "/specs.json")
            test_ptq = True
            if test_ptq:
                if save_model:
                    _, y_predict_ptq = validate_tf_lite_model(model_path=tf_lite_model_path, X_test=X_test,
                                                              y_test=y_test, escalation_dict=escalation_dict,
                                                              specs=specs, scale_back=False)
                    print("Post-training quantized model's accuracy (R2):", float(verify_R2(y_test, y_predict_ptq,
                                                                                  verbose=False)))
                    if args.plot_results:
                        plot_colormap(y_test, y_predict_ptq)
                else:
                    print("You have to select option 'save'=True to save the TF Lite and load it from disk if you"
                          " want to evaluate it")
        else:
            print("Error: you have to choose either --PTQ or --QAT")
            exit(-1)

    # Export to C array
    if args.e:
        print("Exporting final C array model in " + export_path)
        if args.QAT:
            name = "quantization_aware_training"
        elif args.PTQ:
            name = "post_training_quantization"
        if args.prune:
            name += "_pruned"
        # Convert to C-array
        os.system("xxd -i ./" + export_path + "/" + name + ".tflite > " + export_path + "/" + name + ".cc")
        # Convert this C-array into a Python object (this is the file that runs in an MCU micro-python code)
        os.system("./helpers/cc_to_python.sh ./" + export_path + "/" + name + ".cc ")
        print("Final models have been exported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Input directory containing model data")
    parser.add_argument("-e", action="store_true", help="Export final C-array model")
    parser.add_argument("--export_dir", type=str, help="Output file directory for exported model", default=None)
    parser.add_argument("-p", "--prune", action="store_true", help="Whether to prune the model")
    parser.add_argument("-q", "--quantization_type", type=str, choices=QUANTIZATION_OPTIONS, default="None",
                        help="Select which type of quantization to do for model conversion (default is None)")
    parser.add_argument("--PTQ", action="store_true", help="Enable Post-Training Quantization (PTQ)")
    parser.add_argument("--QAT", action="store_true", help="Enable Quantization-Aware Training (QAT)")
    parser.add_argument("--plot_results", action="store_true", help="Whether to plot the accuracy achieved"
                                                                    "by each of the generated models.", default=False)
    args = parser.parse_args()
    main(args)
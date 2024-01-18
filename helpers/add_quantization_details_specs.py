import json
import os

from helpers.utils import prepare_tflite_model


def add_quantization_details(model_path, specs_path):
    model, input_details, output_details = prepare_tflite_model(model_path)
    specs = json.load(open(specs_path))
    specs["quantization"] = {
        "input": {
            "scale": input_details[0]["quantization"][0],
            "zero_point": input_details[0]["quantization"][1]
        },
        "output": {
            "scale": output_details[0]["quantization"][0],
            "zero_point": output_details[0]["quantization"][1]
        }
    }

    with open(specs_path, mode="w") as model_spec:
        model_spec.write(json.dumps(specs, indent=4))

#!/bin/bash

EXPORT_ROUTE=$1

# Take the content of the file from line 2 to n-2
model=$(tail -n +2 $EXPORT_ROUTE | head -n -2)

# Take the len of the model
len=$(awk '{print $NF}' $EXPORT_ROUTE | tail -1)

# Remove semi-colon
len=${len%?}

# Append the text "converted_model" to the variable
text="converted_model = bytearray([\n$model])\nconverted_model_len = $len"

# Write the result
echo -e $text > "${EXPORT_ROUTE%??}"py
#!/bin/bash

# Loop through each line in the .env file
while IFS='=' read -r key value; do
  # Check if the variable is not already set
  if [[ -z "${!key}" ]]; then
    export "$key=$value"
  fi
done < .env

exec bash
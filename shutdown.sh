#!/bin/bash

MY_PROGRAM="Python" # For example, "apache2" or "nginx"
MY_USER="gordonliu"
CHECKPOINT="/Users/gordonliu/Documents/ml_projects/LightForker-2/src/checkpoints/t_20241017_200614_e_6"
BUCKET_NAME="lf_checkpoints" # For example, "my-checkpoint-files" (without gs://)

echo "Shutting down!  Seeing if ${MY_PROGRAM} is running."

# Find the newest copy of $MY_PROGRAM
PID="$(pgrep -n "$MY_PROGRAM")"

if [[ "$?" -ne 0 ]]; then
  echo "${MY_PROGRAM} not running, shutting down immediately."
  exit 0
fi

echo "Sending SIGINT to $PID"
kill -2 "$PID"

# Portable waitpid equivalent
while kill -0 "$PID"; do
   sleep 1
done

echo "$PID is done, copying ${CHECKPOINT} to gs://${BUCKET_NAME} as ${MY_USER}"

su "${MY_USER}" -c "gcloud storage cp $CHECKPOINT gs://${BUCKET_NAME}/"

echo "Done uploading, shutting down."

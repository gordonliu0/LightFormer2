#!/bin/bash

MY_PROGRAM="Python"
MY_USER="gordonliu"
SOURCE_DIRECTORY="/Users/gordonliu/Documents/ml_projects/LightFormer2"
BUCKET_NAME="lf_checkpoints" # For example, "my-checkpoint-files" (without gs://)
EXCLUDE_REGEX=".*\.DS_Store$|\.venv\/.*$|\.vscode\/.*$|.*\/__pycache__\/.*$|\.gitignore$|\.git\/.*$"
# ignore all ds_store, all .venv, all .vscode, all pycaches, all .gitignore, and all .git.

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

echo "$PID is done, copying ${SOURCE_DIRECTORY} to gs://${BUCKET_NAME} as ${MY_USER}"

# su "${MY_USER}" -c "gcloud storage rsync $SOURCE_DIRECTORY gs://${BUCKET_NAME} --recursive --delete-unmatched-destination-objects --exclude=\".*\/\.DS_Store$|.*\/\.venv\/.*$|.*\/\.vscode\/.*$|.*\/__pycache__\/.*$|.*\/\.gitignore$\""
su "${MY_USER}" -c "gcloud storage rsync ${SOURCE_DIRECTORY} gs://${BUCKET_NAME} --delete-unmatched-destination-objects --recursive --exclude=\"${EXCLUDE_REGEX}\""

echo "Done uploading, shutting down."


#!/bin/bash

if ! ssh-add -l ; then
    echo ssh agent does not have any identities loaded, will not be able to build
    echo add them by running ssh-add on your local machine, or on the remote if you have keys there
    echo you may also need to restart vs code and the remote server for this to work
    exit 1
fi

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
MODEL_PATH=$(yq -r .storage.model_path $SCRIPT_PATH/config.yml)
cp -r $MODEL_PATH $SCRIPT_PATH/models
podman build --format docker -t logo . --network host --build-arg SSH_AUTH_SOCK=$SSH_AUTH_SOCK --volume "${SSH_AUTH_SOCK}:${SSH_AUTH_SOCK}"
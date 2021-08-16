#!/bin/bash -exu

docker run \
    -v $PWD:/iris \
    -it \
    --rm \
    -w /iris \
    --gpus all \
    iris:latest \
    "$@"

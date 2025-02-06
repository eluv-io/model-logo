#!/bin/bash

set -x

rm -rf test_output/
mkdir test_output

[ "$ELV_MODEL_TEST_GPU_TO_USE" != "" ] || ELV_MODEL_TEST_GPU_TO_USE=0

## this model doesn't use cache yet; break if we do so at least we know
## ideally all models don't fetch anything from HF at runtime
mkdir -p .cache-ro

podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/test_output:/elv/tags --volume=$(pwd)/.cache-ro:/root/.cache:ro --network host --device nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE logo test/1.mp4

ex=$?

cd test_output
find

exit $ex



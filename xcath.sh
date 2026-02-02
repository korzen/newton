#!/bin/bash

uv run --extra examples ./newton/examples/cosserat_codex/xcath.py \
    --dll-path ./unity_ref/libDefKitAdv.so \
    --num-points 256 \
    --particle-mass 1.0 \
    --particle-radius 0.02 \
    --use-cuda-graph \
    --rod-solvers warp \
    --rod-count 1 \
    --num-envs 8 8 \
    --env-offset 10.0 5.0 \
    "$@"  # Pass any additional arguments
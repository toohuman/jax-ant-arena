#!/bin/bash

# Sync multirun directory from remote server
rsync -avz michael@kodama:projects/jax-ant-arena/multirun/ multirun

echo "Sync completed!"
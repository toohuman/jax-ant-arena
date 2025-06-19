#!/bin/bash

# Sync multirun directory from remote server
rsync -avz --include ".*" michael@kodama:/data/michael/jax-ant-arena/multirun/ multirun

echo "Sync completed!"
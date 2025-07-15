#!/bin/bash

# Sync multirun directory from remote server
rsync -avz --include ".*" michael@kodama:/data/michael/ants/multirun/ multirun

echo "Sync completed!"
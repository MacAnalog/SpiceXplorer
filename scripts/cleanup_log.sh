#!/bin/bash

# 1. Delete all .log files in the local 'log' directory
# The -f flag suppresses errors if the directory is already empty
find . -type f -name "*.log" -delete

# 2. Recursively find and delete all 'temp_spice_out' directories
# .        -> Start searching in the current directory
# -type d  -> Look only for directories
# -name    -> Look for directories named "temp_spice_out"
# -exec    -> Execute the following command on every match found
find . -type d -name "temp_spice_out" -exec rm -rf {} +

echo "Cleanup complete."
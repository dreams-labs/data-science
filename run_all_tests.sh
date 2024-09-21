#!/bin/bash
set -e

# Copy prod configs to test
for file in config/*; do
    cp "$file" "tests/test_config/test_$(basename "$file")"
done

# Change to tests folder and run pytest
pytest

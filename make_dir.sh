#!/usr/bin/env bash

# Create directory structure
mkdir -p src/functions
mkdir -p data/raw
mkdir data/processed
mkdir -p output/reports
mkdir output/figs

# If README.md does not exist, create it.
test -f README.md || touch README.md

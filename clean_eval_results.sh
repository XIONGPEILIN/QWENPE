#!/bin/bash

BASE_DIR="pico_test"
FILE_TO_REMOVE="siglip2_qwen_eval.csv"

echo "Cleaning up generated evaluation files ($FILE_TO_REMOVE) in $BASE_DIR..."

found_count=0
removed_count=0

# Find all occurrences
files=$(find "$BASE_DIR" -name "$FILE_TO_REMOVE")

if [ -z "$files" ]; then
    echo "No files found to clean."
    exit 0
fi

for f in $files; do
    echo "Removing: $f"
    rm "$f"
    if [ $? -eq 0 ]; then
        removed_count=$((removed_count + 1))
    fi
    found_count=$((found_count + 1))
done

echo "------------------------------------------------"
echo "Cleanup Complete."
echo "Found: $found_count"
echo "Removed: $removed_count"
echo "------------------------------------------------"

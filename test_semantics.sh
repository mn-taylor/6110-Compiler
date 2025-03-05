#!/bin/bash

BASE_DIR="./tests"
SCRIPT="./run.sh"
TEST_TYPE="inter"

./build.sh
for category in "legal" "illegal"; do
    INPUT_DIR="$BASE_DIR/phase2-semantics/public/$category"

    for file in "$INPUT_DIR"/*.dcf; do
        if [ -f "$file" ]; then
            "$SCRIPT" -t "$TEST_TYPE" "$file" >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                if [[ "$category" == "illegal" ]]; then
                    echo "$file: ❌ Legal BUT SHOULD BE ILLEGAL"
                else
                    echo "$file: ✅ Correctly legal"
                fi
            else
                if [[ "$category" == "legal" ]]; then
                    echo "$file: ❌ Illegal BUT SHOULD BE LEGAL"
                else
                    echo "$file: ✅ Correctly illegal"
                fi
            fi
        fi
    done
done


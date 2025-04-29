#!/bin/bash

MAC=0
if [[ "$1" == "-m" ]]; then
    MAC=1
fi

./build.sh


INPUT_DIR="./tests/phase3/input"
ASM_DIR="./tests/phase3/asm"
mkdir -p "$ASM_DIR"

for file in "$INPUT_DIR"/*.dcf; do
    if [ -f "$file" ]; then
        BASENAME=$(basename "$file" .dcf)
        ASM_FILE="$ASM_DIR/$BASENAME.S"
        EXE_FILE="$ASM_DIR/$BASENAME"
        OUTPUT_FILE="$ASM_DIR/$BASENAME.actual"
        EXPECTED_FILE="./tests/phase3/output/$BASENAME.dcf.out"

        > "$OUTPUT_FILE"

        # assemble
        if [[ $MAC -eq 1 ]]; then
            ./run.sh -t assembly -m "$file" -o "$ASM_FILE" -O all > "$OUTPUT_FILE" 2>>"$OUTPUT_FILE" 
        else
            ./run.sh -t assembly "$file" -o "$ASM_FILE" -O all > "$OUTPUT_FILE" 2>>"$OUTPUT_FILE" 
        fi            

        if [ $? -eq 0 ]; then

            # compile
            if [[ $MAC -eq 1 ]]; then
                gcc -O0 -arch x86_64 "$ASM_FILE" -o "$EXE_FILE" 2>>"$OUTPUT_FILE"
            else
                gcc -O0 -m64 "$ASM_FILE" -o "$EXE_FILE" 2>>"$OUTPUT_FILE"
            fi

            if [ $? -eq 0 ]; then
                "$EXE_FILE" > "$OUTPUT_FILE" 2>>"$OUTPUT_FILE"
                if cmp -s "$OUTPUT_FILE" "$EXPECTED_FILE"; then
                    echo "$file: ✅ Output matches"
                else
                    echo "$file: ❌ Output mismatch"
                fi
            else
                echo "$file: ❌ Executable not created (see .actual)" 
            fi
        else
            echo "$file: ❌ Assembly generation failed (see .actual) (don't forget to use -m)"
        fi
    fi
done

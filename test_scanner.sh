#!/usr/bin/env bash

./build.sh > /dev/null 2>&1


files=$(find ./public-tests/phase1-scanner/public/input/ -type f | grep -v "invalid" - | grep "dcf" -)

for f in $files; do
    prefix="./public-tests/phase1-scanner/public/input/";
    prefix_length=${#prefix};
    nice_f=${f:$prefix_length};
    echo "Scanning $nice_f";
    ./run.sh -t scan ./public-tests/phase1-scanner/public/input/$nice_f > ./public-tests/phase1-scanner/public/input/${nice_f::-4}.out;
    diff ./public-tests/phase1-scanner/public/input/${nice_f::-4}.out ./public-tests/phase1-scanner/public/output/${nice_f::-4}.out
    echo 
    done;

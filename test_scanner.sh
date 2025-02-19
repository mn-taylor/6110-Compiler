#!/usr/bin/env bash

./build.sh > /dev/null 2>&1

# valid files

files=$(find ./public-tests/phase1-scanner/public/input/ -type f | grep -v "invalid" - | grep "dcf" -)

for f in $files; do
    prefix="./public-tests/phase1-scanner/public/input/";
    prefix_length=${#prefix};
    nice_f=${f:$prefix_length};
    # echo "Scanning $nice_f";
    ./run.sh -t scan ./public-tests/phase1-scanner/public/input/$nice_f > ./public-tests/phase1-scanner/public/input/${nice_f::-4}.out;
    if ! cmp ./public-tests/phase1-scanner/public/input/${nice_f::-4}.out ./public-tests/phase1-scanner/public/output/${nice_f::-4}.out; then
	echo $nice_f failed
    fi
done;

# invalid files

files=$(find ./public-tests/phase1-scanner/public/input/ -type f | grep "invalid" - | grep "dcf" -)

for f in $files; do
    prefix="./public-tests/phase1-scanner/public/input/";
    prefix_length=${#prefix};
    nice_f=${f:$prefix_length};
    # echo "Scanning $nice_f";
    ./run.sh -t scan ./public-tests/phase1-scanner/public/input/$nice_f > ./public-tests/phase1-scanner/public/input/${nice_f::-4}.out;
    # diff ./public-tests/phase1-scanner/public/input/${nice_f::-4}.out ./public-tests/phase1-scanner/public/output/${nice_f::-4}.out
    if ! grep "ERROR" ./public-tests/phase1-scanner/public/input/${nice_f::-4}.out > /dev/null; then
	echo $nice_f failed
    fi
done;

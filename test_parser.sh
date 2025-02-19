#!/usr/bin/env bash

./build.sh > /dev/null 2>&1

# legal files

files=$(find ./public-tests/phase1-parser/public/legal/ -type f | grep "dcf" -)

for f in $files; do
    prefix="./public-tests/phase1-parser/public/legal/";
    prefix_length=${#prefix};
    nice_f=${f:$prefix_length};
    # echo "Parsing $nice_f";
    if ./run.sh -t parse $prefix$nice_f > /dev/null;
    then : # echo good;
    else echo FAILED;
    fi
done;

# # illegal files

files=$(find ./public-tests/phase1-parser/public/illegal/ -type f | grep "dcf" -)

for f in $files; do
    prefix="./public-tests/phase1-parser/public/illegal/";
    prefix_length=${#prefix};
    nice_f=${f:$prefix_length};
    # echo "Parsing $nice_f";
    if ./run.sh -t parse $prefix$nice_f > /dev/null 2>&1;
    then echo FAILED $nice_f;
    else : # echo good; :
    fi
done;

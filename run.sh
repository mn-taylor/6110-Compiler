#!/usr/bin/env bash

chmod +x ./run_1.sh
( ./run_1.sh "$@" ) & pid=$!
( sleep 50 && kill -HUP $pid ) 2>/dev/null & watcher=$!
if wait $pid 2>/dev/null; then
    echo "your_command finished"
    pkill -HUP -P $watcher
    wait $watcher
    exit 0
else
    echo "your_command interrupted"
    ./run_1.sh "$@" -O dumb
    exit 0
fi

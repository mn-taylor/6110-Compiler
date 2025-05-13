#!/usr/bin/env bash

chmod +x ./run_1.sh
( ./run_1.sh "$@" ) & pid=$!
( sleep 2 && kill -HUP $pid ) 2>/dev/null & watcher=$!
if wait $pid 2>/dev/null; then
    echo "your_command finished"
    pkill -HUP -P $watcher
    wait $watcher
else
    echo "your_command interrupted"
    ./run_1.sh "$@" -O dumb
fi

#!/bin/bash
mkdir -p /logs/verifier
if printf 'hello world\n' | cmp -s - /hello.txt; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi

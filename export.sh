#!/usr/bin/env bash

./build.sh

docker save zerofield | gzip -c > Zerofield.tar.gz

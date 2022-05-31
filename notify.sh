#!/bin/sh

cargo run --release $1
while true; do
	for f in shaders/*.glsl; do
		echo doing $f
		TIMES=$(make all 2>&1| grep Running | wc -l)

		echo times is $TIMES

		if [ "$TIMES" -lt "1" ]; then
			cargo run --release $f 50
		fi
	done
done

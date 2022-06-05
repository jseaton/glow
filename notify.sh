#!/bin/sh

CURTIME=$(stat current -c %Y)
while true; do
	for f in shaders/*.glsl; do
		NEWTIME=$(stat current -c %Y)

		echo $CURTIME $NEWTIME

		if [ $NEWTIME -gt $CURTIME ]; then
			echo "YES IT IS NEW"
			cat current | xargs cargo run --release
			NEWTIME=$CURTIME
		else
			echo "NO IT IS NOT NEW"
			cargo run --release $f
		fi
	done
done

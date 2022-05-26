#!/bin/sh

for f in shaders/*.glsl; do
	base="${f%.*}"
	echo Compiling $f
	cat prefix.glsl $f postfix.glsl | glslangValidator $f -V --stdin -S frag -o $base.spv
done
cargo run --release $@

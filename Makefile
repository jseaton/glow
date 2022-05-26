all: $(patsubst %.glsl,%.spv,$(wildcard shaders/*.glsl))

shaders/%.spv: shaders/%.glsl prefix.glsl postfix.glsl
	cat prefix.glsl $< postfix.glsl | glslangValidator $f -V --stdin -S frag -o $@

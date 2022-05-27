// tex tex/domegrid.png
layout(set = 1, binding = 0) uniform sampler2D iChannel0;

vec4 mainImage() {
    return texture(iChannel0, tex_coords.xy);
}

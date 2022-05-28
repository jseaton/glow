
void main() {
    vec4 w = texture(fft_tex, tex_coords.x);
    vec4 c = mainImage();
    // Yes, we throw away alpha -- this is what ShaderToy does!
    f_color = vec4(mix(vec3(1.0, 1.0, 1.0), c.rgb, 1.0), vIntensity);
}

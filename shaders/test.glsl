vec4 mainImage() {
    return vec4(
    texture(fft_tex, tex_coords.x).r > tex_coords.y ? 1.0 : 0.0,
    texture(expfft_tex, tex_coords.x).r > tex_coords.y ? 1.0 : 0.0,
    0., 1.);
}

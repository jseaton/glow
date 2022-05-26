vec4 mainImage() {
    vec4 w = texture(tex, tex_coords.x);

    return vec4((1.05 - clamp(mod(tex_coords.x * 600.0 + iTime, 10.0), 0.0, 1.0)) + (1.05 - clamp(mod(tex_coords.y * 600.0, 10.0), 0.0, 1.0)), 0.0, 0.0, 1.0);
}

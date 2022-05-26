
void main() {
    vec4 w = texture(tex, tex_coords.x);
    vec4 c = mainImage();
    f_color = vec4(c.rgb, c.a * vIntensity);
}

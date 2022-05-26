// CC0: Spiral Surprise (mrange)
// More spiral experimenting gave a surprisingly appealing result

#define RESOLUTION  iResolution
#define TIME        iTime
#define PI          3.141592654
#define TAU         (2.0*PI)
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))

const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}

// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
float sRGB(float t) { return mix(1.055*pow(t, 1./2.4) - 0.055, 12.92*t, step(t, 0.0031308)); }
// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
vec3 sRGB(in vec3 c) { return vec3 (sRGB(c.x), sRGB(c.y), sRGB(c.z)); }

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x)+PI);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
vec2 mod2(inout vec2 p, vec2 size) {
  vec2 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5,size) - size*0.5;
  return c;
}

vec2 spiralEffect(vec2 p, float a, float n) {
  vec2 op = p;
  float b = a/TAU;
  vec2 pp   = toPolar(op);
  float  aa = pp.y;
  pp        -= vec2(pp.y*n*b, (pp.x/b-PI)/n);
  vec2  nn  = mod2(pp, vec2(a, TAU/n));
  return vec2(pp.x, mod(nn.y, n));
}

// License: Unknown, author: Unknown, found: don't remember
float hash(float co) {
  return fract(sin(co*12.9898) * 13758.5453);
}

vec4 mainImage() {
  vec2 q  = fragCoord/RESOLUTION.xy;
  vec2 p  = -1. + 2. * q;
  p.x     *= RESOLUTION.x/RESOLUTION.y;
  vec3 col = vec3(1.0);
  
  float a = 0.5;
  vec2 se0 = spiralEffect(p*ROT(-0.523*TIME), a, 4.0);
  vec2 se1 = spiralEffect(p.yx*ROT(0.1*TIME), a, 6.0);
  vec2 se = vec2(max(se0.x, se1.x), se0.y+se1.y);
  float h = hash(se.y+123.4)+0.5*length(p)+0.25*p.y;
  col = hsv2rgb(vec3(fract(h), 0.95, abs(se.x)/a));
  col = sRGB(col);
  return vec4(col, 1.0);
}



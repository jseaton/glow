// License CC0: Smooth Kaleidoscope 2 (mrange)
// Been working a bit too much and lacked the motivation do shaders for awhile
// Wanted to do something simple to get started again so went back to kaleidoscopic 
// effects that tend to be rather simple yet produce appealing results.

// SABS by ollj.
// rgb2hsv from: https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl

#define TIME        iTime
#define RESOLUTION  iResolution
#define PI          3.141592654
#define TAU         (2.0*PI)
#define LESS(a,b,c) mix(a,b,step(0.,c))
#define SABS(x,k)   LESS((.5/(k))*(x)*(x)+(k)*.5,abs(x),abs(x)-(k))
#define ROT(x)      mat2(cos(x), -sin(x), sin(x), cos(x))
#define PSIN(x)     (0.5+0.5*sin(x))

const mat2[] rotations = mat2[](ROT(0.0*PI/2.0), ROT(1.0*PI/2.0), ROT(2.0*PI/2.0), ROT(3.0*PI/2.0));

float rand(vec2 n) { 
  return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

vec2 toRect(vec2 p) {
  return p.x*vec2(cos(p.y), sin(p.y));
}

vec2 mod2_1(inout vec2 p) {
  vec2 pp = p + 0.5;
  vec2 nn = floor(pp);
  p = fract(pp) - 0.5;
  return nn;
}

float modMirror1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize,size) - halfsize;
  p *= mod(c, 2.0)*2.0 - 1.0;
  return c;
}

float circle(vec2 p, float r) {
  return length(p) - r;
}

float smoothKaleidoscope(inout vec2 p, float sm, float rep) {
  vec2 hp = p;

  vec2 hpp = toPolar(hp);
  float rn = modMirror1(hpp.y, TAU/rep);

  float sa = PI/rep - SABS(PI/rep - abs(hpp.y), sm);
  hpp.y = sign(hpp.y)*(sa);

  hp = toRect(hpp);

  p = hp;

  return rn;
}

float cell0(vec2 p) {
  float d0 = circle(p+0.5, 0.5);
  float d1 = circle(p-0.5, 0.5);

  float d = 1E6;
  d = min(d, d0);
  d = min(d, d1);

  return d;
}

float cell1(vec2 p) {
  float d0 = abs(p.x);
  float d1 = abs(p.y);
  float d2 = circle(p, 0.25);
  
  float d = 1E6;
  d = min(d, d0);
  d = min(d, d1);
  d = min(d, d2);

  return d;
}

float cell(vec2 p, vec2 cp, vec2 n, float lw) {
  float r = rand(n+1237.0);
  cp *= rotations[int(4.0*r)];
  float rr = fract(13.0*r);
  float d = 0.0;
  if (rr > 0.25) {
    d = cell0(cp);
  } else {
    d = cell1(cp);
  }
  return abs(d) - lw;
}

float truchet(vec2 p, float lw) {
  float s = 0.1;
  p /= s;
  vec2 cp = p;
  vec2 n = mod2_1(cp);
  float d = cell(p, cp, n, lw)*s;
  return d;
}

float df(vec2 p, float rep, float time) {
  const float lw = 0.05;

  vec2 pp = toPolar(p);
  pp.x /= 1.0+pp.x;
  p = toRect(pp);
  
  vec2 cp = p;
  
  float sm  = 3.0/rep;

  float n = smoothKaleidoscope(cp, sm, rep);
  cp *= ROT(0.05*time);
  cp-=0.1*time;
  
  return truchet(cp, lw*mix(0.25, 2.0, PSIN(-2.0*time+5.0*cp.x+(0.5*rep)*cp.y)));
}

vec3 color(vec2 q, vec2 p, float rep, float tm) {
  vec2 pp = toPolar(p);
  
  float d = df(p, rep, tm);
  
  vec3 col = vec3(0.0);
  float aa = 2.0/RESOLUTION.y;
  vec3 baseCol = hsv2rgb(vec3(-0.25*tm+sin(5.5*d), tanh(pp.x), 1.0));
  col = mix(col, baseCol, smoothstep(-aa, aa, -d));
  col = baseCol + 0.5*col.zxy;
  return col;
}

vec3 postProcess(vec3 col, vec2 q)  {
  col=pow(clamp(col,0.0,1.0),vec3(0.75)); 
  col=col*0.6+0.4*col*col*(3.0-2.0*col);
  col=mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col*=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

vec4 mainImage() {
  vec2 q    = fragCoord/RESOLUTION.xy;
  vec2 p    = -1. + 2. * q;
  p.x       *= RESOLUTION.x/RESOLUTION.y;
  vec2 op   = p;
  vec3 col  = vec3(0.0);
  
  float tm = TIME*0.25;
  float aa = -1.0+0.5*(length(p));
  float a = 1.0;
  float ra = tanh(length(0.5*p));

  for (int i = 0; i < 6; ++i) {
    p *= ROT(sqrt(0.1*float(i))*tm-ra);
    col += +a*color(q, p, 30.0-6.0*float(i), 1.0*tm+float(i));
    a *= aa;
  }

  col = tanh(col);
  col = abs(p.y-col);
  col = max(1.0-col, 0.0);
  col = pow (col, tanh(0.0+length(op)*1.0*vec3(1.0, 1.5, 3.0)));
  col = postProcess(col, q);
  return vec4(col, 1.0);
}


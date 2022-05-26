// License CC0: Flying through kaleidoscoped truchet patterns (mrange)
// Experimenting with simple truchet patterns + kaleidoscope turned out rather nice 
//  so I wanted to share.

// SABS by ollj

#define PI              3.141592654
#define TAU             (2.0*PI)
#define TIME            iTime
#define RESOLUTION      iResolution
#define LESS(a,b,c)     mix(a,b,step(0.,c))
#define SABS(x,k)       LESS((.5/(k))*(x)*(x)+(k)*.5,abs(x),abs(x)-(k))
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))

const vec3 std_gamma        = vec3(2.2, 2.2, 2.2);

float hash(float co) {
  return fract(sin(co*12.9898) * 13758.5453);
}

float hash(vec3 co) {
  return fract(sin(dot(co, vec3(12.9898,58.233, 12.9898+58.233))) * 13758.5453);
}

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

vec2 mod2_1(inout vec2 p) {
  vec2 c = floor(p + 0.5);
  p = fract(p + 0.5) - 0.5;
  return c;
}

float modMirror1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize,size) - halfsize;
  p *= mod(c, 2.0)*2.0 - 1.0;
  return c;
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

vec3 toScreenSpace(vec3 col) {
  return pow(col, 1.0/std_gamma);
}

vec3 alphaBlend(vec3 back, vec4 front) {
  vec3 colb = back.xyz;
  vec3 colf = front.xyz;
  vec3 xyz = mix(colb, colf.xyz, front.w);
  return xyz;
}

float circle(vec2 p, float r) {
  return length(p) - r;
}

vec3 offset(float z) {
  float a = z;
  vec2 p = -0.075*(vec2(cos(a), sin(a*sqrt(2.0))) + vec2(cos(a*sqrt(0.75)), sin(a*sqrt(0.5))));
  return vec3(p, z);
}

vec3 doffset(float z) {
  float eps = 0.1;
  return 0.5*(offset(z + eps) - offset(z - eps))/eps;
}

vec3 ddoffset(float z) {
  float eps = 0.1;
  return 0.125*(doffset(z + eps) - doffset(z - eps))/eps;
}

// -----------------------------------------------------------------------------
// PLANE0__BEGIN
// -----------------------------------------------------------------------------

const float plane0_lw = 0.05;

const mat2[] plane0_rots = mat2[](ROT(0.0*PI/2.0), ROT(1.00*PI/2.0), ROT(2.0*PI/2.0), ROT(3.0*PI/2.0));

vec2 plane0_cell0(vec2 p, float h) {
  float d0  = circle(p-vec2(0.5), 0.5);
  float d1  = circle(p+vec2(0.5), 0.5);

  float d = 1E6;
  d = min(d, d0);
  d = min(d, d1);
  return vec2(d, 1E6); // 1E6 gives a nice looking bug, 1E4 produces a more "correct" result
}

vec2 plane0_cell1(vec2 p, float h) {
  float d0  = abs(p.x);
  float d1  = abs(p.y);
  float d2  = circle(p, mix(0.2, 0.4, h));

  float d = 1E6;
  d = min(d, d0);
  d = min(d, d1);
  d = min(d, d2);
  return vec2(d, d2+plane0_lw);
}

vec2 plane0_df(vec3 pp, float h, out vec3 n) {
  vec2 p = pp.xy*ROT(TAU*h+TIME*fract(23.0*h)*0.5);
  float hd = circle(p, 0.4);

  vec2 hp = p;
  float rep = 2.0*floor(mix(5.0, 25.0, fract(h*13.0)));
  float sm = mix(0.05, 0.125, fract(h*17.0))*24.0/rep;
  float kn = 0.0;
  kn = smoothKaleidoscope(hp, sm, rep);
  vec2 hn = mod2_1(hp);
  float r = hash(vec3(hn, h));

  hp *= plane0_rots[int(r*4.0)];
  float rr = fract(r*31.0);
  vec2 cd0 = plane0_cell0(hp, rr);
  vec2 cd1 = plane0_cell1(hp, rr);
  vec2 d0 = mix(cd0, cd1, vec2(fract(r*13.0) > 0.5));

  hd = min(hd, d0.y);

  float d = 1E6;
  d = min(d, d0.x);
  d = abs(d) - plane0_lw;
  d = min(d, hd - plane0_lw*2.0);

  n = vec3(hn, kn);

  return vec2(hd, d);
}

vec4 plane0(vec3 ro, vec3 rd, vec3 pp, vec3 off, float aa, float n) {

  float h = hash(n);
  float s = mix(0.05, 0.25, h);

  vec3 hn;
  vec3 p = pp-off*vec3(1.0, 1.0, 0.0);

  vec2 dd = plane0_df(p/s, h, hn)*s;
  float d = dd.y;

  float a  = smoothstep(-aa, aa, -d);
  float ha = smoothstep(-aa, aa, dd.x);

  vec4 col = vec4(mix(vec3(1.0), vec3(0.0), a), ha);

  return col;
}

// -----------------------------------------------------------------------------
// PLANE0__END
// -----------------------------------------------------------------------------

vec4 plane(vec3 ro, vec3 rd, vec3 pp, vec3 off, float aa, float n) {
  return plane0(ro, rd, pp, off, aa, n);
}

vec3 skyColor(vec3 ro, vec3 rd) {
  float ld = max(dot(rd, vec3(0.0, 0.0, 1.0)), 0.0);
  return vec3(tanh(3.0*pow(ld, 100.0)));
}

vec3 color(vec3 ww, vec3 uu, vec3 vv, vec3 ro, vec2 p) {
  float lp = length(p);
  vec2 np = p + 1.0/RESOLUTION.xy;
  float rdd = (2.0+0.5*tanh(lp));
  vec3 rd = normalize(p.x*uu + p.y*vv + rdd*ww);
  vec3 nrd = normalize(np.x*uu + np.y*vv + rdd*ww);

  const vec3 errorCol = vec3(1.0, 0.0, 0.0);

  float planeDist = 1.0-0.25;
  const int furthest = 6;
  const int fadeFrom = max(furthest-4, 0);

  float nz = floor(ro.z / planeDist);

  vec3 skyCol = skyColor(ro, rd);

  vec3 col = skyCol;

  for (int i = furthest; i >= 1 ; --i) {
    float pz = planeDist*nz + planeDist*float(i);

    float pd = (pz - ro.z)/rd.z;

    if (pd > 0.0) {
      vec3 pp = ro + rd*pd;
      vec3 npp = ro + nrd*pd;

      float aa = 3.0*length(pp - npp);

      vec3 off = offset(pp.z);

      vec4 pcol = plane(ro, rd, pp, off, aa, nz+float(i));

      float nz = pp.z-ro.z;
      float fadeIn = (1.0-smoothstep(planeDist*float(fadeFrom), planeDist*float(furthest), nz));
      float fadeOut = smoothstep(0.0, planeDist*0.1, nz);
      pcol.xyz = mix(skyCol, pcol.xyz, (fadeIn));
      pcol.w *= fadeOut;

      col = alphaBlend(col, pcol);
    } else {
      break;
    }

  }

  return col;
}

vec3 postProcess(vec3 col, vec2 q) {
  col = clamp(col, 0.0, 1.0);
  col = toScreenSpace(col);
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col *=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

vec3 effect(vec2 p, vec2 q) {
  float tm  = TIME*0.4;
  vec3 ro   = offset(tm);
  vec3 dro  = doffset(tm);
  vec3 ddro = ddoffset(tm);

  vec3 ww = normalize(dro);
  vec3 uu = normalize(cross(normalize(vec3(0.0,1.0,0.0)+ddro), ww));
  vec3 vv = normalize(cross(ww, uu));

  vec3 col = color(ww, uu, vv, ro, p);
  col = postProcess(col, q);
  return col;
}

vec4 mainImage() {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;

  vec3 col = effect(p, q);

  return vec4(col, 1.0);
}


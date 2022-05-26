// CC0: Random sunday shader (mrange)

#define RESOLUTION  iResolution
#define TIME        iTime
#define PI          3.141592654
#define PI_2        (0.5*3.141592654)
#define TAU         (2.0*PI)
#define ROT(a)      mat2(cos(a), sin(a), -sin(a), cos(a))
#define PCOS(x)     (0.5+0.5*cos(x))
#define BPM         125.0

//#define CRT_EFFECT

const float planeDist = 1.0-0.825;

// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}
// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
//  Macro version of above to enable compile-time constants
#define HSV2RGB(c)  (c.z * mix(hsv2rgb_K.xxx, clamp(abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www) - hsv2rgb_K.xxx, 0.0, 1.0), c.y))

// License: Unknown, author: Unknown, found: don't remember
vec4 alphaBlend(vec4 back, vec4 front) {
  float w = front.w + back.w*(1.0-front.w);
  vec3 xyz = (front.xyz*front.w + back.xyz*back.w*(1.0-front.w))/w;
  return w > 0.0 ? vec4(xyz, w) : vec4(0.0);
}

// License: Unknown, author: Unknown, found: don't remember
vec3 alphaBlend(vec3 back, vec4 front) {
  return mix(back, front.xyz, front.w);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float mod1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize, size) - halfsize;
  return c;
}

float planex(vec2 p, float w) {
  return abs(p.y) - w;
}

float circle(vec2 p, float r) {
  return length(p) - r;
}

// https://iquilezles.org/articles/spherefunctions
float sphered(vec3 ro, vec3 rd, vec4 sph, float dbuffer) {
  float ndbuffer = dbuffer/sph.w;
  vec3  rc = (ro - sph.xyz)/sph.w;

  float b = dot(rd,rc);
  float c = dot(rc,rc) - 1.0;
  float h = b*b - c;
  if( h<0.0 ) return 0.0;
  h = sqrt( h );
  float t1 = -b - h;
  float t2 = -b + h;

  if( t2<0.0 || t1>ndbuffer ) return 0.0;
  t1 = max( t1, 0.0 );
  t2 = min( t2, ndbuffer );

  float i1 = -(c*t1 + b*t1*t1 + t1*t1*t1/3.0);
  float i2 = -(c*t2 + b*t2*t2 + t2*t2*t2/3.0);
  return (i2-i1)*(3.0/4.0);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/smin
float pmin(float a, float b, float k) {
  float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float pmax(float a, float b, float k) {
  return -pmin(-a, -b, k);
}

// License: Unknown, author: Unknown, found: don't remember
float tanh_approx(float x) {
  //  Found this somewhere on the interwebs
  //  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(float co) {
  return fract(sin(co*12.9898) * 13758.5453);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(vec2 p) {
  float a = dot (p, vec2 (127.1, 311.7));
  return fract(sin(a)*43758.5453123);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/index.htm
vec3 postProcess(vec3 col, vec2 q) {
#ifdef CRT_EFFECT  
  col *= 1.5*smoothstep(-2.0, 1.0, sin(0.5*PI*q.y*RESOLUTION.y));
#endif  
  col = clamp(col, 0.0, 1.0);
  col = pow(col, vec3(1.0/2.2));
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col *=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

const float logo_radius= 0.25;
const float logo_off   = 0.25;
const float logo_dx    = 0.5/sqrt(3.0);
const float logo_width = 0.1;

float rcp(float x) {
  return 1.0 / x;
}

//P. Gilcher '21, strange approximation
// Source found at: https://www.shadertoy.com/view/flSXRV
float fast_atan2(float y, float x) {
  float cosatan2 = x * rcp(abs(x) + abs(y));
  float t = PI_2 - cosatan2 * PI_2;
  return y < 0.0 ? -t : t;
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/distfunctions2d
float hex(vec2 p, float r) {
  const vec3 k = vec3(-sqrt(3.0)*0.5,0.5,sqrt(1.0/3.0));
  p = abs(p);
  p -= 2.0*min(dot(k.xy,p),0.0)*k.xy;
  p -= vec2(clamp(p.x, -k.z*r, k.z*r), r);
  return length(p)*sign(p.y);
}

// License: Unknown, author: Martijn Steinrucken, found: https://www.youtube.com/watch?v=VmrIDyYiJBA
vec2 hextile(inout vec2 p) {
  // See Art of Code: Hexagonal Tiling Explained!
  // https://www.youtube.com/watch?v=VmrIDyYiJBA
  const vec2 sz       = vec2(1.0, sqrt(3.0));
  const vec2 hsz      = 0.5*sz;

  vec2 p1 = mod(p, sz)-hsz;
  vec2 p2 = mod(p - hsz, sz)-hsz;
  vec2 p3 = dot(p1, p1) < dot(p2, p2) ? p1 : p2;
  vec2 n = ((p3 - p + hsz)/sz);
  p = p3;

  n -= vec2(0.5);
  // Rounding to make hextile 0,0 well behaved
  return round(n*2.0)*0.5;
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float modPolar(inout vec2 p, float repetitions) {
  float angle = 2.0*PI/repetitions;
  float a = atan(p.y, p.x) + angle/2.;
  float r = length(p);
  float c = floor(a/angle);
  a = mod(a,angle) - angle/2.;
  p = vec2(cos(a), sin(a))*r;
  // For an odd number of repetitions, fix cell index of the cell in -x direction
  // (cell index would be e.g. -5 and 5 in the two halves of the cell):
  if (abs(c) >= (repetitions/2.0)) c = abs(c);
  return c;
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float pabs(float a, float k) {
  return pmax(a, -a, k);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float modMirror1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize,size) - halfsize;
  p *= mod(c, 2.0)*2.0 - 1.0;
  return c;
}


// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

// License: CC0, author: Mårten Rånge, found: https://github.com/mrange/glsl-snippets
float smoothKaleidoscope(inout vec2 p, float sm, float rep) {
  vec2 hp = p;

  vec2 hpp = toPolar(hp);
  float rn = modMirror1(hpp.y, TAU/rep);

  float sa = PI/rep - pabs(PI/rep - abs(hpp.y), sm);
  hpp.y = sign(hpp.y)*(sa);

  hp = toRect(hpp);

  p = hp;

  return rn;
}

float stripes(float d) {
  const float cc = 0.42;
  d = abs(d)-logo_width*cc;
  d = abs(d)-logo_width*cc*0.5;
  return d;
}

vec4 merge(vec4 s0, vec4 s1) {
  bool dt = s0.z < s1.z; 
  vec4 b = dt ? s0 : s1;
  vec4 t = dt ? s1 : s0;

  b.x *= 1.0-exp(-max(80.0*(t.w), 0.0));

  vec4 r = vec4(
      mix(b.xy, t.xy, t.y)
    , b.w < t.w ? b.z : t.z 
    , min(b.w, t.w)
    );
  
  return r;
}

vec4 figure_8(float aa, vec2 p) {
  vec2  p1 = p-vec2(logo_dx, -logo_off);
  float d1 = abs(circle(p1, logo_radius));
  float a1 = fast_atan2(-p1.x, -p1.y);
  float s1 = stripes(d1);
  float o1 = d1 - logo_width;

  vec2  p2 = p-vec2(logo_dx, logo_off);
  float d2 = abs(circle(p2, logo_radius));
  float a2 = fast_atan2(p2.x, p2.y);  
  float s2 = stripes(d2);
  float o2 = d2 - logo_width;

  vec4 c0 = vec4(smoothstep(aa, -aa, s1), smoothstep(aa, -aa, o1), a1, o1);
  vec4 c1 = vec4(smoothstep(aa, -aa, s2), smoothstep(aa, -aa, o2), a2, o2);

  return merge(c0, c1);
}

vec4 figure_half_8(float aa, vec2 p) {
  vec2  p1 = p-vec2(logo_dx, -logo_off);
  float d1 = abs(circle(p1, logo_radius));
  float a1 = fast_atan2(-p1.x, -p1.y);
  float s1 = stripes(d1);
  float o1 = d1 - logo_width;

  vec4 c0 = vec4(smoothstep(aa, -aa, s1), smoothstep(aa, -aa, o1), a1, o1);

  return c0;
}

vec2 flipy(vec2 p) {
  return vec2(p.x, -p.y);
}

vec4 clogo(vec2 p, float aa, float z, out float d) {
  float iz = 1.0/z;
  p *= iz;
  float  n = modPolar(p, 3.0);

  vec4 s0 = figure_8(aa, p);
  vec4 s1 = figure_half_8(aa, p*ROT(2.0*PI/3.0));
  vec4 s2 = figure_half_8(aa, flipy(p*ROT(4.0*PI/3.0)));
  s1.z += -PI;
  
  vec4 s = s0;
  s = merge(s, s1);
  s = merge(s, s2);

  d = s.w;
  vec3 hsv = vec3(fract(s.z/PI+TIME*0.5), 0.9, 1.0);
  return vec4(hsv2rgb(hsv)*s.x, s.y);
}


vec3 offset(float z) {
  float a = z;
  vec2 p = -0.15*(vec2(cos(a), sin(a*sqrt(2.0))) + vec2(cos(a*sqrt(0.75)), sin(a*sqrt(0.5))));
  return vec3(p, z);
}

vec3 doffset(float z) {
  float eps = 0.1;
  return 0.5*(offset(z + eps) - offset(z - eps))/eps;
}

vec3 ddoffset(float z) {
  float eps = 0.1;
  return 0.5*(doffset(z + eps) - doffset(z - eps))/eps;
}

vec4 plane(vec3 ro, vec3 rd, vec3 pp, vec3 off, float aa, float n) {
  float hn = hash(n);
  float hn0 = hn;
  float hn1 = fract(137.0*hn);
  float z = mix(0.1, 0.3, hn0);
  
  float pd = length(ro - pp);
  aa *= mix(1.0, 15.0, smoothstep(planeDist*4.0, 0.0, pd));

  vec2 p = (pp-off*1.0*vec3(1.0, 1.0, 0.0)).xy;
  p *= ROT(mix(0.125, 0.66, hn1)*TIME);
  p /= z;
  aa /= z;
  float d;
  float a = TAU*TIME/300.0;

  float cd = hex(p.yx, 0.5);
  vec2 hp = p;
  vec2 np = hextile(hp);
  vec2 cp = hp;
  float h = hash(np);
  float hh = fract(137.0*h);
  float sm = mix(mix(0.025, 0.25, hh), 0.025, h);
  float rep = 2.0*floor(mix(8.0, 30.0, h));
  float cn = 0.0; 
  cn = smoothKaleidoscope(cp, sm, rep);
  cp *= ROT(TIME*0.2+TAU*h);
  
  vec4 ccol = clogo(cp, aa, 0.6, d);
  vec3 gcol = hsv2rgb(vec3(h, 0.8, 4.0));
  vec3 col  = vec3(0.0);
  float g = exp(-40.0*max(d, 0.0));
  col += gcol*g;
  col = mix(col, ccol.xyz, ccol.w);

  float t0 = smoothstep(aa, -aa, d);
  float t1 = smoothstep(aa, -aa, -cd);
  float t2 = 2.0*g;
  float t = t1*tanh_approx(t0 +t2);
  
//  return vec4(col, tanh_approx(t+g));
  return vec4(col, t);
}

vec3 skyColor(vec3 ro, vec3 rd) {
  vec3 col = vec3(0.0);
  return col;
}

vec3 color(vec3 ww, vec3 uu, vec3 vv, vec3 ro, vec2 p) {
  float lp = length(p);
  vec2 np = p + 1.0/RESOLUTION.xy;
  const float rdd_per   = 5.0;
  //float rdd = (2.0+1.0*tanh_approx(lp));  // Playing around with rdd can give interesting distortions
  //float rdd = 2.0;
  float rdd =  (1.75+0.75*pow(lp,1.5)*tanh_approx(lp+0.9*PCOS(rdd_per*p.x)*PCOS(rdd_per*p.y)));
  
  vec3 rd = normalize(p.x*uu + p.y*vv + rdd*ww);
  vec3 nrd = normalize(np.x*uu + np.y*vv + rdd*ww);

  const int furthest = 8;
  const int fadeFrom = max(furthest-6, 0);

  const float fadeDist = planeDist*float(furthest - fadeFrom);
  float nz = floor(ro.z / planeDist);

  vec3 skyCol = skyColor(ro, rd);


  vec4 acol = vec4(0.0);
  const float cutOff = 0.95;
  bool cutOut = false;

  float maxpd = 0.0;

  // Steps from nearest to furthest plane and accumulates the color 
  for (int i = 1; i <= furthest; ++i) {
    float pz = planeDist*nz + planeDist*float(i);

    float pd = (pz - ro.z)/rd.z;

    if (pd > 0.0 && acol.w < cutOff) {
      vec3 pp = ro + rd*pd;
      maxpd = pd;
      vec3 npp = ro + nrd*pd;

      float aa = 3.0*length(pp - npp);

      vec3 off = offset(pp.z);

      vec4 pcol = plane(ro, rd, pp, off, aa, nz+float(i));

      float nz = pp.z-ro.z;
      float fadeIn = smoothstep(planeDist*float(furthest), planeDist*float(fadeFrom), nz);
      float fadeOut = smoothstep(0.0, planeDist*0.1, nz);
//      pcol.xyz = mix(skyCol, pcol.xyz, fadeIn);
      pcol.w *= fadeOut*fadeIn;
      pcol = clamp(pcol, 0.0, 1.0);

      acol = alphaBlend(pcol, acol);
    } else {
      cutOut = true;
      acol.w = acol.w > cutOff ? 1.0 : acol.w;
      break;
    }

  }

  vec3  glowCol   = hsv2rgb(vec3(fract(dot(p, p)-0.25*TIME), 0.66, 1.0));
  vec3  glowDown  = 2.5*glowCol;
  vec3  glowUp    = 5.0*glowCol;
  
  float beat = smoothstep(0.5, 1.0, sin(BPM*TIME*TAU/60.0));
  
  float glowRadius = mix(0.4, 0.5, beat);
  vec3 pp    = ro + rd*planeDist*float(furthest-1);
  vec3 off   = offset(pp.z);
  float sd   = sphered(ro, rd, vec4(off, glowRadius), mix(1E6, maxpd, acol.w));
  vec3 bcol  = mix(glowDown, glowUp, beat);
  vec3 gcol  = tanh(sd*bcol);
  skyCol += gcol;

  vec3 col = alphaBlend(skyCol, acol);
// To debug cutouts due to transparency  
//  col += cutOut ? vec3(1.0, -1.0, 0.0) : vec3(0.0);
  return col;
}

vec3 effect(vec2 p, vec2 q) {
  float tm  = planeDist*TIME*BPM/60.0;
  vec3 ro   = offset(tm);
  vec3 dro  = doffset(tm);
  vec3 ddro = ddoffset(tm);

  vec3 ww = normalize(dro);
  vec3 uu = normalize(cross(normalize(vec3(0.0,1.0,0.0)+ddro), ww));
  vec3 vv = normalize(cross(ww, uu));

  vec3 col = color(ww, uu, vv, ro, p);
  
  return col;
}

vec4 mainImage() {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;

  vec3 col = effect(p, q);
  col *= smoothstep(0.0, 4.0, TIME);
  col = postProcess(col, q);
  
  return vec4(col, 1.0);
}


// License CC0: Random little accidents (mrange)
//  Random coding turned out rather nice
//  I suspect 60% of the code is unnecessary :)
    
#define PI              3.141592654
#define TAU             (2.0*PI)
#define TIME            iTime
#define RESOLUTION      iResolution
#define ORT(p)          vec2((p).y, -(p).x)
#define LESS(a,b,c) mix(a,b,step(0.,c))
#define SABS(x,k)   LESS((.5/(k))*(x)*(x)+(k)*.5,abs(x),abs(x)-(k))
#define ROT(x)      mat2(cos(x), -sin(x), sin(x), cos(x))

const vec2 sz       = vec2(1.0, sqrt(3.0));
const vec2 hsz      = 0.5*sz;
const float is3     = 1.0/sqrt(3.0);
const vec2 cdir     = normalize(vec2(1.0, is3));
const vec2 flipy    = vec2(1.0, -1.0);

const vec2 coords[6] = vec2[6](
  is3*cdir*1.0/3.0,
  is3*cdir*2.0/3.0,
  vec2(0.5, is3/6.0),
  vec2(0.5, -is3/6.0),
  is3*cdir*2.0/3.0*flipy,
  is3*cdir*1.0/3.0*flipy
  );

const vec2 dcoords[6] = vec2[6](
  ORT(cdir),
  ORT(cdir),
  vec2(-1.0, 0.0),
  vec2(-1.0, 0.0),
  ORT(-cdir*flipy),
  ORT(-cdir*flipy)
  );

const int corners[] = int[](
  0, 1, 2, 3, 4, 5, 
  0, 1, 2, 4, 3, 5, 
  0, 1, 2, 5, 3, 4, 
  0, 2, 1, 3, 4, 5, 
  0, 2, 1, 4, 3, 5, 
  0, 2, 1, 5, 3, 4, 
  0, 3, 1, 2, 4, 5, 
  0, 3, 1, 4, 2, 5, 
  0, 3, 1, 5, 2, 4, 
  0, 4, 1, 2, 3, 5, 
  0, 4, 1, 3, 2, 5, 
  0, 4, 1, 5, 2, 3, 
  0, 5, 1, 2, 3, 4, 
  0, 5, 1, 3, 2, 4, 
  0, 5, 1, 4, 2, 3
  );
const int noCorners = corners.length()/6;

float hash(vec3 r)  { 
  return fract(sin(dot(r.xy,vec2(1.38984*sin(r.z),1.13233*cos(r.z))))*653758.5453); 
}

vec3 rgb2hsv(vec3 c) {
  const vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
  const vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 hextile(inout vec2 p) {
  // See Art of Code: Hexagonal Tiling Explained!
  // https://www.youtube.com/watch?v=VmrIDyYiJBA

  vec2 p1 = mod(p, sz)-hsz;
  vec2 p2 = mod(p - hsz, sz)-hsz;
  vec2 p3 = mix(p2, p1, vec2(dot(p1, p1) < dot(p2, p2)));
  vec2 n = ((p3 - p + hsz)/sz);
  p = p3;

  // Rounding to make hextile 0,0 well behaved
  return round(n*2.0)/2.0;
}

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

vec2 toRect(vec2 p) {
  return p.x*vec2(cos(p.y), sin(p.y));
}

// https://mercury.sexy/hg_sdf/
float mod1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize, size) - halfsize;
  return c;
}

vec3 alphaBlend(vec3 back, vec4 front) {
  vec3 colb = back.xyz;
  vec3 colf = front.xyz;
  vec3 xyz = mix(colb, colf.xyz, front.w);
  return xyz;
}

float dot2(vec2 v) { return dot(v,v); }
    
// IQ Bezier: https://www.shadertoy.com/view/MlKcDD
float bezier(vec2 pos, vec2 A, vec2 B, vec2 C) {    
  const float sqrt3 = sqrt(3.0);
  vec2 a = B - A;
  vec2 b = A - 2.0*B + C;
  vec2 c = a * 2.0;
  vec2 d = A - pos;

  float kk = 1.0/dot(b,b);
  float kx = kk * dot(a,b);
  float ky = kk * (2.0*dot(a,a)+dot(d,b))/3.0;
  float kz = kk * dot(d,a);      

  float res = 0.0;

  float p = ky - kx*kx;
  float p3 = p*p*p;
  float q = kx*(2.0*kx*kx - 3.0*ky) + kz;
  float h = q*q + 4.0*p3;

  if(h>=0.0) {   // 1 root
      h = sqrt(h);
      vec2 x = (vec2(h,-h)-q)/2.0;
      vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
      float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);
      res = dot2(d+(c+b*t)*t);
  } else {   // 3 roots
      float z = sqrt(-p);
      float v = acos(q/(p*z*2.0))/3.0;
      float m = cos(v);
      float n = sin(v)*sqrt3;
      vec3  t = clamp(vec3(m+m,-n-m,n-m)*z-kx, 0.0, 1.0);
      res = min(dot2(d+(c+b*t.x)*t.x), dot2(d+(c+b*t.y)*t.y));
      // the third root cannot be the closest. See https://www.shadertoy.com/view/4dsfRS
      // res = min(res,dot2(d+(c+b*t.z)*t.z));
  }
  
  return sqrt(res);
}

float bezier2(vec2 p, float f, vec2 p0, vec2 dp0, vec2 p1, vec2 dp1) {
  float dist = length(p0 - p1);
  float hdist = 0.5*f*dist;
  vec2 mp0 = p0 + hdist*dp0;
  vec2 mp1 = p1 + hdist*dp1;
  vec2 jp = (mp0 + mp1)*0.5;
  float d0 = bezier(p, p0, mp0, jp);
  float d1 = bezier(p, p1, mp1, jp);
  
  float d = d0;
  d = min(d, d1);
  return d;
}

float emin(float a, float b, float k) {
    float res = exp2( -k*a ) + exp2( -k*b );
    return -log2( res )/k;
}

float pmin(float a, float b, float k) {
  float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
  return mix( b, a, h ) - k*h*(1.0-h);
}

float df(vec2 p, float s, float aa) {
  vec2 hp = p/s;
  vec2 hn = hextile(hp);
  
  vec2 pp = toPolar(hp);
  float pn = mod1(pp.y, TAU/6.0);
  vec2 tp = toRect(pp);

  pn = mod(pn+3.0, 6.0);
  vec3 nn = vec3(hn, pn);

  float r = hash(nn);
  int sel = int(float(noCorners)*r);
  int off = sel*6;
  

  const float sw = 0.05;

  float d = 1E6;

  for (int i = 0; i < 3; ++i) {
    int c0 = corners[off + i*2 + 0];
    int c1 = corners[off + i*2 + 1];
    
    int c = max(c0, c1) - min(c0, c1);
    
    vec2 p0 = coords[c0];
    vec2 p1 = coords[c1];

    vec2 dp0 = dcoords[c0];
    vec2 dp1 = dcoords[c1];
    
    float mi = 0.5;    
    float mx = 0.5;    
    
    float rr = fract(r*27.0*float(i+1));
    switch(c) {
      case 1:
        mx = 1.75;
        break;
      case 2:
        mx = .95;
        break;
      case 3:
        mx = 1.5;
        break;
      case 4:
        mx = 0.75;
        break;
      case 5:
        mx = 1.95;
        break;
      default:
        break;
    }
    
    float f = mix(mi, mx, rr);
    
    float dd = (bezier2(tp, f, p0, dp0, p1, dp1)-0.005)*s;
    d = pmin(d, dd, 0.025);
    
  }

  d = abs(d) - 0.005;
  d = abs(d) - 0.0025;
  return d;
}

vec3 effect(vec2 p, vec2 q) {
  vec2 op = p;
  vec2 pp = toPolar(p);
  pp.y += pp.x*0.5+q.y*0.5-q.x;
  pp.x *= (-1.0+length(p));
  p = toRect(pp);
  float s = 1.75;
  float aa = 2.0/RESOLUTION.y;
  vec3 n;
  p += TIME*0.1;
  float d = df(p, s, aa);
  
  
  vec3 col = vec3(0.0);
  vec3 glowHsv = mix(vec3(0.0, 0.5, 1.0), vec3(1.0, 1.0, 1.0), 0.5 + 0.5*sin(-TIME+0.25*TAU*length(op)));
  vec3 glowCol = hsv2rgb(glowHsv);

  col = col += glowCol*exp(-d*30.0);  
  col = col += 4.0*(glowCol+vec3(0.2))*exp(-d*800.0);  

  return col;
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

vec3 postProcess(vec3 col, vec2 q)  {
  col=pow(clamp(col,0.0,1.0),vec3(0.75)); 
  col=col*0.6+0.4*col*col*(3.0-2.0*col);
  col=mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col*=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

vec4 mainImage() {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  vec2 op = p;
  
  const float rep = 42.00;
  const float srep = 0.075*40.0/rep;
  float n = smoothKaleidoscope(p, srep, rep);
  p *= ROT(iTime*0.25);
  vec3 col = effect(p, q);

  vec2 pp = toPolar(op);
  float per = 20.0/(0.1+length(p));
  float s = mix(10.0, 100.0, 0.5+0.5*(sin(per*op.y+iTime)*sin(per*op.x-iTime)));
  col = pow(col, 0.5*vec3(0.5, 0.75, 1.0)*tanh(-0.125+s*dot2(p)));
  col = postProcess(col, q);
  
  return vec4(col, 1.0);
}


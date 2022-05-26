// License: CC0 (mrange)
//  Results after some random coding while listening to online concert

// Experiment with these parameters for variants
const float size  = 0.75 ;
const float offc  = 1.05;
const float width = 0.0125;
const int   rep   = 15 ;

#define PHI   (.5*(1.+sqrt(5.)))
#define PI    3.141592654
#define TAU   (2.0*PI)
#define TTIME (TAU*iTime)

const vec3 plnormal = normalize(vec3(1, 1, -1));
const vec3 n1 = normalize(vec3(-PHI,PHI-1.0,1.0));
const vec3 n2 = normalize(vec3(1.0,-PHI,PHI+1.0));
const vec3 n3 = normalize(vec3(0.0,0.0,-1.0));

float pmin(float a, float b, float k) {
  float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
  return mix( b, a, h ) - k*h*(1.0-h);
}

float dodec(in vec3 z) {
  vec3 p = z;
  float t;
  z = abs(z);
  t=dot(z,n1); if (t>0.0) { z-=2.0*t*n1; }
  t=dot(z,n2); if (t>0.0) { z-=2.0*t*n2; }
  z = abs(z);
  t=dot(z,n1); if (t>0.0) { z-=2.0*t*n1; }
  t=dot(z,n2); if (t>0.0) { z-=2.0*t*n2; }
  z = abs(z);

  float dmin=dot(z-vec3(size,0.,0.),plnormal);
        
  dmin = abs(dmin) - width*7.5*(0.55 + 0.45*sin(10.0*length(p) - 0.5*p.y + TTIME/9.0));
        
  return dmin;
}

void rot(inout vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  p = vec2(c*p.x + s*p.y, -s*p.x + c*p.y);
}

float df(vec2 p) {
  float d = 100000.0;
  float off = 0.30  + 0.25*(0.5 + 0.5*sin(TTIME/11.0));
  for (int i = 0; i < rep; ++i) {
    vec2 ip = p;
    rot(ip, float(i)*TAU/float(rep));
    ip -= vec2(offc*size, 0.0);
    vec2 cp = ip;
    rot(ip, TTIME/73.0);
    float dd = dodec(vec3(ip, off*size));
    float cd = length(cp - vec2(0.25*sin(TTIME/13.0), 0.0)) - 0.125*size;
    cd = abs(cd) - width*0.5;
    d = pmin(d, dd, 0.05);
    d = pmin(d, cd, 0.025);
  }
  return d;
}

vec3 postProcess(vec3 col, vec2 q, vec2 p) {
  col=pow(clamp(col,0.0,1.0),vec3(0.75)); 
  col=col*0.6+0.4*col*col*(3.0-2.0*col);  // contrast
  col=mix(col, vec3(dot(col, vec3(0.33))), -0.4);  // satuation
  const float r = 1.5;
  float d = max(r - length(p), 0.0)/r;
  col *= vec3(1.0 - 0.25*exp(-200.0*d*d));
  return col;
}

vec4 mainImage() {
  vec2 q = fragCoord.xy / iResolution.xy;
  vec2 p = 2.0*(q - 0.5);
  p.x *= iResolution.x/iResolution.y;
  float d = df(p);

  float fuzzy = 0.0025;
    
  vec3 col = vec3(0.0);

  const vec3 baseCol = vec3(240.0, 175.0, 20.0)/255.0;
  
  col += 0.9*baseCol*vec3(smoothstep(fuzzy, -fuzzy, d));

  vec3 rgb = 0.5 + 0.5*vec3(sin(TAU*vec3(50.0, 49.0, 48.0)*(d - 0.050) + TTIME/3.0));

  col += baseCol.xyz*pow(rgb, vec3(8.0, 9.0, 7.0)); 
  col *= 1.0 - tanh(0.05+length(8.0*d));

  float phase = TAU/4.0*(-length(p) - 0.5*p.y) + TTIME/11.0;
 
  float wave = sin(phase);
  float fwave = sign(wave)*pow(abs(wave), 0.75);
 
  col = abs(0.79*(0.5 + 0.5*fwave) - col);
  col = pow(col, vec3(0.25, 0.5, 0.75));
  col = postProcess(col, q, p);

  return vec4(col, 1.0);
}


// CC BY-NC-SA 3.0 Unported mrange

#define PI  3.141592654
#define TAU (2.0*PI)

vec3 saturate(vec3 col) {
  return clamp(col, 0.0, 1.0);
}


void rot(inout vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  p = vec2(c*p.x + s*p.y, -s*p.x + c*p.y);
}

vec2 mod2(inout vec2 p, vec2 size)  {
  vec2 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5,size) - size*0.5;
  return c;
}

vec2 modMirror2(inout vec2 p, vec2 size) {
  vec2 halfsize = size*0.5;
  vec2 c = floor((p + halfsize)/size);
  p = mod(p + halfsize, size) - halfsize;
  p *= mod(c,vec2(2.0))*2.0 - vec2(1.0);
  return c;
}


vec2 toSmith(vec2 p)  {
  // z = (p + 1)/(-p + 1)
  // (x,y) = ((1+x)*(1-x)-y*y,2y)/((1-x)*(1-x) + y*y)
  float d = (1.0 - p.x)*(1.0 - p.x) + p.y*p.y;
  float x = (1.0 + p.x)*(1.0 - p.x) - p.y*p.y;
  float y = 2.0*p.y;
  return vec2(x,y)/d;
}

vec2 fromSmith(vec2 p)  {
  // z = (p - 1)/(p + 1)
  // (x,y) = ((x+1)*(x-1)+y*y,2y)/((x+1)*(x+1) + y*y)
  float d = (p.x + 1.0)*(p.x + 1.0) + p.y*p.y;
  float x = (p.x + 1.0)*(p.x - 1.0) + p.y*p.y;
  float y = 2.0*p.y;
  return vec2(x,y)/d;
}

vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,vec2(0))) + min(max(d.x,d.y),0.0);
}

float circle(vec2 p, float r) {
  return length(p) - r;
}



float mandala_df(float localTime, vec2 p) {
  vec2 pp = toPolar(p);
  float a = TAU/64.0;
  float np = pp.y/a;
  pp.y = mod(pp.y, a);
  float m2 = mod(np, 2.0);
  if (m2 > 1.0) {
    pp.y = a - pp.y;
  }
  pp.y += localTime/40.0;
  p = toRect(pp);
  p = abs(p);
  p -= vec2(0.5);
  
  float d = 10000.0;
  
  for (int i = 0; i < 4; ++i) {
    mod2(p, vec2(1.0));
    float da = -0.2 * cos(localTime*0.25);
    float sb = box(p, vec2(0.35)) + da ;
    float cb = circle(p + vec2(0.2), 0.25) + da;
    
    float dd = max(sb, -cb);
    d = min(dd, d);
    
    p *= 1.5 + 1.0*(0.5 + 0.5*sin(0.5*localTime));
    rot(p, 1.0);
  }

  
  return d;
}

vec3 mandala_postProcess(float localTime, vec3 col, vec2 uv) 
{
  float r = length(uv);
  float a = atan(uv.y, uv.x);
  col = clamp(col, 0.0, 1.0);   
  col=pow(col,mix(vec3(0.5, 0.75, 1.5), vec3(0.45), r)); 
  col=col*0.6+0.4*col*col*(3.0-2.0*col);  // contrast
  col=mix(col, vec3(dot(col, vec3(0.33))), -0.4);  // satuation
  col*=sqrt(1.0 - sin(-localTime + (50.0 - 25.0*sqrt(r))*r))*(1.0 - sin(0.5*r));
  col = clamp(col, 0.0, 1.0);
  float ff = pow(1.0-0.75*sin(20.0*(0.5*a + r + -0.1*localTime)), 0.75);
  col = pow(col, vec3(ff*0.9, 0.8*ff, 0.7*ff));
  col *= 0.5*sqrt(max(4.0 - r*r, 0.0));
  return clamp(col, 0.0, 1.0);
}

vec2 mandala_distort(float localTime, vec2 uv) {
  float lt = 0.1*localTime;
  vec2 suv = toSmith(uv);
  suv += 1.0*vec2(cos(lt), sin(sqrt(2.0)*lt));
//  suv *= vec2(1.5 + 1.0*sin(sqrt(2.0)*time), 1.5 + 1.0*sin(time));
  uv = fromSmith(suv);
  modMirror2(uv, vec2(2.0+sin(lt)));
  return uv;
}

vec3 mandala_sample(float localTime, vec2 p)
{
  float lt = 0.1*localTime;
  vec2 uv = p;
  uv *=8.0;
  rot(uv, lt);
  //uv *= 0.2 + 1.1 - 1.1*cos(0.1*iTime);

  vec2 nuv = mandala_distort(localTime, uv);
  vec2 nuv2 = mandala_distort(localTime, uv + vec2(0.0001));

  float nl = length(nuv - nuv2);
  float nf = 1.0 - smoothstep(0.0, 0.002, nl);

  uv = nuv;
  
  float d = mandala_df(localTime, uv);

  vec3 col = vec3(0.0);
 
  const float r = 0.065;

  float nd = d / r;
  float md = mod(d, r);
  
  if (abs(md) < 0.025) {
    col = (d > 0.0 ? vec3(0.25, 0.65, 0.25) : vec3(0.65, 0.25, 0.65) )/abs(nd);
  }

  if (abs(d) < 0.0125) {
    col = vec3(1.0);
  }

  col += 1.0 - pow(nf, 5.0);
  
  col = mandala_postProcess(localTime, col, uv);;
  
  col += 1.0 - pow(nf, 1.0);

  return saturate(col);
}

vec3 mandala_main(vec2 p) {

  float localTime = iTime + 30.0;
  vec3 col  = vec3(0.0);
  vec2 unit = 1.0/iResolution.xy;
  const int aa = 2;
  for(int y = 0; y < aa; ++y)
  {
    for(int x = 0; x < aa; ++x)
    {
      col += mandala_sample(localTime, p - 0.5*unit + unit*vec2(x, y));
    }
  }

  col /= float(aa*aa);
  return col;
}

vec4 mainImage()
{
  float time = 0.1*iTime;
  vec2 uv = fragCoord/iResolution.xy - vec2(0.5);
  uv.x *= iResolution.x/iResolution.y;

  vec3 col = mandala_main(uv);
    
  return vec4(col, 1.0);
}

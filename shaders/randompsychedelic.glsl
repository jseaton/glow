// CC-AT-BY paperjack
// Inspiration and original code - 'Danilo Guanabara'

#define t iTime
#define r iResolution.xy

vec2 rotateUV(vec2 uv, float rotation, vec2 mid)
{
    return vec2(
      cos(rotation) * (uv.x - mid.x) + sin(rotation) * (uv.y - mid.y) + mid.x,
      cos(rotation) * (uv.y - mid.y) - sin(rotation) * (uv.x - mid.x) + mid.y
    );
}

vec4 mainImage(){
	vec3 c;
	float l,z=t;
	for(int i=0;i<9;i++) {
		vec2 uv,p=fragCoord.xy/r;
		uv=p;
		p-=.5;
        p.x += sin(t*0.5)*0.1+p.x;
        p.y += cos(t*0.5)*0.1+p.y;
		p.x*=r.x/r.y;
		//p.y*=r.y/r.x;
        
        p = rotateUV(p, t*clamp((distance(p, vec2(0.0,0.0)))*0.0025, 0.01, 1.0)*5.0, vec2(0.0,0.0));
		z+=sin(t*0.1)*0.5;
		l=length(p)/abs(2.0+cos(t*5.0)*0.25);
		uv+=p/l*(cos(z*0.25)+1.)*abs(cos(l*11.-z-z));
		c[i]=(0.005+abs(cos(t*112.0))*0.005)/length(mod(uv,1.)-.5)*(1.+abs(sin(t))*2.0);
	}
	return vec4(c/l,t);
}

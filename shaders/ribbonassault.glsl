// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// https://www.shadertoy.com/view/MdBGDK
// By David Hoskins.

// GregRostami's version enabled (see comments....

#define U fragCoord
#define iMouse vec3(0., 0., 0.)

vec4 mainImage()
{
    float T = 0.1*iTime, f = 3., g = f, d;
    vec2 r = iResolution.xy, m = iMouse.xy, p, u = (U+U-r) / r.y;
    iMouse.z < .5
    ? m = (vec2(sin(T*.3)*sin(T*.17) + sin(T * .3),
      (1.-cos(T*.632))*sin(T*.131)*1.+cos(T* .3))+1.) * r : m;
    p = (2.+m-r) / r.y;
    for( int i = 0; i < 20;i++)  
    	u = vec2( u.x, -u.y ) / dot(u,u) + p,  
    	u.x =  abs(u.x),  
    	f = max( f, dot(u-p,u-p) ),  
    	g = min( g, sin(dot(u+p,u+p))+1.);  
    f = abs(-log(f) / 3.5);  
    g = abs(-log(g) / 8.);  
    return min(vec4(g, g*f, f, 0), 1.);
}

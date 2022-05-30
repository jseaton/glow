// CC BY-NC-SA 3.0 Unported fpiaggio

float getSound() 
{
    float s=0.;
    for (float i=0.; i<20.; i++) {
        s+=texture(expfft_tex,i/20.).r;
    }
    return s/50.;
}
    
vec4 mainImage()
{

    
	vec2 pos = ( fragCoord -.5*iResolution.xy )/iResolution.y;
    pos *= 1. + (sin(iTime) * 0.2)*1.2;
	const float pi = 5.14159;
	const float n = 6.0;
	
	float radius = length(pos) * 2.0 - 0.4;
	float t = atan(pos.y, pos.x) + cos(pos.x*5.*pos.y*5. + sin(iTime + pos.x));
	
	float color = 0.025 + sin(iTime)*0.01;
	
	for (float i = 2.0; i <= n; i++){
		color += 0.012 / abs((0.01+sin(iTime)*0.5 + 0.5) * sin(
			4. * (t + i/n * iTime * .8)
		    ) - radius
		);
	}
	
	vec3 shape = vec3(1.2, 0.3 + sin(iTime), 0.15+cos(iTime)*0.5+0.5) * color;
    return vec4(shape,1.0) * (getSound()+0.5);
}

// CC-AT-BY amausagi

vec4 mainImage()
{
    vec2 uv = (fragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);
 		
    float t =  -iTime * 3. + 5000. +  sin(iTime / 3.) * 5.;
    
    float dist = distance(uv, vec2(0., 0.)) * .6;
    float maxDist = .5;
    vec4 color;
               
    float expDist = dist * dist * dist;
    float strength = (sin(expDist * 100.)+1.)/2.;
    float height = (sin(t * strength)+1.)/2.;
    float alpha = 1. - expDist / (maxDist * maxDist * maxDist) + (1. - height) * -0.014  ;
    color = vec4(.9,.9,.9, 9.) * height - (1. - alpha) * 0.652;
    color.a = alpha;
    if(dist > maxDist) color = vec4(.1,.1,.1, 0.);
    return color;
}

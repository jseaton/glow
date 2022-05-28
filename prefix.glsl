#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 1) in float vIntensity;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler1D fft_tex;

layout(push_constant) uniform PushConstantData {
  float rms;
  float lowpass;
  float specflux[8];
  float specavg;
  float time;
} pc;

#define rms pc.rms
#define lowpass pc.lowpass
#define specflux pc.specflux
#define specavg pc.specavg

// ShaderToy compat
#define iTime pc.time
#define iResolution vec2(1280.0, 720.0)
#define fragCoord (tex_coords * iResolution)


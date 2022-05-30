// Copyright (c) 2022 Josephine Seaton and 2016 The vulkano developers
// Licensed under the MIT license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet, single_layout_pool::SingleLayoutDescSetPool, DescriptorSetWithOffsets},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount,
        SwapchainImage,
    },
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use std::convert::TryFrom;
use realfft::RealFftPlanner;
use ringbuf::RingBuffer;
use dsp::window;
use std::time::SystemTime;
use std::iter::zip;
use num_complex::Complex;
use std::io::{BufReader, BufRead, Read, Cursor};
use std::fs::File;
use std::env;

const FRAME_SIZE: usize = 512;
const FFT_SIZE: usize = FRAME_SIZE / 2 + 1;
const EXPFFT_SIZE: usize = FFT_SIZE / 8;

fn compress(x: &Complex<f32>) -> f32 {
    let mag = (x.re * x.re + x.im * x.im).sqrt();
    return (1.0 + 1.5 * mag).ln();
}

fn main() {
    let filename = env::args().nth(1).unwrap();

    let lines = BufReader::new(File::open(&filename).unwrap()).lines();

    let mut texture_names = vec![];

    for l in lines {
        if let Ok(line) = l {
            if let Some(tex_name) = line.strip_prefix("// tex ") {
                texture_names.push(tex_name.to_string());
            }
        }
    }

    let (client, _status) =
        jack::Client::new("glow", jack::ClientOptions::NO_START_SERVER | jack::ClientOptions::SESSION_ID).unwrap();
    client.set_buffer_size(FRAME_SIZE as u32).unwrap();

    let ringbuf = RingBuffer::<[f32; FRAME_SIZE]>::new(2);
    let (mut writer, mut reader) = ringbuf.split();

    let inp = client
        .register_port("rust_in", jack::AudioIn::default())
        .unwrap();
    let process_callback = move |_: &jack::Client, ps: &jack::ProcessScope| -> jack::Control {
        let ins = inp.as_slice(ps);

        let mut m = [0_f32; FRAME_SIZE];

        for i in 0..FRAME_SIZE {
            m[i] = ins[i];
        }

        let _ = writer.push(m);

        jack::Control::Continue
    };
    let process = jack::ClosureProcessHandler::new(process_callback);

    // Activate the client, which starts the processing.
    let active_client = client.activate_async((), process).unwrap();
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let mut real_planner = RealFftPlanner::<f32>::new();
    let r2c = real_planner.plan_fft_forward(FRAME_SIZE);

    let mut fft_data    = [realfft::num_complex::Complex {re: 0.0, im: 0.0}; FFT_SIZE];
    let mut expfft_data = vec![0_f32; FFT_SIZE];

    let mut image_data = [[0_f32; 2]; FFT_SIZE];

    assert_eq!(r2c.make_output_vec().len(), FFT_SIZE);

    let mut rms = 0.0;

    let mut prev_spectrum = [0_f32; FFT_SIZE];

    let win = window::hamming(FRAME_SIZE);

    let mut specavg = [0_f32; 16];
    let mut specavg_idx = 0;


    ///////////
    // Graphics
    ///////////

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: surface.window().inner_size().into(),
                image_usage: ImageUsage::color_attachment(),
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    struct Vertex {
        position: [f32; 2],
        coord:    [f32; 2],
        intensity: f32
    }
    impl_vertex!(Vertex, position, coord, intensity);

    let file = BufReader::new(File::open("standard_16x9.data").unwrap());

    let vertices = file.lines().skip(2).map(|line| {
        let l = line.unwrap().split_whitespace().map(|x| x.parse::<f32>().unwrap()).collect::<Vec<f32>>();
        Vertex { position: [l[0] / 1.777780, -l[1]], coord: [l[2], 1.0 - l[3]], intensity: l[4] }
    }).collect::<Vec<_>>();

    let width  = 100;
    let height =  60;

    let mut indices = Vec::<u16>::new();
    for i in 0..width-1 {
        for j in 0..height-1 {
            if vertices[(i*height + j) as usize].intensity > 0.1 {
                indices.push(i*height + j);
                indices.push((i+1)*height + j+1);
                indices.push(i*height + j+1);

                indices.push((i+1)*height + j+1);
                indices.push(i*height + j);
                indices.push((i+1)*height + j);
            }
        }
    }

    let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        vertices,
    )
    .unwrap();

    let index_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::index_buffer(),
        false,
        indices,
    )
    .unwrap();

    let vs = vs::load(device.clone()).unwrap();

    let fs = {
        let spirvname = filename.strip_suffix(".glsl").unwrap().to_owned() + &".spv".to_owned();
        let mut f = File::open(&spirvname)
            .expect(&("Can't find file ".to_owned() + &spirvname));
        let mut v = vec![];
        f.read_to_end(&mut v).unwrap();
        unsafe { ShaderModule::from_bytes(device.clone(), &v) }.unwrap()
    };

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let sampler = Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::MirroredRepeat; 3],
            ..Default::default()
        },
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleList))
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .color_blend_state(ColorBlendState::new(subpass.num_color_attachments()).blend_alpha())
        .render_pass(subpass)
        .build(device.clone())
        .unwrap();

    pipeline.layout().set_layouts();

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let mut pool = SingleLayoutDescSetPool::new(layout.clone());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;
    let mut push_constants = fs::ty::PushConstantData {
        rms:   0.0,
        lowpass: 0.0,
        specflux: [0.0; 8],
        specavg: 0.0,
        time: 0.0
    };

    let textures_sampler = Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::ClampToBorder; 3],
            ..Default::default()
        },
    )
    .unwrap();

    let mut texture_descriptors = vec![];

    for (i, texture_name) in texture_names.iter().enumerate() {
        let png_bytes = std::fs::read(&texture_name).unwrap();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        let dimensions = ImageDimensions::Dim2d {
            width: info.width,
            height: info.height,
            array_layers: 1,
        };
        let mut image_data = Vec::<u8>::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        let (image, future) = ImmutableImage::from_iter(
            image_data,
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            queue.clone(),
        )
        .unwrap();

        texture_descriptors.push(
            WriteDescriptorSet::image_view_sampler(
                2 + i as u32,
                ImageView::new_default(image).unwrap().clone(),
                textures_sampler.clone(),
            )
        );

        future.boxed().as_mut().cleanup_finished();
    }

    let textures_count = texture_descriptors.len();

    let textures_set = PersistentDescriptorSet::new(
        layout.clone(),
        texture_descriptors
    )
    .unwrap();

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let start_time = SystemTime::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: surface.window().inner_size().into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

                swapchain = new_swapchain;
                framebuffers =
                    window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport);
                recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let ins_m = reader.pop();

            match ins_m {
                Some(ins) => {
                    push_constants.rms   = (ins.iter().map(|x| x*x).sum::<f32>() / FRAME_SIZE as f32).sqrt();
                    push_constants.lowpass = push_constants.rms - rms;

                    rms = push_constants.rms;

                    let mut windowed = [0_f32; FRAME_SIZE];

                    win.apply(&ins, &mut windowed);

                    r2c.process(&mut windowed.to_owned(), &mut fft_data).unwrap();

                    expfft_data = zip(
                        expfft_data.iter(),
                        fft_data.chunks(8).map(|e| {
                            e.iter().map(|c| (c.re*c.re + c.im*c.im).sqrt()).sum::<f32>() / 8.0
                        })
                    ).map(|(prev, cur)| if &cur > prev { 0.1 * prev + 0.9 * cur } else { 0.5 * prev + 0.5 * cur}).collect();

                    let spectrum = fft_data.iter().map(compress).collect::<Vec<_>>();

                    let mut specflux = [0.0; 8];

                    let it = zip(prev_spectrum.iter(), spectrum.iter()).collect::<Vec<_>>();
                    let mut chunks = it.chunks(FFT_SIZE/8);

                    for i in 0..8 {
                        chunks.next().unwrap().iter().for_each(|(&p, &s)| {
                            if s > p { specflux[i] += s - p };
                        });

                        specflux[i] /= (FFT_SIZE/8) as f32;

                        push_constants.specflux[i] = push_constants.specflux[i] * 0.1 + specflux[i] * 0.9;
                    }

                    for i in 0..FFT_SIZE {
                        prev_spectrum[i] = spectrum[i];
                    }

                    let specsum = spectrum.iter().sum();

                    let average = specavg.iter().sum::<f32>() / 8.0;
                    push_constants.specavg = if specsum > average { specsum - average } else { 0.0};

                    specavg[specavg_idx] = specsum;
                    specavg_idx += 1;
                    if specavg_idx >= 16 { specavg_idx = 0; }

                    push_constants.time = (SystemTime::now().duration_since(start_time).unwrap().as_millis() as f32) / 1000.0;
                },
                None => ()
            };

            let (fft_texture, mut fft_tex_future) = {
                for i in 0..FFT_SIZE {
                    image_data[i][0] = image_data[i][0] * 0.1 + fft_data[i].re * 0.9;
                    image_data[i][1] = image_data[i][1] * 0.1 + fft_data[i].im * 0.9;
                }

                let dimensions = ImageDimensions::Dim1d {
                    width: FFT_SIZE as u32,
                    array_layers: 1
                };

                let (image, future) = ImmutableImage::from_iter(
                    image_data,
                    dimensions,
                    MipmapsCount::One,
                    Format::R32G32_SFLOAT,
                    queue.clone(),
                    )
                    .unwrap();

                (ImageView::new_default(image).unwrap(), future)
            };

            let (expfft_texture, mut expfft_tex_future) = {
                let dimensions = ImageDimensions::Dim1d {
                    width: EXPFFT_SIZE as u32,
                    array_layers: 1
                };

                let (image, future) = ImmutableImage::from_iter(
                    expfft_data.clone(),
                    dimensions,
                    MipmapsCount::One,
                    Format::R32_SFLOAT,
                    queue.clone(),
                    )
                    .unwrap();

                (ImageView::new_default(image).unwrap(), future)
            };

            let set = pool.next(
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    fft_texture.clone(),
                    sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    expfft_texture.clone(),
                    sampler.clone(),
                )],
            ).unwrap();

            let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            ).unwrap();

            let mut sets: Vec<DescriptorSetWithOffsets> = vec![DescriptorSetWithOffsets::from(set.clone())];
            if textures_count > 0 {
                sets.push(DescriptorSetWithOffsets::from(textures_set.clone()));
            }

            builder
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    clear_values,
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    sets,
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .bind_index_buffer(index_buffer.clone())
                .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();
            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
            fft_tex_future.cleanup_finished();
        }
        _ => (),
    });

    active_client.deactivate().unwrap();
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 coord;
layout(location = 2) in float intensity;
layout(location = 0) out vec2 tex_coords;
layout(location = 1) out float vIntensity;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = coord;
    vIntensity = intensity;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
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

void main() {
    f_color = texture(fft_tex, tex_coords.x);
}
"
    }
}

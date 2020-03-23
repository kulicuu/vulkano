



extern crate multiinput;

use multiinput::*;







use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use std::collections::HashMap;

use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};

use std::sync::Arc;

use std::path::Path;
use tobj;

use cgmath::{Matrix3, Matrix4, Point3, Vector3, Rad};
use std::iter;

use std::time::Instant;





#[derive(Default, Copy, Clone)]
struct Vertex {
    position: (f32, f32, f32)
}

vulkano::impl_vertex!(Vertex, position);

#[derive(Default, Copy, Clone)]
struct Normal {
    normal: (f32, f32, f32)
}

vulkano::impl_vertex!(Normal, normal);




struct Package {
    vertex_buffer: std::sync::Arc<CpuAccessibleBuffer::<[f32]>>,
    normals_buffer: std::sync::Arc<CpuAccessibleBuffer::<[f32]>>,
    index_buffer: std::sync::Arc<CpuAccessibleBuffer::<[u32]>>
}






fn main() {





    let mut manager = RawInputManager::new().unwrap();

    manager.register_devices(DeviceType::Joysticks(XInputInclude::True));




    let required_extensions = vulkano_win::required_extensions();

    let instance = Instance::new(None, &required_extensions, None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

    let dimensions: [u32; 2] = surface.window().inner_size().into();

    let queue_family = physical.queue_families().find(|&q|
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    ).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none()};


    // We can see in the triangle example the comment which explains the 0.5 -- it's a priority score for the queue.
    let (device, mut queues) = Device::new(
        physical, physical.supported_features(), &device_ext, [(queue_family, 0.5)].iter().cloned()
    ).unwrap();


    let queue = queues.next().unwrap();


    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format, dimensions, 1,
            usage, &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo,
            FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()
    };


    let lear = tobj::load_obj(&Path::new("./examples/src/bin/scratch/lear_300.obj"));
    let (models, materials) = lear.unwrap();


    let mut mashes : Vec<Package> = Vec::new();


    for (index, model) in models.iter().enumerate() {
        let mesh_500 = &model.mesh;

        mashes.push(Package {
            vertex_buffer: CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, mesh_500.positions.iter().cloned()).unwrap(),
            normals_buffer: CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, mesh_500.normals.iter().cloned()).unwrap(),
            index_buffer: CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, mesh_500.indices.iter().cloned()).unwrap()
        });
    }



    let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());
    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();


    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).unwrap()
    );


// https://docs.rs/vulkano/0.16.0/vulkano/command_buffer/struct.StateCacher.html

    let (mut pipeline, mut framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    let rotation_start = Instant::now();






    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            },
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };

                    swapchain = new_swapchain;
                    let (new_pipeline, new_framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &new_images, render_pass.clone());
                    pipeline = new_pipeline;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let elapsed = rotation_start.elapsed();
                    let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                    let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
                    let view = Matrix4::look_at(Point3::new(1., 1., 1.0), Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
                    let scale = Matrix4::from_scale(0.0021);

                    let uniform_data = vs::ty::Data {
                        world: Matrix4::from(rotation).into(),
                        view: (view * scale).into(),
                        proj: proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let layout = pipeline.descriptor_set_layout(0).unwrap();
                let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                    .add_buffer(uniform_buffer_subbuffer).unwrap()
                    .build().unwrap()
                );

                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                if suboptimal {
                    recreate_swapchain = true;
                }


                let x_input: f64;
                let y_input: f64;

                if let Some(event) = manager.get_event(){
                    match &event{
                        RawEvent::KeyboardEvent(_,  KeyId::Escape, State::Pressed)
                            => println!("aseunthaesunth"),
                        RawEvent::JoystickAxisEvent(_, axe, foo)
                            => {
                                println!("12312323 {:?} {:?}", axe, foo);
                                match *axe {
                                    Axis::X => {
                                        x_input = *foo;
                                    },
                                    Axis::Y => {
                                        y_input = *foo;
                                    },
                                    _ => (),
                                }
                            },
                        _ => (),

                    }

                    println!("event1212312 {:?}", event);

                }


                let mut cb1 = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
                .begin_render_pass(
                    framebuffers[image_num].clone(), false,
                    vec![
                        [0.0, 0.0, 1.0, 1.0].into(),
                        1f32.into()
                    ]
                ).unwrap();



                for (index, package) in mashes.iter().enumerate() {
                    cb1 = cb1
                    .draw_indexed(
                        pipeline.clone(),
                        &DynamicState::none(),
                        vec!(package.vertex_buffer.clone(), package.normals_buffer.clone()),
                        package.index_buffer.clone(), set.clone(), ()).unwrap();
                }


                let command_buffer = cb1.end_render_pass().unwrap()
                .build().unwrap();


                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();


                match future {
                   Ok(future) => {
                       previous_frame_end = Some(Box::new(future) as Box<_>);
                   },
                   Err(FlushError::OutOfDate) => {
                       recreate_swapchain = true;
                       previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                   }
                   Err(e) => {
                       println!("Failed to flush future: {:?}", e);
                       previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                   }
               }


            },
            _ => ()
        }
    });

}



/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vs::Shader,
    fs: &fs::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
) -> (Arc<dyn GraphicsPipelineAbstract + Send + Sync>, Vec<Arc<dyn FramebufferAbstract + Send + Sync>> ) {
    let dimensions = images[0].dimensions();

    let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

    let framebuffers = images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>();

    // In the triangle example we use a dynamic viewport, as its a simple example.
    // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
    // This allows the driver to optimize things, at the cost of slower window resizes.
    // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .viewports(iter::once(Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0 .. 1.0,
        }))
        .fragment_shader(fs.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    (pipeline, framebuffers)
}




mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/bin/scratch/vert.glsl"
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/bin/scratch/frag.glsl"
    }
}

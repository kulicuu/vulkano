

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace};
use vulkano::swapchain;
use vulkano::sync::GpuFuture;
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::Window;

use cgmath::{Matrix3, Matrix4, Point3, Vector3, Rad};


mod temp_verts_norms;
use temp_verts_norms::{Normal, Vertex};



use std::path::Path;
use std::ffi::OsStr;

use std::fs::File;
use std::io::BufReader;


use obj::*;

use std::env;


use std::iter::Map;
use std::iter;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let path = env::current_dir();

    println!("The current directory is {:?}", path);


    let falcon_input = BufReader::new(File::open("./examples/src/bin/peregrine/export_700.obj").unwrap());





    // println!("The Obj type {:?}", Obj);
    let falcon_model : Obj = load_obj(falcon_input).unwrap();

    // println!("Vertices {:?}", falcon_model.vertices);


    let x383 = falcon_model.indices.len();
    println!("x383 {:?}", x383);



    // let n33 = falcon_model.vertices.clone().into_iter().map(|v88| Normal {normal: v88});
    let NORMALS: Vec<Normal> = falcon_model.vertices.clone().into_iter().map(|v88| {
        // let x80 = vec!(v88.normal);
        // let norm_tuple : (f32, f32, f32) = (x80[0], x80[1], x80[2]);
        Normal { normal: (v88.normal[0], v88.normal[1], v88.normal[2]) }

        // norm_tuple;
    }).collect::<Vec<Normal>>();


    // for n77 in n33.clone().into_iter() {
    //     println!("n77 {:?}", n77);
    // }
    //
    // for v33 in falcon_model.vertices.clone().into_iter() {
    //     println!("v33 {:?}", v33);
    // }


    let VERTICES = falcon_model.vertices;
    let INDICES = falcon_model.indices;

    // let NORMALS = vec!(n33.clone());





    let extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &extensions, None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let mut events_loop = winit::EventsLoop::new();
    let surface = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();

    // unlike the triangle example we need to keep track of the width and height so we can calculate
    // render the teapot with the correct aspect ratio.
    let mut dimensions = if let Some(dimensions) = window.get_inner_size() {
        let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
        [dimensions.0, dimensions.1]
    } else {
        return;
    };

    let queue_family = physical.queue_families().find(|&q|
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    ).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };

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
            usage, &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, ColorSpace::SrgbNonLinear).unwrap()
    };

    let vertices = VERTICES.iter().cloned();
    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices).unwrap();

    let normals = NORMALS.iter().cloned();
    let normals_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), normals).unwrap();

    let indices = INDICES.iter().cloned();
    let index_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), indices).unwrap();

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

    let (mut pipeline, mut framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame = Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>;
    let rotation_start = Instant::now();

    loop {
        previous_frame.cleanup_finished();

        if recreate_swapchain {
            dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err)
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

            // note: this teapot was meant for OpenGL where the origin is at the lower left
            //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
            let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
            let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
            let view = Matrix4::look_at(Point3::new(0.2, 0.4, 0.2), Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, -0.2, 0.0));
            let scale = Matrix4::from_scale(0.045);

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

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            }
            Err(err) => panic!("{:?}", err)
        };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(
                framebuffers[image_num].clone(), false,
                vec![
                    [0.0, 0.0, 1.0, 1.0].into(),
                    1f32.into()
                ]
            ).unwrap()
            .draw_indexed(
                pipeline.clone(),
                &DynamicState::none(),
                vec!(vertex_buffer.clone(), normals_buffer.clone()),
                index_buffer.clone(), set.clone(), ()).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        let future = previous_frame.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
                future.wait(None).unwrap();
                previous_frame = Box::new(future) as Box<_>;
            }
            Err(sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::CloseRequested, .. } => done = true,
                winit::Event::WindowEvent { event: winit::WindowEvent::Resized(_), .. } => recreate_swapchain = true,
                _ => ()
            }
        });
        if done { return; }
    }
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
        path: "src/bin/teapot/vert.glsl"
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/bin/teapot/frag.glsl"
    }
}

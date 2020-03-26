



extern crate multiinput;

use multiinput::*;



extern crate srtm;

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
use std::fs::File;
use std::io::prelude::*;

use std::time::Instant;

use std::str::FromStr;

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




// This function takes a string which we hope has three floats in there separated by spaces.
// We want to return a Some(Vec<f64>) or None.
fn find_three_floats(input : &str) -> Option<Vec<f64>> {

    let x300 = String::from(input);
    let x301 : Vec<&str> = x300.split(" ").collect();


    if x301.len() == 3 {
        let x302 = f64::from_str(x301[0]);
        let x303 = f64::from_str(x301[1]);
        let x304 = f64::from_str(x301[2]);
        // println!("x302 {:?}", x302);

        if (x302.is_ok() == true) && (x304.is_ok() == true) && (x304.is_ok() == true)  {
            // println!("The values: {:?} {:?} {:?}", x302.unwrap(), x303.unwrap(), x304.unwrap());
            return Some(vec!(x302.unwrap(), x303.unwrap(), x304.unwrap()));
        }
        else {
            return None
        }
    } else {
        return None
    }
}


fn process_str_ints(input : &str) -> Vec<u32> {

    let start = String::from(input);

    let mut lines = start.lines();
    let mut condition = true;

    let mut ret_vec : Vec<u32> = Vec::new();

    while condition == true {
        let cursor = lines.next();


        if cursor == None {
            condition = false;
        } else {
            let x300 = u32::from_str(cursor.unwrap());
            if x300.is_ok() == true {
                ret_vec.push(x300.unwrap());
            } else {
                println!("error on index parse with");
            }
        }
    }
    ret_vec
}




fn process_str_floats(input: &str) -> Vec<Vec<f64>> {

    let start = String::from(input);

    let mut lines = start.lines();
    let mut condition = true;

    let mut ret_vec : Vec<Vec<f64>> = Vec::new();

    while condition == true {
        let cursor = lines.next();

        if cursor == None {
            condition = false;
        } else {
            // println!("The line: {:?}", cursor.unwrap());
            let x200 = find_three_floats(&cursor.unwrap());
            if x200 != None {
                ret_vec.push(x200.unwrap());
            }
        }

    }
    ret_vec
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




    let mut terrain_f = std::fs::File::open("./examples/src/bin/scratch/terrain_mesh_003.txt").unwrap();
    let mut terrain_buffer = String::new();



    terrain_f.read_to_string(&mut terrain_buffer).unwrap();
    let x99 : Vec<&str> = terrain_buffer.split("Vertices:").collect();

    let x100 = String::from(x99[1]);

    let x101 : Vec<&str> = x100.split("Indices:").collect();

    let x102 = String::from(x101[0]);  // This should just mostly be vertices with maybe a blank line and the title line "Vertices:"
    let x103 = String::from(x101[1]); // This should have indices and normals

    let x104 : Vec<&str> = x103.split("Normals:").collect();

    let x160 = String::from(x104[0]); // This should be indices
    let x105 = String::from(x104[1]); // This should be normals

    let x106 = process_str_floats(&x102); // This should be a vector that we can turn into a positions buffer vertex_buffer

    // println!("x106: {:?}", x106);

    let mut x200 : Vec<Vertex> = Vec::new();

    for (idx, item) in x106.iter().enumerate() {
        // println!("item {:?} idx {:?}", item, idx);
        x200.push( Vertex { position: (item[0] as f32, item[1] as f32, item[2] as f32)} );
    }

    let vertex_buffer_terrain = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, x200.iter().cloned()).unwrap();


    let x107 = process_str_floats(&x105);


    let mut x300 : Vec<Normal> = Vec::new();
    for (idx, item) in x107.iter().enumerate() {
        x300.push( Normal { normal: (item[0] as f32, item[1] as f32, item[2] as f32)} );
    }


    let normals_buffer_terrain = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, x300.iter().cloned()).unwrap();


    let x161 = process_str_ints(&x160);

    let index_buffer_terrain = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, x161.iter().cloned()).unwrap();



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
    let vsTerrain = vsTerrain::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();


    let x700 = vulkano::single_pass_renderpass!(device.clone(),
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
    ).unwrap();



    let render_pass = Arc::new(x700);


    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

// https://docs.rs/vulkano/0.16.0/vulkano/command_buffer/struct.StateCacher.html

    let (mut pipeline, mut framebuffers, mut pipelineTerrain) = window_size_dependent_setup(device.clone(), &vs, &vsTerrain, &fs, &images, render_pass.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    let rotation_start = Instant::now();




    let mut x77 : f64 = 1.0;

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


                let mut x_input: f64 = 0.0;
                let mut y_input: f64 = 0.0;

                if let Some(event) = manager.get_event(){
                    match &event{
                        RawEvent::KeyboardEvent(_,  KeyId::Escape, State::Pressed)
                            => println!("keyboard event"),
                        RawEvent::JoystickAxisEvent(_, axe, foo)
                            => {
                                // println!("12312323 {:?} {:?}", axe, foo);
                                match *axe {
                                    Axis::X => {
                                        x_input = *foo;
                                        x77 = x77 + x_input;
                                    },
                                    Axis::Y => {
                                        y_input = *foo;
                                    },
                                    _ => (),
                                }
                            },
                        _ => (),

                    }


                }

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };

                    swapchain = new_swapchain;
                    let (new_pipeline, new_framebuffers, new_pipelineTerrain) = window_size_dependent_setup(device.clone(), &vs, &vsTerrain, &fs, &new_images, render_pass.clone());
                    pipeline = new_pipeline;
                    pipelineTerrain = new_pipelineTerrain;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let elapsed = rotation_start.elapsed();

                    let x88 : f64 = elapsed.subsec_nanos() as f64;
                    let x99 : f64 = elapsed.as_secs() as f64;

                    let rotation = (x99 * x77) + ((x88 * x77) / 1_000_000_000.0);
                    let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
                    let view = Matrix4::look_at(Point3::new(1., 1., 1.0), Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
                    let scale = Matrix4::from_scale(0.0011);

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


                let mut cb20 = AutoCommandBufferBuilder::secondary_graphics(device.clone(), queue.family(), subpass.clone()).unwrap();

                cb20 = cb20
                .draw_indexed(
                    pipelineTerrain.clone(),
                    &DynamicState::none(),
                    vec!(vertex_buffer_terrain.clone(), normals_buffer_terrain.clone()),
                    index_buffer_terrain.clone(), set.clone(), ()).unwrap();

                let command_buffer_terrain = cb20.build().unwrap();



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


                unsafe {
                    cb1 = cb1.execute_commands(command_buffer_terrain).unwrap();
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
    vsTerrain: &vsTerrain::Shader,
    fs: &fs::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
) -> (Arc<dyn GraphicsPipelineAbstract + Send + Sync>, Vec<Arc<dyn FramebufferAbstract + Send + Sync>>, Arc<dyn GraphicsPipelineAbstract + Send + Sync>) {
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




    let pipelineTerrain = Arc::new(GraphicsPipeline::start()
        .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
        .vertex_shader(vsTerrain.main_entry_point(), ())
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

    (pipeline, framebuffers, pipelineTerrain)
}




mod vsTerrain {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/bin/scratch/vertTerrain.hlsl"
    }
}


mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/bin/scratch/vert.hlsl"
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/bin/scratch/frag.hlsl"
    }
}

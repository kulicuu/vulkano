



use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};

use std::sync::Arc;

use std::path::Path;
use tobj;



#[derive(Default, Copy, Clone)]
struct Vertex {
    position: (f32, f32, f32)
}

vulkano::impl_vertex!(Vertex, position);

#[derive(Default, Copy, Clone)]
pub struct Normal {
    normal: (f32, f32, f32)
}

vulkano::impl_vertex!(Normal, normal);

fn main() {



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






    let lear = tobj::load_obj(&Path::new("./examples/src/bin/scratch_000/lear_000.obj"));

    let (models, materials) = lear.unwrap();









    // let mut counter = 0;

    // we'll need the count at some point, might as well try the upfront approach first.

    let card = models.iter().count();

    println!("card {:?}", card);
    // but this number doesn't help.

    let mut vertices : Vec<Vertex> = Vec::new();
    let mut normals : Vec<Normal> = Vec::new();

    for (idx, model) in models.iter().enumerate() {
        let mesh = &model.mesh;
        // println!("mesh counter: {:?}", &counter);
        // println!("idx {:?}", idx);


        // println!("123123123: {:?}", mesh.positions);

        let vertices_count = (&mesh.positions.iter().count() + 1) / 3;
        for jdx in 0..vertices_count {
            let cursor = &mesh.positions[(jdx * 3)..((jdx * 3) + 3)];
            let normal_cursor = &mesh.normals[(jdx * 3)..((jdx * 3) + 3)];
            // println!("aeuauejjj :: 3939 {:?}", cursor[0]);
            vertices.push(Vertex { position: (cursor[0], cursor[1], cursor[2]) });
            normals.push(Normal { normal: (normal_cursor[0], normal_cursor[1], normal_cursor[2])});
        }

        // if counter == 0 {
        //     println!("Hopefully you only see this line once.");
        //     // we need to roll through this data three elements at a time making
        //     // a Vertex type
        // }


        // counter = &counter + 1;
    }

    println!("888484 {:?}", vertices.iter().count());
    // println!("Now we see count {:?}", &vertices.iter().count());

    // each mesh in models has fields :
    // positions, normals, texcoords, indices, material_id
    println!("# of models: {}", models.len());
    println!("# of materials: {}", materials.len());
    println!("aeou {:?}", models[0]);
    println!("bsnth {:?}", models[1]);




    // for (i, m) in models.iter().enumerate() {
    //     let mesh = &m.mesh;
    //
    //     println!("x333 {:?}", &mesh.positions.len());
    //
    //     // let length = &mesh.positions.len();
    //
    //     let v1: Vec<f32> = Vec::with_capacity(mesh.positions.len());
    //
    //
    //
    // }





    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());



    // let vertex_buffer = ImmutableBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertices).unwrap();






}

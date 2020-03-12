



use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
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



fn main() {



    let required_extensions = vulkano_win::required_extensions();

    let instance = Instance::new(None, &required_extensions, None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();




    let lear = tobj::load_obj(&Path::new("./examples/src/bin/scratch_000/lear_000.obj"));

    let (models, materials) = lear.unwrap();

    println!("# of models: {}", models.len());
    println!("# of materials: {}", materials.len());


    let mut x = Vec::new();
    for (i, m) in models.iter().enumerate() {
        let mesh = &m.mesh;
        x.push(&mesh.positions);

    }

    println!("x {:?}", x);









}

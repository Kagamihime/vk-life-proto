extern crate image;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano;

use image::{ImageBuffer, Rgba};

use std::sync::Arc;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::format::Format;
use vulkano::image::Dimensions;
use vulkano::image::StorageImage;
use vulkano::instance::Features;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::PhysicalDeviceType::DiscreteGpu;
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

const GRID_SIZE: u32 = 16;

mod ngs {
    #[derive(VulkanoShader)]
    #[ty = "compute"]
    #[src = "
    #version 450

    layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

    layout(set = 0, binding = 0, rgba8) uniform readonly image2D img_in;

    layout(set = 0, binding = 1, rgba8) uniform writeonly image2D img_out;

    void main() {
        ivec2 offsets[8] = { ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0), ivec2(1, 0),
                             ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1) };
        ivec2 grid_size = imageSize(img_in);
        int living_neighbors = 0;

        for (int i = 0; i < 8; i++) {
            ivec2 access_coord = ivec2(gl_GlobalInvocationID.xy) + offsets[i];

            if (access_coord.x >= 0 && access_coord.x < grid_size.x && access_coord.y >= 0 &&
                access_coord.y < grid_size.y) {
                if (imageLoad(img_in, access_coord) == vec4(1.0, 1.0, 1.0, 1.0)) {
                    living_neighbors++;
                }
            }
        }

        if (imageLoad(img_in, ivec2(gl_GlobalInvocationID.xy)) == vec4(1.0, 1.0, 1.0, 1.0)) {
            if (living_neighbors == 2 || living_neighbors == 3) {
                imageStore(img_out, ivec2(gl_GlobalInvocationID.xy), vec4(1.0, 1.0, 1.0, 1.0));
            } else {
                imageStore(img_out, ivec2(gl_GlobalInvocationID.xy), vec4(0.0, 0.0, 0.0, 0.0));
            }
        } else {
            if (living_neighbors == 3) {
                imageStore(img_out, ivec2(gl_GlobalInvocationID.xy), vec4(1.0, 1.0, 1.0, 1.0));
            } else {
                imageStore(img_out, ivec2(gl_GlobalInvocationID.xy), vec4(0.0, 0.0, 0.0, 0.0));
            }
        }
    }
    "]
    struct Dummy;
}

fn main() {
    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance)
        .find(|&dev| dev.ty() == DiscreteGpu)
        .expect("no discrete GPU available");

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = Device::new(
        physical,
        &Features::none(),
        &DeviceExtensions::none(),
        [(queue_family, 0.5)].iter().cloned(),
    ).expect("failed to create device");

    let queue = queues.next().unwrap();

    let mut grid_in = vec![0u8; (GRID_SIZE * GRID_SIZE * 4) as usize];
    for i in 0..12 {
        grid_in[5 * GRID_SIZE as usize + i] = 255u8;
    }

    let buff_in =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), grid_in.into_iter())
            .expect("failed to create buffer");

    let image_in = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: GRID_SIZE,
            height: GRID_SIZE,
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family()),
    ).expect("failed to create image");

    let image_out = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: GRID_SIZE,
            height: GRID_SIZE,
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family()),
    ).expect("failed to create image");

    let buff_out = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0..GRID_SIZE * GRID_SIZE * 4).map(|_| 0u8),
    ).expect("failed to create buffer");

    let shader = ngs::Shader::load(device.clone()).expect("failed to create shader module");
    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let set = Arc::new(
        PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
            .add_image(image_in.clone())
            .unwrap()
            .add_image(image_out.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .copy_buffer_to_image(buff_in.clone(), image_in.clone())
        .unwrap()
        .dispatch(
            [GRID_SIZE / 8, GRID_SIZE / 8, 1],
            compute_pipeline.clone(),
            set.clone(),
            (),
        )
        .unwrap()
        .copy_image_to_buffer(image_out.clone(), buff_out.clone())
        .unwrap()
        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let input_content = buff_in.read().unwrap();
    let input =
        ImageBuffer::<Rgba<u8>, _>::from_raw(GRID_SIZE, GRID_SIZE, &input_content[..]).unwrap();
    input.save("input.png").unwrap();

    let output_content = buff_out.read().unwrap();
    let output =
        ImageBuffer::<Rgba<u8>, _>::from_raw(GRID_SIZE, GRID_SIZE, &output_content[..]).unwrap();
    output.save("output.png").unwrap();
}

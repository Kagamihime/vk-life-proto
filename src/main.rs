extern crate image;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano;

use image::{ImageBuffer, Luma};

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

    layout(set = 0, binding = 0, r8) uniform readonly image2D img_in;

    layout(set = 0, binding = 1, r8) uniform writeonly image2D img_out;

    layout(set = 0, binding = 2) buffer Toroidal {
        int opt;
    } tor;

    layout(set = 0, binding = 3) buffer Survival {
        uint rules[];
    } srvl;

    layout(set = 0, binding = 4) buffer Birth {
        uint rules[];
    } brth;

    void main() {
        ivec2 offsets[8] = { ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0), ivec2(1, 0),
                             ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1) };
        ivec2 grid_size = imageSize(img_in);
        int living_neighbors = 0;

        for (int i = 0; i < 8; i++) {
            ivec2 access_coord = ivec2(gl_GlobalInvocationID.xy) + offsets[i];

            if (tor.opt != 0) {
                if (access_coord.x == -1) {
                    access_coord.x = grid_size.x - 1;
                }
                if (access_coord.y == -1) {
                    access_coord.y = grid_size.y - 1;
                }
                if (access_coord.x == grid_size.x) {
                    access_coord.x = 0;
                }
                if (access_coord.y == grid_size.y) {
                    access_coord.y = 0;
                }
            }

            if (access_coord.x >= 0 && access_coord.x < grid_size.x && access_coord.y >= 0 &&
                access_coord.y < grid_size.y) {
                if (imageLoad(img_in, access_coord).x == 1.0) {
                    living_neighbors++;
                }
            }
        }

        vec4 to_write = vec4(0.0);

        if (imageLoad(img_in, ivec2(gl_GlobalInvocationID.xy)).x == 1.0) {
            for (int i = 0; i < srvl.rules.length(); i++) {
                if (living_neighbors == srvl.rules[i]) {
                    to_write.x = 1.0;
                }
            }
        } else {
            for (int i = 0; i < brth.rules.length(); i++) {
                if (living_neighbors == brth.rules[i]) {
                    to_write.x = 1.0;
                }
            }
        }

        imageStore(img_out, ivec2(gl_GlobalInvocationID.xy), to_write);
    }
    "]
    struct Dummy;
}

mod fms {
    #[derive(VulkanoShader)]
    #[ty = "compute"]
    #[src = "
    #version 450

    layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

    layout(set = 0, binding = 0, r8) uniform readonly image2D img;

    layout(set = 0, binding = 1) buffer FlatMapX {
        int data[];
    } fmx;

    layout(set = 0, binding = 2) buffer FlatMapY {
        int data[];
    } fmy;

    void main() {
        ivec2 access_coord = ivec2(gl_GlobalInvocationID.xy);

        if (imageLoad(img, access_coord).x == 1.0) {
            fmx.data[access_coord.x] = 1;
            fmy.data[access_coord.y] = 1;
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

    let img_extended_formats_feature = Features {
        shader_storage_image_extended_formats: true,
        ..Features::none()
    };

    let (device, mut queues) = Device::new(
        physical,
        &img_extended_formats_feature,
        &DeviceExtensions::none(),
        [(queue_family, 0.5)].iter().cloned(),
    ).expect("failed to create device");

    let queue = queues.next().unwrap();

    let mut grid_in = vec![0u8; (GRID_SIZE * GRID_SIZE) as usize];
    for i in 0..3 {
        grid_in[0 * GRID_SIZE as usize + 0 + i] = 255u8;
    }
    for j in 0..3 {
        grid_in[(4 + j) * GRID_SIZE as usize + 3] = 255u8;
    }

    let buff_in =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), grid_in.into_iter())
            .expect("failed to create buffer");

    recenter_pattern(device.clone(), queue.clone(), buff_in.clone());

    let image_in = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: GRID_SIZE,
            height: GRID_SIZE,
        },
        Format::R8Unorm,
        Some(queue.family()),
    ).expect("failed to create image");

    let image_out = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: GRID_SIZE,
            height: GRID_SIZE,
        },
        Format::R8Unorm,
        Some(queue.family()),
    ).expect("failed to create image");

    let buff_out = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0..GRID_SIZE * GRID_SIZE).map(|_| 0u8),
    ).expect("failed to create buffer");

    let toroidal_opt = 1;
    let toroidal_buff =
        CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), toroidal_opt)
            .expect("failed to create buffer");

    let survival_opt: Vec<u32> = vec![2, 3];
    let survival_buff = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        survival_opt.into_iter(),
    ).expect("failed to create buffer");

    let birth_opt: Vec<u32> = vec![3];
    let birth_buff =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), birth_opt.into_iter())
            .expect("failed to create buffer");

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
            .add_buffer(toroidal_buff.clone())
            .unwrap()
            .add_buffer(survival_buff.clone())
            .unwrap()
            .add_buffer(birth_buff.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .copy_buffer_to_image(buff_in.clone(), image_in.clone())
        .unwrap()
        .dispatch(
            [
                (GRID_SIZE as f64 / 8.0).ceil() as u32,
                (GRID_SIZE as f64 / 8.0).ceil() as u32,
                1,
            ],
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
        ImageBuffer::<Luma<u8>, _>::from_raw(GRID_SIZE, GRID_SIZE, &input_content[..]).unwrap();
    input.save("input.png").unwrap();

    let output_content = buff_out.read().unwrap();
    let output =
        ImageBuffer::<Luma<u8>, _>::from_raw(GRID_SIZE, GRID_SIZE, &output_content[..]).unwrap();
    output.save("output.png").unwrap();
}

fn compute_pattern_boundaries(
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    grid_buff: Arc<CpuAccessibleBuffer<[u8]>>,
) -> (Option<usize>, Option<usize>, Option<usize>, Option<usize>) {
    let grid_img = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: GRID_SIZE,
            height: GRID_SIZE,
        },
        Format::R8Unorm,
        Some(queue.family()),
    ).expect("failed to create image");

    let flat_map_x = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0..GRID_SIZE).map(|_| 0),
    ).expect("failed to create buffer");

    let flat_map_y = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0..GRID_SIZE).map(|_| 0),
    ).expect("failed to create buffer");

    let shader = fms::Shader::load(device.clone()).expect("failed to create shader module");
    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let set = Arc::new(
        PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
            .add_image(grid_img.clone())
            .unwrap()
            .add_buffer(flat_map_x.clone())
            .unwrap()
            .add_buffer(flat_map_y.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .copy_buffer_to_image(grid_buff.clone(), grid_img.clone())
        .unwrap()
        .dispatch(
            [
                (GRID_SIZE as f64 / 8.0).ceil() as u32,
                (GRID_SIZE as f64 / 8.0).ceil() as u32,
                1,
            ],
            compute_pipeline.clone(),
            set.clone(),
            (),
        )
        .unwrap()
        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let min_x = flat_map_x.read().unwrap().iter().position(|&n| n > 0);
    let max_x = flat_map_x.read().unwrap().iter().rposition(|&n| n > 0);

    let min_y = flat_map_y.read().unwrap().iter().position(|&n| n > 0);
    let max_y = flat_map_y.read().unwrap().iter().rposition(|&n| n > 0);

    (min_x, max_x, min_y, max_y)
}

fn recenter_pattern(
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    grid_buff: Arc<CpuAccessibleBuffer<[u8]>>,
) {
    let (min_x, max_x, min_y, max_y) =
        compute_pattern_boundaries(device.clone(), queue.clone(), grid_buff.clone());

    if min_x.is_none() || max_x.is_none() || min_y.is_none() || max_y.is_none() {
        return;
    }

    let (min_x, max_x, min_y, max_y) = (
        min_x.unwrap(),
        max_x.unwrap(),
        min_y.unwrap(),
        max_y.unwrap(),
    );

    let pattern_origin = (min_x as i32, min_y as i32);
    let pattern_size = ((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32);

    let grid_img = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: GRID_SIZE,
            height: GRID_SIZE,
        },
        Format::R8Unorm,
        Some(queue.family()),
    ).expect("failed to create image");

    let centered_img = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: pattern_size.0 + 2,
            height: pattern_size.1 + 2,
        },
        Format::R8Unorm,
        Some(queue.family()),
    ).expect("failed to create image");

    let centered_buff = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        (0..(pattern_size.0 + 2) * (pattern_size.1 + 2)).map(|_| 0u8),
    ).expect("failed to create buffer");

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .clear_color_image(
            centered_img.clone(),
            vulkano::format::ClearValue::Float([0.0, 0.0, 0.0, 0.0]),
        )
        .unwrap()
        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .copy_buffer_to_image(grid_buff.clone(), grid_img.clone())
        .unwrap()
        .copy_image(
            grid_img.clone(),
            [pattern_origin.0, pattern_origin.1, 0],
            0,
            0,
            centered_img.clone(),
            [1, 1, 0],
            0,
            0,
            [pattern_size.0, pattern_size.1, 1],
            1,
        )
        .unwrap()
        .copy_image_to_buffer(centered_img.clone(), centered_buff.clone())
        .unwrap()
        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let centered_content = centered_buff.read().unwrap();
    let output = ImageBuffer::<Luma<u8>, _>::from_raw(
        (pattern_size.0 + 2) as u32,
        (pattern_size.1 + 2) as u32,
        &centered_content[..],
    ).unwrap();
    output.save("centered.png").unwrap();
}

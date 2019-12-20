#![allow(unused)]

use vulkano::instance::{
    Instance,
    InstanceExtensions,
    PhysicalDevice,
};

use vulkano::device::{
    Device,
    DeviceExtensions,
    Features,
};

use vulkano::buffer::{
    BufferUsage,
    CpuAccessibleBuffer,
};

use vulkano::command_buffer::{
    CommandBuffer,
    AutoCommandBufferBuilder,
};

use vulkano::sync::{
    GpuFuture,
};
use std::sync::Arc;
use vulkano::pipeline::ComputePipeline;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::image::{StorageImage, Dimensions};
use vulkano::format::{Format, ClearValue};

use image::{ImageBuffer, Rgba};


pub fn test()
{
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create instance.");

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available.");

    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }

    //Creating a device
    let queue_family = physical.queue_families()
        .find(|&q|q.supports_graphics())
        .expect("couldn't find a graphical queue family.");

    // Creating a device returns two things: the device itself, a list of queue objects(an iterator)
    let (device, mut queues) = {
        Device::new(
            physical, &Features::none(), &DeviceExtensions::none(),
            [(queue_family, 0.5)].iter().cloned()
        )
            .expect("failed to create device.")
    };

    let queue = queues.next().unwrap();

    // Reading and writing the contents of a buffer

    // Once create a buffer, we can `read()` or `write()` buffer contents
    // while `read()` will grant you shared access to the content of the buffer,
    // and `write()` will grant you exclusive access, which similar to using a `RwLock`.

    // #Type1

    let data = 12;
    let buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), data)
        .expect("failed to create buffer.");

    // In order to tell compiler write close, we must enclose write operate in curly braces
    {
        let mut content = buffer.write().unwrap();
        *content = 15;
    }

    let content = &*buffer.read().unwrap();
    println!("Read and Write buffer, Type1 content: {:?}", content);

    // #Type2

    // From_data and from_iter
    struct MyStruct {
        a: u32,
        b: bool,
    }

    let data = MyStruct{a:5, b: true};
    let buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), data)
        .unwrap();

    // In order to tell compiler write close, we must enclose write operate in curly braces
    {
        // `content` implements `DerefMut` whose target is of type `MyStruct` (the content of the buffer)
        let mut content = buffer.write().unwrap();
        content.a *= 2;
        content.b = false;
    }

    let content = &*buffer.read().unwrap();
    println!("Read and Write buffer, Type2 content: MyStruct:{{ a:{:?}, b:{:?} }}", content.a, content.b);

    // #Type3

    // value 5 of type u8, 128 times
    let iter = (0..128).map(|_| 5u8);
    let buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), iter)
        .unwrap();

    // Alternatively, suppose that the content of buffer is of type [u8] (like with the example that uses from_iter):
    // this time `content` derefs to `[u8]`
    // in Rust, represents an array of u8s whose size is only known at runtime
    // In order to tell compiler write close, we must enclose write operate in curly braces
    {
        let mut content = buffer.write().unwrap();
        content[12] = 83;
        content[7] = 3;
    }
    let content:&[u8] = &*buffer.read().unwrap();
    println!("Read and Write buffer, Type3 content: {:?}", content);


    // Example operation: ask the GPU to actually do something

    // Creating the buffers
    let source_content = 0..64;
    let source = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), source_content)
        .unwrap();

    let dest_content = (0..64).map(|_| 0);
    let dest = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), dest_content)
        .unwrap();

    // Command buffers: ask the GPU to perform an operation

    // Submitting a command to the GPU can take up to several hundred microseconds,
    // which is why we submit as many things as we can at once.

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .copy_buffer(source.clone(), dest.clone()).unwrap()
        .build().unwrap();

    // Submission and synchronization

    // `finished` is an object that represents the execution of the command buffer
    let finished = command_buffer.execute(queue.clone()).unwrap();
    // Only after this is done can we call `destination.read()` and check that our copy succeeded.
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
    let src_content = source.read().unwrap();
    let dest_content = dest.read().unwrap();
    assert_eq!(&*src_content, &*dest_content);

    // Introduction to compute operations

    //  perform an operation with multiple values at once,
    let data_iter = 0..65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data_iter)
        .unwrap();

    // A program that runs on the GPU is called a shader.
    // 1. write the source code of the program in a programming language called GLSL.
    //   Vulkano will compile the GLSL code at compile-time into intermediate representation called SPIR-V.
    // 2. At runtime we pass this SPIR-V to the Vulkan implementation,
    //   which in turn converts it into its own implementation-specific format.
    // Note: In the very far future it may be possible to write shaders in Rust, or in a domain specific language that resembles Rust.

    // compile GLSL and generate several structs and methods
    // including one named Shader that provides a method named load.
    mod cs {
        vulkano_shaders::shader!{
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
}
        "
        }
    }

    let shader = cs::Shader::load(device.clone())
        .unwrap();

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
                .expect("failed to create compute pipeline.")
    );

    // Creating a descriptor set

    // device is determined from `compute_pipeline`.
    let set = Arc::new(
        PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
            .add_buffer(data_buffer.clone()).unwrap()
            .build().unwrap()
    );

    // Dispatch: create the command buffer that will execute our compute pipeline.

    // spawn 1024 work groups
    // The last parameter contains the push constants, which are a way to pass a small amount of data to a shader.
    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue_family).unwrap()
        .dispatch([1024, 1, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
        .build().unwrap();

    // submit the command buffer
    let finished = command_buffer.execute(queue.clone()).unwrap();
    // wait for it to complete
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    //check that the pipeline has been correctly executed
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");


    // Creating an image

    // Properties of an image: in the context of Vulkan images can be one to three dimensional.
    // The dimensions of an image are chosen when you create it.
    // There are two kinds of three-dimensional images: actual three-dimensional images, and arrays of two-dimensional layers.
    // The difference is that with the former the layers are expected to be contiguous, while for the latter you can manage layers individually as if they were separate two-dimensional images.

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d {width:1024, height:1024},
        Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    // Clearing an image:  ask the GPU to fill our image with a specific color.

    // Exporting the content of an image

    // Copying from the image to the buffer
    // The buffer has to be large enough
    // Each pixel of the image contains four unsigned 8-bit values, and the image dimensions are 1024 by 1024 pixels.
    // Hence why the number of elements in the buffer is 1024 * 1024 * 4.

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                             (0..1024*1024*4).map(|_| 0u8))
        .expect("failed to create buffer");

    // Note: The function is called clearing a color image, as opposed to depth and/or stencil images which we haven't covered yet.
    // The image was created with the R8G8B8A8Unorm format.
    // The R8G8B8A8 part means that the four components are stored in 8 bits each, while the Unorm suffix means "unsigned normalized".
    // The coordinates being "normalized" means that their value in memory (ranging between 0 and 255) is interpreted as floating point values.
    // The in-memory value 0 is interpreted as the floating-point 0.0, and the in-memory value 255 is interpreted as the floating-point 1.0.
    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0])).unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
        .build().unwrap();


    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    // Turning the image into a PNG
    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();


    mod mandelrot_set {
        // Draw a Mandelbrot set
        vulkano_shaders::shader!{
            ty: "compute",
            src: r#"
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set=0, binding=0, rgba8) uniform writeonly image2D img;

void main() {
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    vec2 z = vec2(0.0, 0.0);
    float i;
    for (i=0.0; i<1.0; i+=0.005) {
        z = vec2(
            z.x*z.x - z.y*z.y + c.x,
            z.y*z.x + z.x*z.y + c.y
        );

        if (length(z)>4.0) {
            break;
        }
    }

    vec4 to_write = vec4(vec3(i), 1.0);
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}
            "#
        }
    }

    let shader = mandelrot_set::Shader::load(device.clone())
        .unwrap();

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline.")
    );

    // Calling this shader
    let image = StorageImage::new(device.clone(), Dimensions::Dim2d {width:1024, height:1024},
        Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    // This time we use the add_image function instead of add_buffer
    let set = Arc::new(
       PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
           .add_image(image.clone()).unwrap()
           .build().unwrap()
    );

    // create a buffer where to write the output
    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                             (0..1024*1024*4).map(|_| 0u8))
        .expect("failed to create buffer");

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([1024/8, 1024/8, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
        .build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
}
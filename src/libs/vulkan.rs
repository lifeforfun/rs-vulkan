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

}
#![allow(unused)]

use vulkano::{
    descriptor::{
        descriptor_set::PersistentDescriptorSet,
    },
    device::{
        Device,
        DeviceExtensions,
        Features,
    },
    buffer::{
        BufferUsage,
        CpuAccessibleBuffer,
    },
    command_buffer::{
        CommandBuffer,
        AutoCommandBufferBuilder,
        DynamicState,
    },
    format::{
        Format,
        ClearValue,
    },
    framebuffer::{
        Framebuffer,
        Subpass,
    },
    image::{
        StorageImage,
        Dimensions,
    },
    instance::{
        Instance,
        InstanceExtensions,
        PhysicalDevice,
    },
    pipeline::{
        ComputePipeline,
        shader::GraphicsShaderType::Vertex,
        GraphicsPipeline,
        viewport::Viewport,
    },
    swapchain::{
        Swapchain,
        SurfaceTransform,
        PresentMode,
    },
    sync::{
        GpuFuture,
    },
};

use std::sync::Arc;

use image::{ImageBuffer, Rgba};

use vulkano_win::VkSurfaceBuild;

use winit::{
    EventsLoop,
    WindowBuilder,
};


pub fn test()
{
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None)
            .expect("failed to create instance.")
    };

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
        let device_ext = DeviceExtensions{
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        Device::new(
            physical, &Features::none(), &device_ext,
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
    image.save("image1.png").unwrap();


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
    let image_buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image_buffer.save("image2.png").unwrap();

    // Graphics pipeline introduction: Using the graphics pipeline is more restrictive than using compute operations, but it is also much faster

    // start by executing a `vertex shader` (that is part of the graphics pipeline object)
    // then executes a `fragment shader` (also part of the graphics pipeline object)

    // Vertex buffer
    #[derive(Default, Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
    }

    vulkano::impl_vertex!(Vertex, position);

    let vertex1 = Vertex{position: [-0.5, -0.5]};
    let vertex2 = Vertex{position: [0.0, 0.5]};
    let vertex3 = Vertex{position: [0.5, -0.25]};

    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
        vec![vertex1, vertex2, vertex3].into_iter()).unwrap();

    // Vertex shader: GPU pick each element from this buffer

    // Note: Calling the impl_vertex! macro is what makes it possible for vulkano to build the link between the content of the buffer and the input of the vertex shader.
    // The line layout(location = 0) in vec2 position; declares that each vertex has an attribute named position and of type vec2.
    // This corresponds to the definition of the Vertex struct we created.
    // The main function is called once for each vertex
    // gl_Position is a special "magic" global variable that exists only in the context of a vertex shader and whose value must be set to the position of the vertex on the surface.
    // This is how the GPU knows how to position our shape.
    mod vertex_shader {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: r#"
                #version 450

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            "#
        }
    }

    let vertex_shader = vertex_shader::Shader::load(device.clone())
        .expect("failed to create vertex shader");

    // Fragment shader

    // Only pixels that within the shape of the triangle will be modified on the final image.
    // The `layout(location = 0) out vec4 f_color;` line declares an output named f_color.
    mod fragment_shader {
        vulkano_shaders::shader!{
            ty: "fragment",
            src: r#"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            "#
        }
    }

    let fragment_shader = fragment_shader::Shader::load(device.clone())
        .expect("failed to create fragment shader");

    // Render passes: In order to fully optimize and parallelize commands execution, it is only once we have entered a special "rendering mode"  called render pass that you can draw.

    // Creating a render pass: A render pass is made of attachments and passes
    let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear, // we want the GPU to clear the image when entering the render pass (ie. fill it with a single color)
                store: Store, // we want the GPU to actually store the output of our draw commands to the image
                format: Format::R8G8B8A8Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap());

    // Entering the render pass

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d {width:1024, height:1024},
        Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                             (0..1024*1024*4).map(|_| 0u8))
        .expect("failed to create buffer");

    // creating a framebuffer to indicate the actual list of attachments
    let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
        .add(image.clone()).unwrap()
        .build().unwrap()
    );
    // ready the enter drawing mode!
    AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
        .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
        .end_render_pass().unwrap();

    let pipeline = Arc::new(GraphicsPipeline::start()
        // Defines what kind of vertex input is expected.
        .vertex_input_single_buffer::<Vertex>()
        // The vertex shader.
        .vertex_shader(vertex_shader.main_entry_point(), ())
        // Defines the viewport (explanations below).
        // If the viewport state wasn't dynamic, then we would have to create a new pipeline object if we wanted to draw to another image of a different size.
        // Note: If you configure multiple viewports, you can use geometry shaders to choose which viewport the shape is going to be drawn to.
        .viewports_dynamic_scissors_irrelevant(1)
        // The fragment shader.
        .fragment_shader(fragment_shader.main_entry_point(), ())
        // This graphics pipeline object concerns the first pass of the render pass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())

        .build(device.clone()).unwrap()
    );

    // Drawing
    let dynamic_state = DynamicState{
        viewports: Some(vec![Viewport{
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0..1.0,
        }]),
        ..DynamicState::none()
    };

    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
        .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
        .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ()).unwrap()
        .end_render_pass().unwrap()

        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
        .build().unwrap()
    ;

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("triangle.png").unwrap();


    // Windowing: draw graphics on it,
    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    // Events handling
    events_loop.run_forever(|evt|{
        match evt {
            winit::WindowEvent{event: winit::WindowEvent::CloseRequested, ..} => {
                winit::ControlFlow::Break
            },
            _ => winit::ControlFlow::Continue,
        }
    });

    // Swapchains

    // Creating a swapchain
    // first query the capabilities of the surface
    let caps = surface.capabilities(physical)
        .expect("failed to get surface capabilities");

    // choose the dimensions of the image (minimum and a maximum)
    let dimensions = caps.current_extent().unwrap_or([1280, 1024]);
    // the behavior when it comes to transparency
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    // the format of the images
    let format = caps.supported_formats[0].0;

    //create the swapchain
    let (swapchain, images) = Swapchain::new(device.clone(), surface.clone(),
        caps.min_image_count, format, dimensions, 1, caps.supported_usage_flags, &queue,
        SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, None)
        .expect("failed to create swapchain");

    // Usage of a swapchain
    let (image_num, acquire_future) = swapchain::acquire_next_image(swapchain.clone()).unwrap();
}
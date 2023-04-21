struct Params
{
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;
    unsigned int max_depth;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    OptixTraversableHandle handle;

    // Time and mouse position for dynamic scene changes
    // Update the value every frame on the host side (CPU) and transfer it to the device side (GPU)
    float time;
    float mouse_x;
    float mouse_y;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};

struct SphereData
{
    // center of the sphere
    float3 center;
    // sphere radius
    float radius;
};

struct MeshData
{
    // vertices of the mesh
    float3* vertices;
    // Vertex number 3 points for constructing a triangle
    uint3* indices;
};

struct LambertianData {
    // Lambert material color
    void* texture_data;
    unsigned int texture_prg_id;
};

struct DielectricData {
    // dielectric color
    void* texture_data;
    unsigned int texture_prg_id;
    // refractive index
    float ior; 
};

struct MetalData {
    // metal color
    void* texture_data;
    unsigned int texture_prg_id;
    // A parameter that specifies the pseudo-roughness of a metal
    float fuzz;
};

struct Material {
    // Materials (Lambertian, Glass, Metal) data
    // Bind the pointer allocated on the device
    // By making it a general-purpose pointer without using a union,
    // It becomes easier to handle when you want to add structures of different data types.
    void* data; 

    // ID of a Callables program for calculating scattering directions and colors in materials
    // Since OptiX 7.x cannot use virtual functions, use a Callables program to
    // Realize pseudo-polymorphism
    unsigned int prg_id;
};

struct ConstantData
{
    float4 color;
};

struct CheckerData
{
    float4 color1; 
    float4 color2;
    float scale;
};

struct HitGroupData
{
    // Data on object shape
    // Bind the pointer allocated on the device
    void* shape_data;

    Material material;
};

struct EmptyData
{

};
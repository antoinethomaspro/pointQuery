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

    // 動的にシーンを変更するための時間とマウス位置
    // ホスト側(CPU)で毎フレーム値を更新して、デバイス側(GPU)に転送する
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
    // 球の中心
    float3 center;
    // 球の半径
    float radius;
};

struct MeshData
{
    // メッシュの頂点
    float3* vertices;
    // 三角形を構成するための頂点番号3点
    uint3* indices;
};

struct LambertianData {
    // Lambert マテリアルの色
    void* texture_data;
    unsigned int texture_prg_id;
};

struct DielectricData {
    // 誘電体の色
    void* texture_data;
    unsigned int texture_prg_id;
    // 屈折率
    float ior; 
};

struct MetalData {
    // 金属の色
    void* texture_data;
    unsigned int texture_prg_id;
    // 金属の疑似粗さを指定するパラメータ
    float fuzz;
};

struct Material {
    // マテリアル(Lambertian, Glass, Metal)のデータ
    // デバイス上に確保されたポインタを紐づけておく
    // 共用体(union)を使わずに汎用ポインタにすることで、
    // 異なるデータ型の構造体を追加したいときに対応しやすくなる。
    void* data; 

    // マテリアルにおける散乱方向や色を計算するためのCallablesプログラムのID
    // OptiX 7.x では仮想関数が使えないので、Callablesプログラムを使って
    // 疑似的なポリモーフィズムを実現する
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
    // 物体形状に関するデータ
    // デバイス上に確保されたポインタを紐づける
    void* shape_data;

    Material material;
};

struct EmptyData
{

};
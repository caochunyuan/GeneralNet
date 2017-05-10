#include <metal_stdlib>

using namespace metal;

kernel void adjust_mean_rgb(texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<float, access::write> outTexture [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  float4 inColor = inTexture.read(gid);
  float4 outColor = float4(inColor.z*255.0 - 120.0, inColor.y*255.0 - 120.0, inColor.x*255.0 - 120.0, 0.0);
  outTexture.write(outColor, gid);
}

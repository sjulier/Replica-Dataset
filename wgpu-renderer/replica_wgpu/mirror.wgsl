// Mirrors and glass: a port of ReplicaSDK/shaders/mirror.{vert,frag}.
//
// Draws a surface's bounding rectangle over the image, showing the reflected
// scene (rendered beforehand into `reflection`, same size as the target) with
// alpha = Fresnel term x polygon mask, blended with src-alpha / one-minus-src-alpha.
// fs_mirror_depth optionally replaces the depth image inside a mirror's polygon:
// with the reflected scene's depth (what a depth sensor sees in a mirror) or 0
// (a sensor dropout). By default the depth image keeps the surface, as ReplicaRenderer.

struct Mirror {
    mvp: mat4x4<f32>,
    mv: mat4x4<f32>,
    corners: array<vec4<f32>, 4>,  // world-space rectangle, uv (0,0), (1,0), (1,1), (0,1)
    normal: vec4<f32>,             // world-space plane normal (xyz)
    reflectivity: f32,
    depth_mode: u32,               // fs_mirror_depth: 0 = reflected depth, 1 = invalid (0)
};

@group(0) @binding(0) var<uniform> mirror: Mirror;
@group(0) @binding(1) var reflection: texture_2d<f32>;
@group(0) @binding(2) var reflection_depth: texture_2d<f32>;
@group(0) @binding(3) var mask: texture_2d<f32>;
@group(0) @binding(4) var mask_sampler: sampler;

var<private> CORNER: array<u32, 6> = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
var<private> UV: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) eye_pos: vec3<f32>,
};

@vertex
fn vs_mirror(@builtin(vertex_index) vi: u32) -> VsOut {
    let k = CORNER[vi];
    let p = mirror.corners[k];
    var out: VsOut;
    out.clip = mirror.mvp * p;
    out.uv = UV[k];
    out.eye_pos = (mirror.mv * p).xyz;
    return out;
}

// Both entry points write both targets; each pipeline masks out the one it leaves alone.
struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
};

@fragment
fn fs_mirror(in: VsOut) -> FsOut {
    let mv3 = mat3x3<f32>(mirror.mv[0].xyz, mirror.mv[1].xyz, mirror.mv[2].xyz);
    let n = normalize(mv3 * mirror.normal.xyz);
    let v = normalize(in.eye_pos);
    let r = mirror.reflectivity;
    let fresnel = r + (1.0 - r) * pow(1.0 - abs(dot(n, v)), 5.0);
    let c = textureLoad(reflection, vec2<i32>(in.clip.xy), 0);
    let m = textureSample(mask, mask_sampler, in.uv).r;
    return FsOut(vec4<f32>(c.rgb, fresnel * m), 0.0);
}

@fragment
fn fs_mirror_depth(in: VsOut) -> FsOut {
    let m = textureSample(mask, mask_sampler, in.uv).r;
    if (m < 0.5) {
        discard;
    }
    var d = 0.0;
    if (mirror.depth_mode == 0u) {
        d = textureLoad(reflection_depth, vec2<i32>(in.clip.xy), 0).r;
    }
    return FsOut(vec4<f32>(0.0), d);
}

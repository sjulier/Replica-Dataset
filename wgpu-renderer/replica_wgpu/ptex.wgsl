// Replica ptex rendering in WGSL.
//
// A port of ReplicaSDK/shaders/{mesh-ptex.*, atlas.glsl, mesh-depth.*} that needs
// no geometry shader (unavailable on Metal / MoltenVK): the vertex shader pulls
// each quad's corners from storage buffers and emits two triangles in the same
// order and with the same UVs as mesh-ptex.geom. Colour and metric depth are
// written in one pass.

struct Frame {
    mvp: mat4x4<f32>,
    mv: mat4x4<f32>,
    exposure: f32,
    inv_gamma: f32,
    saturation: f32,
    tile_size: i32,
};

struct SubMeshInfo {
    width_in_tiles: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
};

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var<uniform> sub: SubMeshInfo;
@group(1) @binding(1) var<storage, read> positions: array<f32>;  // xyz per vertex
@group(1) @binding(2) var<storage, read> quads: array<u32>;      // 4 indices per face
@group(1) @binding(3) var<storage, read> adjacency: array<u32>;  // 4 entries per face
@group(1) @binding(4) var atlas: texture_2d<f32>;

// Triangle-strip order of mesh-ptex.geom (v1, v0, v2, v3) as a triangle list.
var<private> CORNER: array<u32, 6> = array<u32, 6>(1u, 0u, 2u, 2u, 0u, 3u);
var<private> UV: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) face: u32,
    @location(2) depth: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let face = vi / 6u;
    let corner = CORNER[vi % 6u];
    let v = quads[face * 4u + corner];
    let p = vec4<f32>(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u], 1.0);
    var out: VsOut;
    out.clip = frame.mvp * p;
    out.uv = UV[corner];
    out.face = face;
    out.depth = (frame.mv * p).z;
    return out;
}

// ---- atlas.glsl ---------------------------------------------------------------

const ROTATION_SHIFT = 30u;
const FACE_MASK = 0x3FFFFFFFu;

fn face_to_atlas_pos(face: i32, tsize: i32) -> vec2<i32> {
    let y = face / sub.width_in_tiles;
    let x = face - y * sub.width_in_tiles;
    return vec2<i32>(x, y) * tsize;
}

// rot = number of 90 degree anti-clockwise rotations
fn rotate_uvs(p: vec2<i32>, rot: i32, size: i32) -> vec2<i32> {
    switch rot {
        case 1: { return vec2<i32>(p.y, (size - 1) - p.x); }
        case 2: { return vec2<i32>((size - 1) - p.x, (size - 1) - p.y); }
        case 3: { return vec2<i32>((size - 1) - p.y, p.x); }
        default: { return p; }
    }
}

struct Adj {
    face: i32,
    rot: i32,
    valid: bool,
};

fn get_adj_face(face: i32, edge: i32) -> Adj {
    let data = adjacency[u32(face * 4 + edge)];
    let f = data & FACE_MASK;
    return Adj(i32(f), i32(data >> ROTATION_SHIFT), f != FACE_MASK);
}

struct Lookup {
    face: i32,
    p: vec2<i32>,
};

// Across a corner: step into the edge neighbour, then into its neighbour.
fn corner_step(adj: Adj, p_in: vec2<i32>, tsize: i32, fallback: Lookup) -> Lookup {
    var p = p_in;
    if (p.x > tsize - 1) {
        p.x -= tsize;
        p = rotate_uvs(p, adj.rot, tsize);
        let adj2 = get_adj_face(adj.face, (1 - adj.rot) & 3);
        if (adj2.valid) {
            return Lookup(adj2.face, rotate_uvs(p, adj2.rot, tsize));
        }
        return fallback;
    }
    if (p.x < 0) {
        p.x += tsize;
        p = rotate_uvs(p, adj.rot, tsize);
        let adj2 = get_adj_face(adj.face, (3 - adj.rot) & 3);
        if (adj2.valid) {
            return Lookup(adj2.face, rotate_uvs(p, adj2.rot, tsize));
        }
        return fallback;
    }
    return Lookup(adj.face, rotate_uvs(p, adj.rot, tsize));
}

// p is an integer tile coordinate; if it lies outside the tile, fetch from the
// right adjacent tile, taking account of rotation (indexAdjacentFaces).
//
// In the GLSL original p is `inout` and is modified even on paths that end up
// keeping the original face; the fallbacks below reproduce those values.
fn index_adjacent_faces(face: i32, p_in: vec2<i32>, tsize: i32) -> Lookup {
    var p = p_in;
    if (p.y < 0) {
        let adj = get_adj_face(face, 0);
        if (adj.valid) {
            p.y += tsize;
            var q = p;
            if (q.x > tsize - 1) { q.x -= tsize; } else if (q.x < 0) { q.x += tsize; }
            return corner_step(adj, p, tsize, Lookup(face, rotate_uvs(q, adj.rot, tsize)));
        }
    } else if (p.y > tsize - 1) {
        let adj = get_adj_face(face, 2);
        if (adj.valid) {
            p.y -= tsize;
            var q = p;
            if (q.x > tsize - 1) { q.x -= tsize; } else if (q.x < 0) { q.x += tsize; }
            return corner_step(adj, p, tsize, Lookup(face, rotate_uvs(q, adj.rot, tsize)));
        }
    } else if (p.x < 0) {
        let adj = get_adj_face(face, 3);
        if (adj.valid) {
            p.x += tsize;
            return Lookup(adj.face, rotate_uvs(p, adj.rot, tsize));
        }
    } else if (p.x > tsize - 1) {
        let adj = get_adj_face(face, 1);
        if (adj.valid) {
            p.x -= tsize;
            return Lookup(adj.face, rotate_uvs(p, adj.rot, tsize));
        }
    }
    return Lookup(face, p);
}

fn texel_fetch_atlas_adj(face: i32, p: vec2<i32>) -> vec4<f32> {
    let tsize = frame.tile_size;
    let l = index_adjacent_faces(face, p, tsize);
    let q = clamp(l.p, vec2<i32>(0, 0), vec2<i32>(tsize - 1, tsize - 1));
    return textureLoad(atlas, face_to_atlas_pos(l.face, tsize) + q, 0);
}

// bilinear filtering across tile borders
fn texture_atlas(face: i32, p_in: vec2<f32>) -> vec4<f32> {
    let p = p_in - 0.5;
    let i = vec2<i32>(floor(p));
    let f = p - vec2<f32>(i);
    return mix(
        mix(texel_fetch_atlas_adj(face, i), texel_fetch_atlas_adj(face, i + vec2<i32>(1, 0)), f.x),
        mix(texel_fetch_atlas_adj(face, i + vec2<i32>(0, 1)), texel_fetch_atlas_adj(face, i + vec2<i32>(1, 1)), f.x),
        f.y);
}

fn apply_saturation(c: vec3<f32>, s: f32) -> vec3<f32> {
    let pw = sqrt(c.r * c.r * 0.299 + c.g * c.g * 0.587 + c.b * c.b * 0.114);
    return vec3<f32>(pw) + (c - vec3<f32>(pw)) * s;
}

// ---- outputs -----------------------------------------------------------------

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
};

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var c = texture_atlas(i32(in.face), in.uv * f32(frame.tile_size)).rgb;
    c *= frame.exposure;
    c = apply_saturation(c, frame.saturation);
    c = pow(max(c, vec3<f32>(0.0)), vec3<f32>(frame.inv_gamma));
    var out: FsOut;
    out.color = vec4<f32>(c, 1.0);
    out.depth = in.depth;
    return out;
}

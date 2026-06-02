#!/usr/bin/env python3
"""Convert USD scene to MuJoCo MJCF format with textures.

Each Mesh prim is split into per-GeomSubset OBJ files so every material region
gets its own geom with the correct diffuse texture. Vertices are transformed to
world space via XformCache. UV coordinates (primvars:st, faceVarying) are
included; faces are written as v/vt pairs. Diffuse textures are copied to the
output textures/ directory.

Naming:
  - Actor with 1 total geom     → geom name = actor_name  (e.g. window_0)
  - Actor with N>1 total geoms  → geom name = actor_name_0, actor_name_1, …

Articulation:
  Actors that contain PhysicsRevoluteJoint / PhysicsPrismaticJoint / PhysicsFixedJoint
  children (or a 'joints' Scope child) are treated as articulated.  Each direct Xform
  child of the Actor becomes a separate <body>.  PhysicsFixedJoint children get their
  own <body> element with no <joint> (= rigidly fixed to parent in MuJoCo).
  Joint positions are expressed in actor-local frame.
"""

import argparse
import os
import re
import time
from PIL import Image
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf

MESH_SUBDIR = 'meshes'
TEX_SUBDIR  = 'textures'
TEX_EXTS    = {'.png', '.jpg', '.jpeg', '.tga', '.bmp'}

# Simplification settings.
# Only meshes with more than SIMPLIFY_MIN_FACES faces are simplified.
# Meshes at or below the threshold keep their full geometry and UV coordinates.
# SIMPLIFY_RATIO: fraction of faces to keep after decimation (e.g. 0.1 = 10 %).
# Set SIMPLIFY_RATIO = 0.0 to disable simplification entirely.
SIMPLIFY_MIN_FACES = 10_000   # meshes with fewer faces are kept as-is (with UVs)
SIMPLIFY_RATIO     = 0.2      # keep 10 % of faces for meshes above the threshold

# Collision settings.
# Meshes whose AABB min-dimension / max-dimension ratio is below this threshold
# are classified as "shell" (thin/flat): they get inertia="shell" in the asset
# and an AABB box collision geom.  All other meshes use the mesh itself as the
# collision shape (MuJoCo takes the convex hull).
SHELL_ASPECT_RATIO = 0.15
# Minimum half-size (metres) clamped onto every dimension of an AABB box.
# Prevents zero-size boxes when a mesh is exactly planar (e.g. walls, floors).
SHELL_MIN_HALF = 0.005


def _to_camel(s):
    """Convert underscore_separated or dash-separated string to CamelCase.
    Numbers are kept as-is and concatenated without separator.
    Examples: 'dining_table' → 'DiningTable', 'cabinet_0' → 'Cabinet0'.
    """
    parts = s.replace('-', '_').split('_')
    return ''.join(p[0].upper() + p[1:] if p and p[0].isalpha() else p for p in parts if p)


def _to_snake(s):
    """Convert CamelCase, dash- or underscore-separated string to lowercase snake_case.
    A separator is inserted before uppercase letters and before digit runs.
    Examples: 'DiningTable' → 'dining_table', 'Book45' → 'book_45',
              'KitchenCabinet' → 'kitchen_cabinet', 'Cabinet0' → 'cabinet_0'.
    """
    s = s.replace('-', '_')
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)      # lowercase/digit → uppercase
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)  # consecutive caps before lowercase
    s = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', s)        # letter → digit run: Book45 → book_45
    return s.lower()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def world_verts(mesh_prim, xform_cache):
    """Return (N, 3) float32 world-space vertices for a Mesh prim."""
    pts = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    if pts is None:
        return None
    pts_np = np.array([(p[0], p[1], p[2]) for p in pts], dtype=np.float64)
    xf = xform_cache.GetLocalToWorldTransform(mesh_prim)
    M  = np.array([[xf[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    h  = np.hstack([pts_np, np.ones((len(pts_np), 1))])
    return (h @ M)[:, :3].astype(np.float32)   # USD row-vector: world = local @ M


def actor_pose(actor_prim, xform_cache):
    """Return (pos_np, quat_wxyz_np, M_pose_inv_np) for an actor prim in world space.

    pos          – translation (x, y, z) of the actor origin in world space
    quat_wxyz    – rotation as (w, x, y, z), scale stripped out
    M_pose_inv   – inverse of the pose-only (rotation + translation, NO scale) matrix.
                   Applying this to world-space vertices gives body-local vertices with
                   scale baked in, so the actor's scale is preserved in the OBJ geometry.
    """
    xf = xform_cache.GetLocalToWorldTransform(actor_prim)
    M  = np.array([[xf[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    pos = M[3, :3].copy()

    # Decompose rotation (strips scale/shear) via Gf.Transform
    t = Gf.Transform()
    t.SetMatrix(xf)
    q    = t.GetRotation().GetQuat()
    imag = q.GetImaginary()
    quat = np.array([q.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)

    # Build a pose-only (scale-free) matrix from the normalised rotation rows + translation.
    # Inverting this undoes only translation and rotation, leaving scale baked into the
    # vertex positions so MuJoCo renders the mesh at the correct size.
    R = M[:3, :3].copy()
    R /= np.linalg.norm(R, axis=1, keepdims=True)   # strip scale from each axis row
    M_pose = np.eye(4, dtype=np.float64)
    M_pose[:3, :3] = R
    M_pose[3, :3]  = pos
    return pos, quat, np.linalg.inv(M_pose)


def _quat_mul(q1, q2):
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2], dtype=np.float64)


def mesh_topology(mesh_prim):
    """Return (face_counts, face_vert_indices, face_vert_start) or (None,None,None)."""
    m  = UsdGeom.Mesh(mesh_prim)
    fc = m.GetFaceVertexCountsAttr().Get()
    fi = m.GetFaceVertexIndicesAttr().Get()
    if fc is None or fi is None:
        return None, None, None
    fc_np  = np.array(fc, dtype=np.int32)
    fi_np  = np.array(fi, dtype=np.int32)
    fvs    = np.concatenate([[0], np.cumsum(fc_np)]).astype(np.int32)
    return fc_np, fi_np, fvs


def mesh_uvs(mesh_prim):
    """Return (uv_vals, uv_indices_np_or_None, interp) or (None, None, None)."""
    for name in ('st', 'st0', 'UVMap'):
        pv = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar(name)
        if pv and pv.IsDefined():
            vals = pv.Get()
            if vals is None:
                continue
            raw_idx = pv.GetIndices()
            idx_np  = np.array(raw_idx, dtype=np.int32) if (raw_idx and len(raw_idx) > 0) else None
            return vals, idx_np, pv.GetInterpolation()
    return None, None, None


# ---------------------------------------------------------------------------
# Material / texture helpers
# ---------------------------------------------------------------------------

def _descendants(prim):
    for child in prim.GetChildren():
        yield child
        yield from _descendants(child)


def diffuse_texture(prim):
    """Return (mat_name, tex_path, rgba) for a Mesh or GeomSubset prim.

    tex_path is the resolved path of the diffuse texture file (or None).
    rgba is (r, g, b, a) where alpha comes from opacity inputs when available.
    If a texture exists, rgba is still returned as a tint (usually (1,1,1,a)).
    """
    binding = UsdShade.MaterialBindingAPI(prim)
    mat, _  = binding.ComputeBoundMaterial()
    if not mat or not mat.GetPrim().IsValid():
        return None, None, None
    mat_name  = mat.GetPrim().GetName()
    color_tex = None
    any_tex   = None
    rgba      = None

    # Keywords for diffuse color inputs across OmniPBR and UsdPreviewSurface
    _COLOR_INPUT_KEYWORDS = ('diffuse_color_constant', 'diffusecolor', 'diffuse_color',
                             'basecolor', 'base_color')
    # Keywords for opacity/alpha inputs (OmniPBR: opacity_constant; UsdPreviewSurface: opacity)
    _OPACITY_KEYWORDS = ('opacity_constant', 'opacity', 'alpha_constant', 'alpha')

    opacity_val    = None   # float in [0, 1]; None = not found
    opacity_enabled = True  # OmniPBR's enable_opacity gate; default True for other shaders
    is_glass       = False  # OmniGlass.mdl or similar — no readable inputs, use fallback alpha

    for child in _descendants(mat.GetPrim()):
        shader = UsdShade.Shader(child)
        if not shader.GetPrim().IsValid():
            continue

        # Detect glass / transparent shaders by their MDL source asset or shader id
        for attr_name in ('info:mdl:sourceAsset', 'info:id'):
            attr = shader.GetPrim().GetAttribute(attr_name)
            if attr and attr.IsValid():
                try:
                    v = attr.Get()
                    s = str(v.path if isinstance(v, Sdf.AssetPath) else v).lower()
                    if 'glass' in s or 'transparent' in s:
                        is_glass = True
                except Exception:
                    pass

        for inp in shader.GetInputs():
            try:
                val = inp.Get()
            except Exception:
                continue
            inp_base = inp.GetBaseName().lower()
            # Texture asset
            if isinstance(val, Sdf.AssetPath):
                resolved = val.resolvedPath or val.path
                if not resolved:
                    continue
                ext = os.path.splitext(resolved)[1].lower()
                if ext not in TEX_EXTS:
                    continue
                if any_tex is None:
                    any_tex = resolved
                if ('color' in resolved.lower() or 'diffuse' in resolved.lower()
                        or 'albedo' in resolved.lower()):
                    color_tex = resolved
            # Diffuse color constant (Vec3f / Vec4f)
            elif rgba is None and any(kw in inp_base for kw in _COLOR_INPUT_KEYWORDS):
                try:
                    r, g, b = float(val[0]), float(val[1]), float(val[2])
                    rgba = (r, g, b, 1.0)   # alpha patched below
                except Exception:
                    pass
            # OmniPBR enable_opacity gate
            elif inp_base == 'enable_opacity':
                try:
                    opacity_enabled = bool(val)
                except Exception:
                    pass
            # Opacity / alpha constant
            elif opacity_val is None and any(kw == inp_base for kw in _OPACITY_KEYWORDS):
                try:
                    opacity_val = float(val)
                except Exception:
                    pass
        if color_tex:
            break

    # Resolve final alpha value
    if not opacity_enabled:
        alpha = 1.0                          # opacity explicitly disabled → fully opaque
    elif opacity_val is not None:
        alpha = max(0.0, min(1.0, opacity_val))
    elif is_glass:
        alpha = 0.2                          # OmniGlass with no readable inputs
    else:
        alpha = 1.0

    # Patch alpha into rgba; if no color was found but transparency applies, use white tint
    if rgba is not None:
        rgba = (rgba[0], rgba[1], rgba[2], alpha)
    elif alpha < 1.0:
        rgba = (1.0, 1.0, 1.0, alpha)

    return mat_name, (color_tex or any_tex), rgba


def register_material(mat_name, tex_abs, rgba, tex_dir, materials, mat_key_cache):
    """Deduplicate and register a material. Returns mat_key or None."""
    if not mat_name:
        return None
    cache_key = (mat_name, tex_abs or '')
    if cache_key in mat_key_cache:
        return mat_key_cache[cache_key]
    tex_file = None
    if tex_abs and os.path.isfile(tex_abs):
        dst_name = mat_name + '.png'   # MuJoCo 3.x requires PNG
        dst_path = os.path.join(tex_dir, dst_name)
        if not os.path.exists(dst_path):
            img = Image.open(tex_abs).convert('RGB')
            img.save(dst_path)
        tex_file = dst_name
    mat_key = mat_name
    if mat_key in materials:
        mat_key = f"{mat_name}_{len(materials)}"
    mat_key_cache[cache_key] = mat_key
    materials[mat_key] = {'tex_file': tex_file, 'rgba': rgba}
    return mat_key


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------

def _triangulate(v_ids, uv_ids):
    """Fan-triangulate a polygon. Returns list of (tri_v, tri_uv|None)."""
    tris = []
    for j in range(1, len(v_ids) - 1):
        tv = (v_ids[0], v_ids[j], v_ids[j + 1])
        tu = (uv_ids[0], uv_ids[j], uv_ids[j + 1]) if uv_ids else None
        tris.append((tv, tu))
    return tris


def _simplify(verts, faces):
    """Simplify mesh geometry using Quadric Edge Collapse (pymeshlab).

    verts: (N, 3) float32
    faces: (F, 3) int32
    Returns (new_verts, new_faces). UV coords are intentionally dropped because
    this pymeshlab version has no preservetex support; material rgba is kept.
    """
    if SIMPLIFY_RATIO <= 0.0 or len(faces) < 8 or len(faces) <= SIMPLIFY_MIN_FACES:
        return verts, faces
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts.astype(np.float64),
                               face_matrix=faces.astype(np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=float(SIMPLIFY_RATIO),
        preserveboundary=True,
        preservenormal=True,
        optimalplacement=True,
        autoclean=True,
    )
    m = ms.current_mesh()
    new_verts = m.vertex_matrix().astype(np.float32)
    new_faces = m.face_matrix().astype(np.int32)
    # Fall back to original if result is degenerate (MuJoCo needs ≥ 4 vertices)
    if len(new_verts) < 4 or len(new_faces) == 0:
        return verts, faces
    return new_verts, new_faces


def _write_obj(out_path, verts, uvs, tri_list):
    """Write OBJ from a tri_list [(tri_v, tri_uv|None), …]. uvs may be None."""
    with open(out_path, 'w') as f:
        f.write("# USD → MuJoCo (world-space)\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if uvs:
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for tri_v, tri_uv in tri_list:
            if tri_uv:
                parts = [f"{tv+1}/{tu+1}" for tv, tu in zip(tri_v, tri_uv)]
            else:
                parts = [str(tv + 1) for tv in tri_v]
            f.write("f " + " ".join(parts) + "\n")


def _write_obj_arrays(out_path, verts, faces):
    """Write OBJ from numpy arrays (no UVs — used after simplification)."""
    with open(out_path, 'w') as f:
        f.write("# USD → MuJoCo (world-space, simplified)\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(vi + 1) for vi in face) + "\n")


def export_whole_mesh(mesh_prim, xform_cache, out_path, actor_M_inv=None):
    """Export entire Mesh as OBJ with UVs (or simplified without UVs).
    Returns (True, (bbox_min, bbox_max)) on success, (False, None) on failure.
    bbox is in body-local (actor) space."""
    verts = world_verts(mesh_prim, xform_cache)
    if verts is None:
        return False, None
    if actor_M_inv is not None:
        h = np.hstack([verts.astype(np.float64), np.ones((len(verts), 1))])
        verts = (h @ actor_M_inv)[:, :3].astype(np.float32)
    fc, fi, fvs = mesh_topology(mesh_prim)
    if fc is None:
        return False, None
    uv_vals, uv_idx, uv_interp = mesh_uvs(mesh_prim)
    has_uv = uv_vals is not None and uv_interp == UsdGeom.Tokens.faceVarying

    # Build triangulated face list (and UV list)
    all_uvs  = [(float(uv[0]), float(uv[1])) for uv in uv_vals] if has_uv else None
    tri_list = []
    tri_faces = []  # parallel list of (v0, v1, v2) as ints for simplification
    for f in range(len(fc)):
        count    = int(fc[f])
        fv_start = int(fvs[f])
        v_ids    = [int(fi[fv_start + k]) for k in range(count)]
        if has_uv:
            uv_ids = [int(uv_idx[fv_start + k]) if uv_idx is not None else fv_start + k
                      for k in range(count)]
        else:
            uv_ids = None
        tris = _triangulate(v_ids, uv_ids)
        tri_list.extend(tris)
        tri_faces.extend(tv for tv, _ in tris)

    faces_np = np.array(tri_faces, dtype=np.int32)
    if SIMPLIFY_RATIO > 0.0 and len(faces_np) > SIMPLIFY_MIN_FACES:
        verts, faces_np = _simplify(verts, faces_np)
        _write_obj_arrays(out_path, verts, faces_np)
    else:
        _write_obj(out_path, verts, all_uvs, tri_list)
    bbox = (verts.min(axis=0), verts.max(axis=0))
    return True, bbox


def export_geomsubset(mesh_prim, subset_prim, xform_cache, out_path, actor_M_inv=None):
    """Export one GeomSubset of a parent Mesh as OBJ with UVs.
    Returns (True, (bbox_min, bbox_max)) on success, (False, None) on failure."""
    all_verts = world_verts(mesh_prim, xform_cache)
    if all_verts is None:
        return False, None
    if actor_M_inv is not None:
        h = np.hstack([all_verts.astype(np.float64), np.ones((len(all_verts), 1))])
        all_verts = (h @ actor_M_inv)[:, :3].astype(np.float32)
    fc, fi, fvs = mesh_topology(mesh_prim)
    if fc is None:
        return False
    uv_vals, uv_idx, uv_interp = mesh_uvs(mesh_prim)
    has_uv = uv_vals is not None and uv_interp == UsdGeom.Tokens.faceVarying

    subset_faces = np.array(UsdGeom.Subset(subset_prim).GetIndicesAttr().Get(), dtype=np.int32)
    if len(subset_faces) == 0:
        return False, None

    vert_map = {}   # global vert idx → local
    uv_map   = {}   # global UV idx   → local
    new_verts = []
    new_uvs   = []
    tri_list  = []

    for f_idx in subset_faces:
        count    = int(fc[f_idx])
        fv_start = int(fvs[f_idx])
        v_local  = []
        uv_local = []
        for k in range(count):
            # vertex
            vg = int(fi[fv_start + k])
            if vg not in vert_map:
                vert_map[vg] = len(new_verts)
                new_verts.append(all_verts[vg])
            v_local.append(vert_map[vg])
            # UV
            if has_uv:
                ug = int(uv_idx[fv_start + k]) if uv_idx is not None else fv_start + k
                if ug not in uv_map:
                    uv = uv_vals[ug]
                    uv_map[ug] = len(new_uvs)
                    new_uvs.append((float(uv[0]), float(uv[1])))
                uv_local.append(uv_map[ug])
        tris = _triangulate(v_local, uv_local if has_uv else None)
        tri_list.extend(tris)

    if not new_verts:
        return False, None

    verts_np = np.array(new_verts, dtype=np.float32)
    faces_np = np.array([tv for tv, _ in tri_list], dtype=np.int32)
    if SIMPLIFY_RATIO > 0.0 and len(faces_np) > SIMPLIFY_MIN_FACES:
        verts_np, faces_np = _simplify(verts_np, faces_np)
        _write_obj_arrays(out_path, verts_np, faces_np)
    else:
        _write_obj(out_path, verts_np, new_uvs if has_uv else None, tri_list)
    bbox = (verts_np.min(axis=0), verts_np.max(axis=0))
    return True, bbox


# ---------------------------------------------------------------------------
# Collision classification helper
# ---------------------------------------------------------------------------

def _is_shell_bbox(bbox):
    """Return True if the bbox is thin enough to be treated as a shell mesh."""
    if bbox is None:
        return False
    bmin, bmax = bbox
    dims = np.abs(bmax - bmin)
    max_dim = float(np.max(dims))
    if max_dim < 1e-6:
        return False
    return float(np.min(dims)) / max_dim < SHELL_ASPECT_RATIO


# ---------------------------------------------------------------------------
# Articulation helpers
# ---------------------------------------------------------------------------

_MOVABLE_JOINT_TYPES = {'PhysicsRevoluteJoint', 'PhysicsPrismaticJoint'}
_FIXED_JOINT_TYPES   = {'PhysicsFixedJoint'}
_ALL_JOINT_TYPES     = _MOVABLE_JOINT_TYPES | _FIXED_JOINT_TYPES


def collect_joints(actor_prim):
    """Collect physics joints from an actor prim.

    Scans direct children and any Scope child named 'joints'.
    Returns (movable, fixed) where each entry is (joint_prim, b0_name, b1_name).
    b0/b1 names are the last path component of the body targets.
    """
    movable, fixed = [], []

    def _proc(j):
        b0 = j.GetRelationship('physics:body0').GetTargets()
        b1 = j.GetRelationship('physics:body1').GetTargets()
        if not b0 or not b1:
            return
        b0n = str(b0[0]).split('/')[-1]
        b1n = str(b1[0]).split('/')[-1]
        t = j.GetTypeName()
        if t in _MOVABLE_JOINT_TYPES:
            movable.append((j, b0n, b1n))
        elif t in _FIXED_JOINT_TYPES:
            fixed.append((j, b0n, b1n))

    for child in actor_prim.GetChildren():
        t = child.GetTypeName()
        if t in _ALL_JOINT_TYPES:
            _proc(child)
        elif t == 'Scope':
            for jc in child.GetChildren():
                if jc.GetTypeName() in _ALL_JOINT_TYPES:
                    _proc(jc)
    return movable, fixed


def joint_mjcf_info(joint_prim, child_body_prim, child_M_inv, xc):
    """Compute joint parameters for a MuJoCo <joint> element, in child-body-local frame.

    USD uses row-vector convention: world = local @ M.
    Joint pos and axis are transformed from world space to the child body's local frame
    using child_M_inv (the inverse of the child body's pose-only 4×4 matrix).

    Returns dict with: type ('hinge'|'slide'), pos (3-vec), axis (3-vec), lo, hi.
    """
    jtype = joint_prim.GetTypeName()

    # Map USD axis string to world-space unit vector
    axis_attr = joint_prim.GetAttribute('physics:axis').Get()
    axis_map  = {'X': np.array([1., 0., 0.]),
                 'Y': np.array([0., 1., 0.]),
                 'Z': np.array([0., 0., 1.])}
    axis_w = axis_map.get(str(axis_attr) if axis_attr else 'Z', np.array([0., 0., 1.]))

    # localRot1 rotates the joint frame within the child body; apply to axis
    r1 = joint_prim.GetAttribute('physics:localRot1').Get()
    if r1 is not None:
        try:
            w = float(r1.GetReal())
            im = r1.GetImaginary()
            x, y, z = float(im[0]), float(im[1]), float(im[2])
            if abs(w - 1.0) > 1e-4:
                R = np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
                               [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
                               [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]])
                axis_w = R @ axis_w
        except Exception:
            pass

    # Transform direction to child-body-local frame
    axis_local = axis_w @ child_M_inv[:3, :3]
    n = np.linalg.norm(axis_local)
    if n > 1e-10:
        axis_local /= n

    # localPos1: joint anchor position in child body's local frame
    p1 = joint_prim.GetAttribute('physics:localPos1').Get()
    lp1 = np.array([float(p1[0]), float(p1[1]), float(p1[2])], dtype=np.float64) if p1 else np.zeros(3)

    # Child body world transform (row-vector: world_pt = local_pt @ M + t)
    xf   = xc.GetLocalToWorldTransform(child_body_prim)
    cM   = np.array([[xf[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
    child_world_pos = cM[3, :3]
    child_R = cM[:3, :3]
    child_R_n = child_R / np.maximum(np.linalg.norm(child_R, axis=1, keepdims=True), 1e-10)

    # Joint world pos = child_world_origin + localPos1 expressed in world
    joint_world = child_world_pos + lp1 @ child_R_n

    # Transform to child-body-local frame (point transform with translation)
    joint_local = (np.array([*joint_world, 1.0]) @ child_M_inv)[:3]

    lo = joint_prim.GetAttribute('physics:lowerLimit').Get()
    hi = joint_prim.GetAttribute('physics:upperLimit').Get()

    return {
        'type': 'hinge' if jtype == 'PhysicsRevoluteJoint' else 'slide',
        'pos':  joint_local,
        'axis': axis_local,
        'lo':   float(lo) if lo is not None else None,
        'hi':   float(hi) if hi is not None else None,
    }


def parse_articulation(actor_prim, xc, stage,
                        actor_pos_w=None, actor_quat_w=None, actor_M_inv=None):
    """Parse the articulation graph of an actor prim.

    When actor_pos_w / actor_quat_w / actor_M_inv are provided, relative body
    poses (pos_rel, quat_rel) are computed here — where the body prims are
    already in hand — rather than in a fragile secondary pass.

    Returns None if no joints exist, otherwise:
    {
        'root':    root_sub_body_name,
        'movable': {child_name: {
                        'parent':   parent_name,
                        'joint':    joint_info,
                        'M_inv':    child_M_inv,   # world → child-local (for mesh export)
                        'pos_rel':  np.array | None,
                        'quat_rel': np.array | None,
                   }},
        'fixed':   {child_name: {
                        'parent':   parent_name,
                        'M_inv':    child_M_inv,
                        'pos_rel':  np.array | None,
                        'quat_rel': np.array | None,
                   }},
    }
    """
    movable_joints, fixed_joints = collect_joints(actor_prim)
    if not movable_joints and not fixed_joints:
        return None

    actor_path      = str(actor_prim.GetPath())
    movable_info    = {}
    fixed_info_raw  = {}   # b1n → {'parent': b0n, 'M_inv': ...}
    body_world_data = {}   # sb_name → (virtual_world_pos, world_quat, M_inv)
    all_children    = set()
    all_parents     = set()

    def _joint_anchor_and_Minv(jpr, b0n, b1n_prim):
        """Compute virtual world pos from joint localPos0 + parent USD world pos.

        USD assets in this format store all body prim transforms at -actor_offset
        so every GetLocalToWorldTransform ≈ world-origin.  The joint anchor in
        world space is: parent_USD_world_pos + localPos0 @ parent_R.
        We use that as the child body's frame origin for mesh export and pos_rel.
        """
        parent_prim = stage.GetPrimAtPath(f"{actor_path}/{b0n}")
        if parent_prim and parent_prim.IsValid():
            p_xf = xc.GetLocalToWorldTransform(parent_prim)
            p_M  = np.array([[p_xf[r][c] for c in range(4)] for r in range(4)])
            p_pos_usd = p_M[3, :3]
            norms = np.maximum(np.linalg.norm(p_M[:3, :3], axis=1, keepdims=True), 1e-10)
            p_R   = p_M[:3, :3] / norms
        else:
            p_pos_usd = np.zeros(3)
            p_R       = np.eye(3)

        lp0_attr = jpr.GetAttribute('physics:localPos0').Get()
        lp0 = np.array([float(lp0_attr[i]) for i in range(3)]) if lp0_attr else np.zeros(3)
        c_pos_w = p_pos_usd + lp0 @ p_R  # joint anchor in world

        # Child rotation from its own USD transform (correct even when translation is wrong)
        c_xf = xc.GetLocalToWorldTransform(b1n_prim)
        c_M  = np.array([[c_xf[r][c] for c in range(4)] for r in range(4)])
        norms_c = np.maximum(np.linalg.norm(c_M[:3, :3], axis=1, keepdims=True), 1e-10)
        c_R = c_M[:3, :3] / norms_c

        # Build virtual pose matrix (origin = joint anchor, rotation = child rotation)
        M_virt = np.eye(4)
        M_virt[:3, :3] = c_R
        M_virt[3, :3]  = c_pos_w
        child_M_inv = np.linalg.inv(M_virt)

        return c_pos_w, child_M_inv

    for jpr, b0n, b1n in movable_joints:
        child_prim = stage.GetPrimAtPath(f"{actor_path}/{b1n}")
        if not child_prim or not child_prim.IsValid():
            continue
        c_pos_w, child_M_inv = _joint_anchor_and_Minv(jpr, b0n, child_prim)
        _, c_quat_w, _       = actor_pose(child_prim, xc)  # rotation is correct; pos overridden
        body_world_data[b1n] = (c_pos_w, c_quat_w, child_M_inv)
        info = joint_mjcf_info(jpr, child_prim, child_M_inv, xc)
        movable_info[b1n] = {'parent': b0n, 'joint': info, 'M_inv': child_M_inv}
        all_children.add(b1n)
        all_parents.add(b0n)

    for jpr, b0n, b1n in fixed_joints:
        child_prim = stage.GetPrimAtPath(f"{actor_path}/{b1n}")
        if child_prim and child_prim.IsValid():
            try:
                c_pos_w, fc_M_inv = _joint_anchor_and_Minv(jpr, b0n, child_prim)
                _, c_quat_w, _    = actor_pose(child_prim, xc)
                body_world_data[b1n] = (c_pos_w, c_quat_w, fc_M_inv)
                fixed_info_raw[b1n]  = {'parent': b0n, 'M_inv': fc_M_inv}
            except Exception:
                fixed_info_raw[b1n] = {'parent': b0n, 'M_inv': None}
        else:
            fixed_info_raw[b1n] = {'parent': b0n, 'M_inv': None}
        all_children.add(b1n)
        all_parents.add(b0n)

    # Root = appears as parent but never as a child in any joint (movable or fixed)
    root_candidates = all_parents - all_children
    root = next(iter(root_candidates)) if root_candidates else (
        next(iter(all_parents)) if all_parents else None)

    # --- compute relative poses (world → MJCF-parent-local) ---
    # Root sub-body maps to the actor body element (same world pos as actor),
    # so children of root use actor_M_inv / actor_quat_w.
    # Deeper children (e.g. Handle child-of-Door) use the parent body's own M_inv.
    def _parent_Minv_and_quat(parent_name):
        if parent_name == root:
            return actor_M_inv, actor_quat_w
        pd = body_world_data.get(parent_name)
        if pd is not None:
            return pd[2], pd[1]
        return actor_M_inv, actor_quat_w

    def _compute_rel(child_name, parent_name):
        cd = body_world_data.get(child_name)
        if cd is None or actor_M_inv is None:
            return None, None
        c_pos_w, c_quat_w, _ = cd
        p_M_inv, p_quat_w = _parent_Minv_and_quat(parent_name)
        if p_M_inv is None or p_quat_w is None:
            return None, None
        pos_rel  = (np.append(c_pos_w, 1.0) @ p_M_inv)[:3]
        p_qc     = np.array([p_quat_w[0], -p_quat_w[1], -p_quat_w[2], -p_quat_w[3]])
        quat_rel = _quat_mul(p_qc, c_quat_w)
        return pos_rel, quat_rel

    for child_name, info in movable_info.items():
        pos_rel, quat_rel = _compute_rel(child_name, info['parent'])
        info['pos_rel']  = pos_rel
        info['quat_rel'] = quat_rel

    fixed_info = {}
    for b1n, raw in fixed_info_raw.items():
        pos_rel, quat_rel = _compute_rel(b1n, raw['parent'])
        fixed_info[b1n] = {
            'parent':   raw['parent'],
            'M_inv':    raw['M_inv'],
            'pos_rel':  pos_rel,
            'quat_rel': quat_rel,
        }

    return {'root': root, 'movable': movable_info, 'fixed': fixed_info}


# ---------------------------------------------------------------------------
# MJCF builder helpers
# ---------------------------------------------------------------------------

def _build_sub_bodies(actor_body_elem, actor_name, artic, sub_body_elems):
    """Create nested <body> elements for an articulated actor.

    Root sub-body maps to the actor body element itself (no extra nesting).
    Each movable child gets a new <body> with a <joint>.
    Each fixed-joint child gets a new <body> with no <joint> (rigidly fixed in MuJoCo).
    pos_rel / quat_rel stored in each info dict by parse_articulation are applied directly.
    Results are stored in sub_body_elems: (actor_name, sub_body_name) → Element.
    """
    root = artic['root']
    if root:
        sub_body_elems[(actor_name, root)] = actor_body_elem

    def _apply_pose(elem, pos_rel, quat_rel):
        if pos_rel is not None:
            elem.set('pos', f'{pos_rel[0]:.6f} {pos_rel[1]:.6f} {pos_rel[2]:.6f}')
        if quat_rel is not None:
            if not (abs(quat_rel[0]-1.0) < 1e-5 and abs(quat_rel[1]) < 1e-5
                    and abs(quat_rel[2]) < 1e-5 and abs(quat_rel[3]) < 1e-5):
                elem.set('quat', f'{quat_rel[0]:.6f} {quat_rel[1]:.6f} {quat_rel[2]:.6f} {quat_rel[3]:.6f}')

    for child_name, info in artic['movable'].items():
        parent_elem = sub_body_elems.get((actor_name, info['parent']), actor_body_elem)
        snake_child = _to_snake(child_name)

        sub_b = SubElement(parent_elem, 'body')
        sub_b.set('name', f"{actor_name}_{snake_child}")
        _apply_pose(sub_b, info.get('pos_rel'), info.get('quat_rel'))

        ji  = info['joint']
        jel = SubElement(sub_b, 'joint')
        jel.set('name', f"{actor_name}_{snake_child}")
        jel.set('type', ji['type'])
        p = ji['pos']
        jel.set('pos',  f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
        a = ji['axis']
        jel.set('axis', f"{a[0]:.5f} {a[1]:.5f} {a[2]:.5f}")
        if ji['lo'] is not None and ji['hi'] is not None:
            jel.set('range', f"{ji['lo']:.3f} {ji['hi']:.3f}")

        sub_body_elems[(actor_name, child_name)] = sub_b

    # Fixed-joint children get their own <body> (no <joint> = rigidly fixed in MuJoCo)
    for fixed_child, finfo in artic['fixed'].items():
        parent_elem = sub_body_elems.get((actor_name, finfo['parent']), actor_body_elem)
        sub_b = SubElement(parent_elem, 'body')
        sub_b.set('name', f"{actor_name}_{_to_snake(fixed_child)}")
        _apply_pose(sub_b, finfo.get('pos_rel'), finfo.get('quat_rel'))
        sub_body_elems[(actor_name, fixed_child)] = sub_b


def _target_elem(e, actor_bodies, sub_body_elems):
    """Return the body Element that should own this geom entry."""
    actor    = e.get('actor', e['name'])
    sub_body = e.get('sub_body')
    if sub_body is not None:
        elem = sub_body_elems.get((actor, sub_body))
        if elem is not None:
            return elem
    return actor_bodies.get(actor)


# ---------------------------------------------------------------------------
# MJCF builder
# ---------------------------------------------------------------------------

def build_mjcf(geom_entries, materials, scene_id, actor_poses=None,
               articulations=None, free_actors=None):
    root = Element('mujoco')
    root.set('model', f'scene_{scene_id}')

    compiler = SubElement(root, 'compiler')
    compiler.set('angle', 'degree')
    compiler.set('meshdir', MESH_SUBDIR)
    compiler.set('texturedir', TEX_SUBDIR)

    SubElement(root, 'option').set('gravity', '0 0 -9.81')

    # Default classes: visual geoms (no collision, group 2) and
    # collision geoms (semi-transparent yellow, group 3).
    default = SubElement(root, 'default')
    dj = SubElement(default, 'joint')
    dj.set('armature', '0.01')
    dj.set('damping', '0.5')
    dvis = SubElement(default, 'default')
    dvis.set('class', 'visual')
    dvis_g = SubElement(dvis, 'geom')
    dvis_g.set('contype', '0')
    dvis_g.set('conaffinity', '0')
    dvis_g.set('group', '2')
    dcol = SubElement(default, 'default')
    dcol.set('class', 'collision')
    dcol_g = SubElement(dcol, 'geom')
    dcol_g.set('rgba', '0.9 0.9 0.0 0.2')
    dcol_g.set('group', '3')

    asset     = SubElement(root, 'asset')
    worldbody = SubElement(root, 'worldbody')

    for mat_key, info in sorted(materials.items()):
        if info.get('tex_file'):
            tx = SubElement(asset, 'texture')
            tx.set('name', f'T_{mat_key}')
            tx.set('type', '2d')
            tx.set('file', info['tex_file'])
        m = SubElement(asset, 'material')
        m.set('name', f'M_{mat_key}')
        if info.get('tex_file'):
            m.set('texture', f'T_{mat_key}')
        if info.get('rgba'):
            r, g, b, a = info['rgba']
            m.set('rgba', f'{r:.4f} {g:.4f} {b:.4f} {a:.4f}')


    # Register all meshes in asset
    for e in geom_entries:
        me = SubElement(asset, 'mesh')
        me.set('name', e['name'])
        me.set('file', e['obj_file'])
        if _is_shell_bbox(e.get('bbox')):
            me.set('inertia', 'shell')

    # Group geoms by actor → one <body> per actor
    from collections import OrderedDict
    actor_bodies   = OrderedDict()  # actor_name → body Element
    sub_body_elems = {}             # (actor_name, sub_body_name) → body Element

    for e in geom_entries:
        actor = e.get('actor', e['name'])
        if actor not in actor_bodies:
            body = SubElement(worldbody, 'body')
            body.set('name', actor)
            if actor_poses and actor in actor_poses:
                pos, quat = actor_poses[actor]
                body.set('pos', f'{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}')
                if not (abs(quat[0] - 1.0) < 1e-5 and
                        abs(quat[1]) < 1e-5 and abs(quat[2]) < 1e-5 and abs(quat[3]) < 1e-5):
                    body.set('quat', f'{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}')
            actor_bodies[actor] = body

            # Build nested sub-body structure for articulated actors
            if articulations and actor in articulations:
                _build_sub_bodies(body, actor, articulations[actor], sub_body_elems)
            elif free_actors and actor in free_actors:
                SubElement(body, 'freejoint')

        tgt = _target_elem(e, actor_bodies, sub_body_elems)
        g = SubElement(tgt, 'geom')
        g.set('type', 'mesh')
        g.set('mesh', e['name'])
        g.set('class', 'visual')
        if e.get('mat_key') and e['mat_key'] in materials:
            g.set('material', f"M_{e['mat_key']}")

    # Add AABB box collision geoms for every geom, routed to the correct sub-body element
    actor_geom_map = {}
    for e in geom_entries:
        actor_geom_map.setdefault(e.get('actor', e['name']), []).append(e)

    for actor in actor_bodies:
        for e in actor_geom_map.get(actor, []):
            bbox = e.get('bbox')
            if bbox is None:
                continue
            tgt    = _target_elem(e, actor_bodies, sub_body_elems)
            bmin, bmax = bbox
            half   = np.maximum((bmax - bmin) / 2.0, SHELL_MIN_HALF)
            center = (bmin + bmax) / 2.0
            cg = SubElement(tgt, 'geom')
            cg.set('name',  f'{_to_snake(e["name"])}_aabb')
            cg.set('type',  'box')
            cg.set('pos',   f'{center[0]:.4f} {center[1]:.4f} {center[2]:.4f}')
            cg.set('size',  f'{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}')
            cg.set('class', 'collision')

    return root


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(usd_path, out_dir):
    t0 = time.time()
    # scene_id = USD file stem (e.g. 'table_edit.usd' → 'table_edit'); used for the
    # MJCF model name and the output .xml filename.
    scene_id  = os.path.splitext(os.path.basename(usd_path))[0]
    mesh_dir  = os.path.join(out_dir, MESH_SUBDIR)
    tex_dir   = os.path.join(out_dir, TEX_SUBDIR)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(tex_dir,  exist_ok=True)
    print(f"Output dir : {out_dir}")
    print(f"Loading    : {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    xc    = UsdGeom.XformCache()
    print(f"  Stage loaded in {time.time()-t0:.1f}s")

    # --- pass 1: collect actor → [(Mesh prim, sub_body_name)] mapping ---
    # sub_body_name = name of the direct Xform child of the Actor that owns this mesh.
    # Primary mode: look for Actor_* named Xform prims (kitchen / HM3D style).
    actor_to_meshes = {}   # actor_path → [(mesh_prim, sub_body_name)]
    actor_prim_map  = {}   # actor_path → actor_prim
    for prim in stage.Traverse():
        path  = str(prim.GetPath())
        parts = path.split('/')
        if parts[-1].startswith('Actor_') and prim.GetTypeName() == 'Xform':
            actor_prim_map.setdefault(path, prim)
        if prim.GetTypeName() == 'Mesh':
            for i, pt in enumerate(parts):
                if pt.startswith('Actor_'):
                    actor_path = '/'.join(parts[:i+1])
                    # sub_body = direct child of actor that contains this mesh
                    sub_body = parts[i+1] if i + 2 < len(parts) else None
                    actor_to_meshes.setdefault(actor_path, []).append((prim, sub_body))
                    break

    # Fallback (generic mode): no Actor_ prims found → treat each Mesh's direct
    # parent Xform as its actor.  Works for flat scenes like Isaac Sim simple_room
    # where the hierarchy is /Root/ObjectName/MeshName.
    generic_mode = not actor_prim_map
    if generic_mode:
        print("  No Actor_ prims found — using parent Xform as actor (generic mode)")
        for prim in stage.Traverse():
            if prim.GetTypeName() != 'Mesh':
                continue
            parent = prim.GetParent()
            if not parent or parent.GetTypeName() != 'Xform':
                continue
            ap = str(parent.GetPath())
            actor_prim_map.setdefault(ap, parent)
            actor_to_meshes.setdefault(ap, []).append((prim, None))

    # --- pass 2: assign stable actor names ---
    # Two parallel name spaces are maintained:
    #   actor_names[path]       → body name  (snake_case, e.g. "dining_table_0")
    #   actor_mesh_bases[path]  → mesh base  (CamelCase,  e.g. "DiningTable0")
    # Mesh naming rules stay unchanged (CamelCase); body naming uses snake_case.
    seen_actors      = set()
    mesh_name_seen   = {}
    body_name_seen   = {}
    actor_names      = {}   # path → snake_case body name
    actor_mesh_bases = {}   # path → CamelCase mesh base name
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path not in actor_prim_map or path in seen_actors:
            continue
        seen_actors.add(path)
        if generic_mode:
            mesh_base = _to_camel(prim.GetName())
            body_base = _to_snake(prim.GetName())
        else:
            parts     = path.split('/')
            category  = parts[-2]
            actor_num = int(parts[-1].replace('Actor_', ''))
            mesh_base = _to_camel(f"{category}{actor_num}")
            body_base = f"{_to_snake(category)}_{actor_num}"
        # Mesh name deduplication (CamelCase suffix: base, base1, base2, …)
        if mesh_base in mesh_name_seen:
            mesh_name_seen[mesh_base] += 1
            mesh_name = f"{mesh_base}{mesh_name_seen[mesh_base]}"
        else:
            mesh_name_seen[mesh_base] = 0
            mesh_name = mesh_base
        # Body name deduplication (snake_case suffix: base, base_1, base_2, …)
        if body_base in body_name_seen:
            body_name_seen[body_base] += 1
            body_name = f"{body_base}_{body_name_seen[body_base]}"
        else:
            body_name_seen[body_base] = 0
            body_name = body_base
        actor_names[path]      = body_name
        actor_mesh_bases[path] = mesh_name

    # --- compute actor world poses (pos, quat, M_inv) ---
    actor_xforms = {}   # actor_name → (pos, quat, M_inv)
    for ap, prim in actor_prim_map.items():
        name = actor_names.get(ap)
        if name:
            pos, quat, M_inv = actor_pose(prim, xc)
            actor_xforms[name] = (pos, quat, M_inv)

    # --- parse articulations for actors that have physics joints ---
    # Pass the actor's world transform so parse_articulation can compute relative
    # body poses internally (where body prims are already resolved).
    actor_articulations = {}   # actor_name → articulation_info
    for ap, prim in actor_prim_map.items():
        name = actor_names.get(ap)
        if not name:
            continue
        a_pos_w, a_quat_w, a_M_inv = actor_xforms.get(name, (None, None, None))
        artic = parse_articulation(prim, xc, stage,
                                   actor_pos_w=a_pos_w,
                                   actor_quat_w=a_quat_w,
                                   actor_M_inv=a_M_inv)
        if artic:
            actor_articulations[name] = artic
            print(f"  [ARTIC] {name}: root='{artic['root']}', "
                  f"{len(artic['movable'])} movable, {len(artic['fixed'])} fixed joints")

    # --- detect freely movable actors (rigidBodyEnabled=True, kinematicEnabled!=True) ---
    # Scan all prims once; skip actors that are already articulated (they use joints).
    free_actors = set()
    for prim in stage.Traverse():
        rb  = prim.GetAttribute('physics:rigidBodyEnabled')
        kin = prim.GetAttribute('physics:kinematicEnabled')
        rb_val  = rb.Get()  if rb  and rb.IsValid()  else None
        kin_val = kin.Get() if kin and kin.IsValid() else None
        if rb_val is not True or kin_val is True:
            continue
        path  = str(prim.GetPath())
        parts = path.split('/')
        for i, pt in enumerate(parts):
            if pt.startswith('Actor_'):
                actor_path = '/'.join(parts[:i+1])
                name = actor_names.get(actor_path)
                if name and name not in actor_articulations:
                    free_actors.add(name)
                break
    if free_actors:
        print(f"  [FREE]  {len(free_actors)} freely movable actors: "
              f"{', '.join(sorted(free_actors)[:8])}"
              f"{'…' if len(free_actors) > 8 else ''}")

    # --- pass 3: pre-count total geoms per actor (for naming) ---
    actor_geom_count = {}
    for ap, mp_list in actor_to_meshes.items():
        total = 0
        for mp, _ in mp_list:
            subs = [c for c in mp.GetPrim().GetChildren() if c.GetTypeName() == 'GeomSubset']
            total += len(subs) if subs else 1
        actor_geom_count[ap] = total

    # --- pass 4: export ---
    geom_entries  = []
    materials     = {}
    mat_key_cache = {}

    for ap, mp_list in actor_to_meshes.items():
        actor_body_name = actor_names.get(ap)       # snake_case — for <body> elements
        actor_mesh_base = actor_mesh_bases.get(ap)  # CamelCase  — for mesh file/asset names
        if not actor_body_name or not actor_mesh_base:
            continue
        multi = actor_geom_count[ap] > 1
        counter = [0]

        def geom_name():
            n = f"{actor_mesh_base}{counter[0]}" if multi else actor_mesh_base
            counter[0] += 1
            return n

        actor_M_inv    = actor_xforms.get(actor_body_name, (None, None, None))[2]
        is_articulated = actor_body_name in actor_articulations

        for mp, sub_body in mp_list:
            subsets = [c for c in mp.GetPrim().GetChildren()
                       if c.GetTypeName() == 'GeomSubset']
            # Only propagate sub_body for articulated actors
            sb = sub_body if is_articulated else None
            # For articulated non-root sub-bodies, export vertices in sub-body-local space
            # so the mesh origin matches the <body> origin and AABB pos is body-local.
            # Root sub-body maps to actor_body_elem (at actor world pos), so it must keep
            # actor_M_inv; using its own M_inv would cause a double-offset.
            _artic      = actor_articulations.get(actor_body_name)
            _root_sb    = _artic['root'] if _artic else None
            _is_nonroot = sb is not None and _artic is not None and sb != _root_sb
            if _is_nonroot:
                if sb in _artic['movable']:
                    _sb_M = _artic['movable'][sb].get('M_inv')
                elif sb in _artic['fixed']:
                    _sb_M = _artic['fixed'][sb].get('M_inv')
                else:
                    _sb_M = None
            else:
                _sb_M = None
            export_M_inv = _sb_M if _sb_M is not None else actor_M_inv

            if subsets:
                for subset in subsets:
                    gname    = geom_name()
                    obj_file = f"{gname}.obj"
                    t1 = time.time()
                    ok, bbox = export_geomsubset(mp, subset, xc, os.path.join(mesh_dir, obj_file), export_M_inv)
                    if not ok:
                        print(f"  [FAIL] {gname}")
                        continue
                    mat_name, tex_abs, rgba = diffuse_texture(subset)
                    mat_key = register_material(mat_name, tex_abs, rgba, tex_dir,
                                                materials, mat_key_cache)
                    kb = os.path.getsize(os.path.join(mesh_dir, obj_file)) / 1024
                    print(f"  [OK]   {gname:45s} {kb:8.1f}KB  {mat_key or '-'} ({time.time()-t1:.2f}s)")
                    geom_entries.append({'name': gname, 'obj_file': obj_file,
                                         'mat_key': mat_key, 'actor': actor_body_name,
                                         'bbox': bbox, 'sub_body': sb})
            else:
                gname    = geom_name()
                obj_file = f"{gname}.obj"
                t1 = time.time()
                ok, bbox = export_whole_mesh(mp, xc, os.path.join(mesh_dir, obj_file), export_M_inv)
                if not ok:
                    print(f"  [FAIL] {gname}")
                    continue
                mat_name, tex_abs, rgba = diffuse_texture(mp)
                mat_key = register_material(mat_name, tex_abs, rgba, tex_dir,
                                            materials, mat_key_cache)
                kb = os.path.getsize(os.path.join(mesh_dir, obj_file)) / 1024
                print(f"  [OK]   {gname:45s} {kb:8.1f}KB  {mat_key or '-'} ({time.time()-t1:.2f}s)")
                geom_entries.append({'name': gname, 'obj_file': obj_file,
                                     'mat_key': mat_key, 'actor': actor_body_name,
                                     'bbox': bbox, 'sub_body': sb})

    print(f"\nBuilding MJCF: {len(geom_entries)} geoms, {len(materials)} materials …")
    actor_poses_map = {name: (data[0], data[1]) for name, data in actor_xforms.items()}
    xml_str = minidom.parseString(tostring(
        build_mjcf(geom_entries, materials, scene_id, actor_poses_map,
                   actor_articulations, free_actors)
    )).toprettyxml(indent='  ')
    lines = [l for l in xml_str.splitlines()
             if l.strip() and not l.startswith('<?xml')]
    out_xml = os.path.join(out_dir, f'{scene_id}.xml')
    with open(out_xml, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Done in {time.time()-t0:.1f}s  →  {out_xml}")


def parse_args():
    p = argparse.ArgumentParser(
        description='Convert a USD scene/asset to MuJoCo MJCF (with meshes + textures).')
    p.add_argument('--usd', required=True,
                   help='Path to the input .usd file.')
    p.add_argument('--out', required=True,
                   help='Output directory; the MJCF (<usd-stem>.xml), meshes/ and '
                        'textures/ are written here.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.usd, args.out)

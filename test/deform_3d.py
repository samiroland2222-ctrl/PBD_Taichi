import os
import random

import taichi as ti
import numpy as np
import math

from cons import framework, deform3d, coopers, breast
from cons.torso import UnifiedTorso
from geom import gtet, obj, anatomy, gmesh
from utils import renderer, breast_mesh_generator, parser

from PBD_Taichi.cons.torso import SKIN_HYDRO_ALPHA, SKIN_DEVIA_ALPHA

# ── Skin PBR texture support (wgpu backend only) ──────────────────────────────
try:
    from utils.skin_texture_gen import generate_skin_textures as _gen_skin_tex
    _SKIN_TEX_AVAILABLE = True
except ImportError:
    _SKIN_TEX_AVAILABLE = False

# ── Nipple geometry support ───────────────────────────────────────────────────
try:
    from PBD_Taichi.geom.nipple import (
        make_nipple_pair, eval_hires_positions,
        set_firmness, refresh_hires,
        _NIPPLE_H_FIRM, _NIPPLE_H_RELAX,
        _NIPPLE_R_TIP_FIRM, _NIPPLE_R_TIP_RELAX, _NIPPLE_RADIUS,
    )
    _NIPPLE_AVAILABLE = True
except ImportError:
    _NIPPLE_AVAILABLE = False

# ── Breathing animation ───────────────────────────────────────────────────────
try:
    from PBD_Taichi.utils.breathing import BreathingController, BreathingParams
    _BREATHING_AVAILABLE = True
except ImportError:
    _BREATHING_AVAILABLE = False

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

# ── Ribcage mesh (static visual + optional skin anchor source) ────────────────

def _load_mesh(filepath, scale=1.0, repose=(0, 0, 0)):
    verts, faces = parser.obj_parser(filepath)
    verts = verts * scale
    verts -= verts.mean(axis=0)
    verts += repose
    return gmesh.TrianMesh(verts, faces, dim=3, rho=1.0,
                           get_edge=False, get_edgeside=False,
                           get_edgeNeib=False, get_faceedge=False)

skeleton_mesh = _load_mesh(
    filepath=os.path.join(os.getcwd(), 'assets', 'mesh', 'female_skeleton_first_anatomy_study.OBJ'),
    scale=1/10,
    repose=(0, -0.255, -0.08)
)

ribcage_mesh = _load_mesh(
    filepath=os.path.join(os.getcwd(), 'assets', 'mesh', 'ribcage.obj'),
    scale=1/49,
    repose=(-0.006, 0.04, -0.085)
)

# ── Skeleton (clavicles + upper arms) ─────────────────────────────────────────
skel = anatomy.Skeleton()

# Register the ribcage with the skeleton so it tracks chest_pos each frame.
# Skeleton stores local coords (world − chest_pos) and writes world positions
# back into ribcage_mesh.v_p on every skel.update() call.
skel.set_ribcage_mesh(ribcage_mesh.v_p.to_numpy(),
                      v_p_field=ribcage_mesh.v_p,
                      faces_np=ribcage_mesh.faces_np)

# ── Simulation parameters ─────────────────────────────────────────────────────
g          = (0.0, -9.8, 0.0)
fps        = 60
substep    = 6
solve_step = 2
dt         = 1.0 / (fps * substep)

# ── UnifiedTorso: merged breasts + skin shell ─────────────────────────────────
ribcage_verts = ribcage_mesh.v_p.to_numpy()
torso = UnifiedTorso(
    skel,
    breast_height=0.075,   # slightly shorter for a rounder, less conical profile
    breast_radius=0.07,    # wider base → fuller lower pole
    breast_k=2.0,          # >1 = bulges outward beyond hemisphere (natural round shape)
    breast_spread=0.6,
    breast_tilt=0.3,
    breast_target_tets=600,
    ribcage_verts_np=ribcage_verts,
    ribcage_faces_np=ribcage_mesh.faces_np,
    skin_target_n_tets=20000,   # was 10000 — 2× more outer tris → ~4–5 mm edges, smoother surface
    skin_thickness=0.02,
    g=g,
    dt=dt,
    fps=fps,
    substep=substep,
)

# ── Bounding box ──────────────────────────────────────────────────────────────
all_v = torso.skin_mesh.v_p.to_numpy()
bb_np = np.array([[all_v[:, i].min() - 0.5, all_v[:, i].max() + 0.5]
                  for i in range(3)], dtype=np.float32)
bb = ti.field(dtype=ti.f32, shape=(3, 2))
bb.from_numpy(bb_np)
box3d = obj.BoundBox3D(bound_box=bb, padding=0.01, bound_epsilon=1e-6)
torso.xpbd.add_collision(box3d.collision)

# ── Nipple geometry ───────────────────────────────────────────────────────────
# Build one NippleGeometry per breast.  The hi-res mesh is a pure rendering
# overlay (not in the PBD solver); it follows the breast apex each frame via
# barycentric tracking.
_nipple_l = _nipple_r = None
_nipple_l_vp = _nipple_l_fi = None   # Taichi fields (set below if available)
_nipple_r_vp = _nipple_r_fi = None

if _NIPPLE_AVAILABLE:
    try:
        _ref_verts = torso.skin_mesh.v_p_ref.to_numpy()
        _bl_v_rest = _ref_verts[:torso.n_left_verts].astype(np.float32)
        _br_v_rest = _ref_verts[torso.n_left_verts:
                                 torso.n_left_verts + torso.n_right_verts].astype(np.float32)

        # ── Skin outer-shell anchor mesh ────────────────────────────────────
        # The nipple cage base vertices are bound to the extruded outer skin
        # layer so they track the skin surface, not the underlying breast flesh.
        _skin_outer_anchor_verts = None
        _skin_outer_anchor_faces = None
        if (torso._skin_outer_vert_offset is not None
                and torso._skin_outer_faces_np is not None):
            _skin_outer_n = torso.n_skin_verts - torso.n_skin_inner_verts
            _skin_outer_start = torso._skin_outer_vert_offset
            _skin_outer_anchor_verts = _ref_verts[
                _skin_outer_start : _skin_outer_start + _skin_outer_n
            ].astype(np.float32)
            _skin_outer_anchor_faces = torso._skin_outer_faces_np
            print(f"[nipple] skin outer anchor: {len(_skin_outer_anchor_verts)} verts, "
                  f"{len(_skin_outer_anchor_faces)} faces")

        _nipple_l, _nipple_r = make_nipple_pair(
            _bl_v_rest, torso._breast_l_faces_np,
            _br_v_rest, torso._breast_r_faces_np,
            base_l_vertex_indices=torso._breast_l_base_local_np,
            base_r_vertex_indices=torso._breast_r_base_local_np,
            anchor_verts=_skin_outer_anchor_verts,
            anchor_faces=_skin_outer_anchor_faces,
            n_sides=4, hires_segs=16, hires_rings=7,
        )
        # Create Taichi fields for the hi-res vertices (updated each frame)
        _nipple_l_vp = ti.Vector.field(3, dtype=ti.f32, shape=len(_nipple_l.hires_verts))
        _nipple_r_vp = ti.Vector.field(3, dtype=ti.f32, shape=len(_nipple_r.hires_verts))
        _nipple_l_vp.from_numpy(_nipple_l.hires_verts)
        _nipple_r_vp.from_numpy(_nipple_r.hires_verts)
        # Taichi fields for static face indices
        _nl_fi_np = _nipple_l.hires_faces.flatten().astype(np.int32)
        _nr_fi_np = _nipple_r.hires_faces.flatten().astype(np.int32)
        _nipple_l_fi = ti.field(dtype=ti.i32, shape=len(_nl_fi_np))
        _nipple_r_fi = ti.field(dtype=ti.i32, shape=len(_nr_fi_np))
        _nipple_l_fi.from_numpy(_nl_fi_np)
        _nipple_r_fi.from_numpy(_nr_fi_np)
        print(f"[nipple] built: L={len(_nipple_l.hires_verts)} verts "
              f"{len(_nipple_l.hires_faces)} tris | "
              f"R={len(_nipple_r.hires_verts)} verts "
              f"{len(_nipple_r.hires_faces)} tris")
    except Exception as _e:
        print(f"[nipple] WARNING: failed to build nipple geometry: {_e}")
        import traceback; traceback.print_exc()
        _nipple_l = _nipple_r = None

# ── Nipple rebuild helper (called after breast reshape) ───────────────────────
def _rebuild_nipples_from_torso(torso_inst=None):
    """Rebuild nipple geometry from the current (post-rebuild) breast mesh."""
    global _nipple_l, _nipple_r, _nipple_l_vp, _nipple_r_vp, _nipple_l_fi, _nipple_r_fi
    if not _NIPPLE_AVAILABLE:
        return
    try:
        _ref = torso.skin_mesh.v_p_ref.to_numpy()
        _bl = _ref[:torso.n_left_verts].astype(np.float32)
        _br = _ref[torso.n_left_verts:torso.n_left_verts + torso.n_right_verts].astype(np.float32)

        _anc_v = _skin_outer_anchor_verts
        _anc_f = _skin_outer_anchor_faces

        nl, nr = make_nipple_pair(
            _bl, torso._breast_l_faces_np,
            _br, torso._breast_r_faces_np,
            base_l_vertex_indices=torso._breast_l_base_local_np,
            base_r_vertex_indices=torso._breast_r_base_local_np,
            anchor_verts=_anc_v, anchor_faces=_anc_f,
            n_sides=4, hires_segs=16, hires_rings=7,
        )
        _nipple_l = nl;  _nipple_r = nr

        _nipple_l_vp.from_numpy(_nipple_l.hires_verts)
        _nipple_r_vp.from_numpy(_nipple_r.hires_verts)
        _nipple_l_fi.from_numpy(_nipple_l.hires_faces.flatten().astype(np.int32))
        _nipple_r_fi.from_numpy(_nipple_r.hires_faces.flatten().astype(np.int32))
        print("[nipple] rebuilt after breast reshape")
    except Exception as _e:
        print(f"[nipple] WARNING: rebuild after reshape failed: {_e}")
        import traceback; traceback.print_exc()

# Register the nipple rebuild callback with the torso
torso._breast_rebuild_callbacks.append(_rebuild_nipples_from_torso)

# ── Renderer ──────────────────────────────────────────────────────────────────
tirender = renderer.TaichiRenderer3D("Deform 3D – Unified Torso",
                                     res=(900, 900), fps=fps,
                                     cameraPos=(0.5, 0.15, 0.2),
                                     cameraLookat=(-0.4, -0.03, -0.17))

# ── Detect wgpu backend and set up PBR skin ───────────────────────────────────
_wgpu = hasattr(tirender, 'setup_skin_lighting')

# PBR skin texture state (wgpu only)
_skin_tex       = None   # SkinTextures instance (or None)
_skin_tex_seed  = [42]   # mutable so the keybind closure can update it
_pbr_skin_on    = [True] # GUI toggle — live switch between PBR and flat-color

# Phase 6 / 4 / 5 skin params — applied on next T-key regen
_skin_params = {
    'melanin_amount':      [0.076],    # Phase 6: 0=albino, 1=dark
    'haemo_amount':        [0.38],    # Phase 6: 0=bloodless, 1=flushed
    'oxygenation':         [0.831],    # Phase 6: 0=cyanotic, 1=healthy
    'clearcoat':           [0.054],    # Phase 4: oil/sebum layer strength
    'clearcoat_roughness': [0.174],    # Phase 4: sharpness of oil highlight
    'sheen':               [0.2688],    # Phase 5: peach-fuzz intensity
    'sheen_roughness':     [0.402],    # Phase 5: fuzz softness
}

def _build_skin_textures(seed: int):
    """Generate (or regenerate) skin PBR textures from the current skin mesh."""
    if not (_SKIN_TEX_AVAILABLE and _wgpu and torso.skin_f_i is not None):
        return None
    verts_np = torso.skin_mesh.v_p.to_numpy()
    faces_np = torso.skin_f_i.to_numpy().reshape(-1, 3)
    if len(faces_np) == 0:
        return None

    # Phase 2a: compute per-vertex thinness from skin shell binding distances.
    # thin=1 → close to anatomy (ears/thin areas → stronger SSS glow).
    thickness_pv = torso.get_skin_thickness_per_vert_np()

    print(f"[skin PBR] generating textures (seed={seed}, "
          f"{len(verts_np)} verts, {len(faces_np)} faces) …")
    tex = _gen_skin_tex(
        verts_np, faces_np,
        resolution=4096,
        seed=seed,
        # pore_cell_size=0 → auto-scales to 1/180 of bbox diagonal (fine micro-texture)
        freckle_density=0.40,
        base_roughness=0.55,
        dewy_intensity=0.10,
        emissive_intensity=0.08,
        normal_strength=0.25,
        thickness_per_vertex=thickness_pv,   # Phase 2: modulates SSS emissive glow
        # Phase 6: spectral skin colour model
        melanin_amount      = _skin_params['melanin_amount'][0],
        haemo_amount        = _skin_params['haemo_amount'][0],
        oxygenation         = _skin_params['oxygenation'][0],
        # Phase 4: clearcoat (dual-lobe specular)
        clearcoat           = _skin_params['clearcoat'][0],
        clearcoat_roughness = _skin_params['clearcoat_roughness'][0],
        # Phase 5: sheen (peach-fuzz)
        sheen               = _skin_params['sheen'][0],
        sheen_roughness     = _skin_params['sheen_roughness'][0],
        verbose=True,
        debug_save_dir="/tmp/skin_debug",   # saves atlas PNGs + mesh-overlay PNGs
    )
    if tex is not None:
        tex.save_with_overlay("/tmp/skin_debug", prefix="skin")

    # ── Paint areola / nipple pigmentation onto the skin atlas ───────────────
    if tex is not None and _nipple_l is not None:
        try:
            from PBD_Taichi.utils.nipple_texture import paint_nipple_areola, bake_world_pos_map
            world_pos_map, atlas_mask = bake_world_pos_map(verts_np, tex)
            for nip in [_nipple_l, _nipple_r]:
                paint_nipple_areola(
                    tex,
                    world_pos_map  = world_pos_map,
                    mask           = atlas_mask,
                    nipple_center  = nip.apex_world,
                    melanin_amount = _skin_params['melanin_amount'][0],
                    haemo_amount   = _skin_params['haemo_amount'][0],
                    oxygenation    = _skin_params['oxygenation'][0],
                    rebuild_gpu_textures = True,
                    verbose=True,
                )
            print("[skin PBR] nipple/areola painted onto atlas")
        except Exception as _tex_e:
            print(f"[skin PBR] WARNING: nipple texture painting failed: {_tex_e}")
            import traceback; traceback.print_exc()

    return tex

if _wgpu:
    tirender.setup_skin_lighting()
    tirender.setup_sss(sss_strength=0.28, sigma_r=5.0, sigma_g=3.0, sigma_b=1.5)
    _skin_tex = _build_skin_textures(_skin_tex_seed[0])

skin_color = (0.85, 0.65, 0.55)

# ── Scene render draws ────────────────────────────────────────────────────────
tirender.add_scene_render_draw(skeleton_mesh.get_render_draw(color=(0.7, 0.7, 0.5), wireframe=False))
tirender.add_scene_render_draw(ribcage_mesh.get_render_draw(color=(0.7, 0.7, 0.5), wireframe=False))

# ── Breathing animation ───────────────────────────────────────────────────────
_breath_ctrl = None
if _BREATHING_AVAILABLE:
    _breath_ctrl = BreathingController(BreathingParams(
        rate_bpm=15.0,
        chest_amplitude=0.008,
        abdomen_amplitude=0.012,
    ))
    _breath_ctrl.snapshot_ribcage_rest(skel)
    print("[breathing] controller ready")

if _wgpu and _skin_tex is not None:
    # PBR skin draw: replaces the flat-color all-faces draw when PBR is active.
    # Falls back to flat-color draw automatically if _pbr_skin_on[0] is False.
    def _skin_pbr_draw(scene):
        if _pbr_skin_on[0] and _skin_tex is not None:
            scene.skin_mesh(torso.skin_mesh.v_p, _skin_tex)
        else:
            scene.mesh(torso.skin_mesh.v_p, torso.skin_f_i,
                       color=skin_color, show_wireframe=False, two_sided=False)
    tirender.add_scene_render_draw(_skin_pbr_draw)

    # Breast interior (non-skin all-faces draw) shown when PBR is OFF only,
    # so the inner structure is accessible via the x-slice toggle in PBR mode.
    def _breast_flat_draw(scene):
        if not _pbr_skin_on[0]:
            scene.mesh(torso.skin_mesh.v_p, torso.skin_mesh.f_i,
                       color=skin_color, show_wireframe=False, two_sided=False)
    tirender.add_scene_render_draw(_breast_flat_draw)
else:
    # Taichi backend or texture generation failed → existing flat-color draw
    for draw in torso.get_render_draws():
        tirender.add_scene_render_draw(draw)

for draw in skel.get_render_draws():
    tirender.add_scene_render_draw(draw)
for draw in torso.get_ligament_draws():
    tirender.add_scene_render_draw(draw)
for draw in torso.get_skin_anchor_draws():   # reddish bilateral breast-skin springs
    tirender.add_scene_render_draw(draw)

# ── Nipple render draws ───────────────────────────────────────────────────────
# Rendered as a separate mesh overlay on top of the breast/skin surface.
# The hi-res verts are updated each frame from the breast apex barycentric tracking.
_nipple_color = (0.70, 0.40, 0.30)  # warm-terracotta areola tint (flat renderer)
if _nipple_l is not None and _nipple_l_vp is not None:
    def _draw_nipple_l(scene):
        scene.mesh(_nipple_l_vp, _nipple_l_fi,
                   color=_nipple_color, show_wireframe=False, two_sided=True)
    def _draw_nipple_r(scene):
        scene.mesh(_nipple_r_vp, _nipple_r_fi,
                   color=_nipple_color, show_wireframe=False, two_sided=True)
    tirender.add_scene_render_draw(_draw_nipple_l)
    tirender.add_scene_render_draw(_draw_nipple_r)

# ── Surface-normal overlay (wgpu backend: yellow lines; Taichi backend: no-op) ─
if torso.skin_f_i is not None:
    _skin_fi_np = torso.skin_f_i.to_numpy().reshape(-1, 3)
    _normals_draw = tirender.make_normals_draw_callback(
        lambda: torso.skin_mesh.v_p.to_numpy().astype(np.float32),
        lambda: _skin_fi_np,
    )
    tirender.add_scene_render_draw(_normals_draw)

# ── GUI ───────────────────────────────────────────────────────────────────────
# PBR knobs (wgpu only) — adjusted live via the material properties
_emissive_intensity = [0.312]
_normal_strength    = [0.406]   # matches the baked normal_strength; 0–1 range on slider
_nipple_firmness    = [0.0]     # 0=flat/relaxed, 1=firm/erect
log_b_hydro   = [math.log10(torso.deform_breast.hydro_alpha)]
log_b_devia   = [math.log10(torso.deform_breast.devia_alpha)]
log_s_hydro   = [math.log10(torso.deform_skin.hydro_alpha)]  if torso.deform_skin else [math.log10(SKIN_HYDRO_ALPHA)]
log_s_devia   = [math.log10(torso.deform_skin.devia_alpha)]  if torso.deform_skin else [math.log10(SKIN_DEVIA_ALPHA)]
log_lig_alpha  = [math.log10(torso.ligaments_l.alpha)] if torso.ligaments_l else [0.0]
log_fascia_alpha = [math.log10(torso.breast_l_fascia_springs.alpha)] if torso.breast_l_fascia_springs and torso.breast_l_fascia_springs.n > 0 else [-4.0]
log_skin_alpha = [math.log10(torso.skin_anchors.alpha)] if torso.skin_anchors.n > 0 else [0.0]
log_skel_alpha = [math.log10(torso.skeleton_skin_springs.alpha)] if torso.skeleton_skin_springs.n > 0 else [0.0]
log_rc_alpha   = [math.log10(torso.ribcage_skin_springs.alpha)] if torso.ribcage_skin_springs.n > 0 else [0.0]

def gui_draw(gui):
    gui.text("-- Breast tissue stiffness --")
    log_b_hydro[0] = gui.slider_float("log10(breast hydro)", log_b_hydro[0], -3.0, 3.0)
    log_b_devia[0] = gui.slider_float("log10(breast devia)", log_b_devia[0], -3.0, 3.0)
    torso.deform_breast.hydro_alpha = 10 ** log_b_hydro[0]
    torso.deform_breast.devia_alpha = 10 ** log_b_devia[0]
    gui.text(f"  hydro={torso.deform_breast.hydro_alpha:.2e}  devia={torso.deform_breast.devia_alpha:.2e}")

    if torso.deform_skin:
        gui.text("-- Skin / subcut-fat stiffness --")
        log_s_hydro[0] = gui.slider_float("log10(skin hydro)", log_s_hydro[0], -3.0, 3.0)
        log_s_devia[0] = gui.slider_float("log10(skin devia)", log_s_devia[0], -3.0, 3.0)
        torso.deform_skin.hydro_alpha = 10 ** log_s_hydro[0]
        torso.deform_skin.devia_alpha = 10 ** log_s_devia[0]
        gui.text(f"  hydro={torso.deform_skin.hydro_alpha:.2e}  devia={torso.deform_skin.devia_alpha:.2e}")

    gui.text("-- Cooper's ligaments --")
    log_lig_alpha[0] = gui.slider_float("log10(lig)", log_lig_alpha[0], -1.0, 6.0)
    if torso.ligaments_l:
        torso.ligaments_l.alpha = 10 ** log_lig_alpha[0]
        torso.ligaments_r.alpha = 10 ** log_lig_alpha[0]
        gui.text(f"  alpha={torso.ligaments_l.alpha:.2e}")

    if torso.breast_l_fascia_springs and torso.breast_l_fascia_springs.n > 0:
        gui.text("-- Breast-base <-> fascia springs --")
        log_fascia_alpha[0] = gui.slider_float("log10(fascia)", log_fascia_alpha[0], -3.0, 6.0)
        torso.breast_l_fascia_springs.alpha = 10 ** log_fascia_alpha[0]
        torso.breast_r_fascia_springs.alpha = 10 ** log_fascia_alpha[0]
        gui.text(f"  alpha={torso.breast_l_fascia_springs.alpha:.2e}"
                 f"  n_L={torso.breast_l_fascia_springs.n}"
                 f"  n_R={torso.breast_r_fascia_springs.n}")

    if torso.skin_anchors.n > 0:
        gui.text("-- Breast-skin springs --")
        log_skin_alpha[0] = gui.slider_float("log10(breast-skin)", log_skin_alpha[0], -4.0, 6.0)
        torso.skin_anchors.alpha = 10 ** log_skin_alpha[0]
        gui.text(f"  alpha={torso.skin_anchors.alpha:.2e}  n={torso.skin_anchors.n}")

    if torso.skeleton_skin_springs.n > 0:
        gui.text("-- Skeleton springs (clav+arm) --")
        log_skel_alpha[0] = gui.slider_float("log10(skel)", log_skel_alpha[0], -4.0, 6.0)
        torso.skeleton_skin_springs.alpha = 10 ** log_skel_alpha[0]
        gui.text(f"  alpha={torso.skeleton_skin_springs.alpha:.2e}  n={torso.skeleton_skin_springs.n}")

    if torso.ribcage_skin_springs.n > 0:
        gui.text("-- Ribcage-skin springs --")
        log_rc_alpha[0] = gui.slider_float("log10(ribcage-skin)", log_rc_alpha[0], -4.0, 6.0)
        torso.ribcage_skin_springs.alpha = 10 ** log_rc_alpha[0]
        gui.text(f"  alpha={torso.ribcage_skin_springs.alpha:.2e}  n={torso.ribcage_skin_springs.n}")

    gui.text("-- Clavicle joints --")
    skel.clavicle_left_pitch  = gui.slider_float("L pitch", skel.clavicle_left_pitch, -0.5, 0.6)
    skel.clavicle_left_yaw    = gui.slider_float("L yaw", skel.clavicle_left_yaw, -0.5, 0.5)
    skel.clavicle_right_pitch = gui.slider_float("R pitch", skel.clavicle_right_pitch, -0.5, 0.6)
    skel.clavicle_right_yaw   = gui.slider_float("R yaw", skel.clavicle_right_yaw, -0.5, 0.5)

    gui.text("-- Arm joints --")
    skel.arm_left_flexion    = gui.slider_float("L flex", skel.arm_left_flexion, -1.0, 2.5)
    skel.arm_left_abduction  = gui.slider_float("L abd", skel.arm_left_abduction, -0.3, 2.5)
    skel.arm_right_flexion   = gui.slider_float("R flex", skel.arm_right_flexion, -1.0, 2.5)
    skel.arm_right_abduction = gui.slider_float("R abd", skel.arm_right_abduction, -0.3, 2.5)

    # ── PBR Skin (wgpu only) ──────────────────────────────────────────────────
    if _wgpu and _skin_tex is not None:
        gui.text("-- PBR Skin --")
        _pbr_skin_on[0] = gui.checkbox("PBR skin on", _pbr_skin_on[0])
        if _pbr_skin_on[0]:
            _emissive_intensity[0] = gui.slider_float(
                "SSS emissive", _emissive_intensity[0], 0.0, 0.5)
            _normal_strength[0] = gui.slider_float(
                "Normal strength", _normal_strength[0], 0.0, 1.0)
            # Apply to the live material immediately
            _skin_tex.material.emissive_intensity = _emissive_intensity[0]
            ns = _normal_strength[0]
            _skin_tex.material.normal_scale = (ns, ns)

            # ── Phase 4: clearcoat (live, no regen needed) ─────────────────
            gui.text("-- Clearcoat (oil/sebum layer) --")
            _skin_params['clearcoat'][0] = gui.slider_float(
                "Clearcoat", _skin_params['clearcoat'][0], 0.0, 1.0)
            _skin_params['clearcoat_roughness'][0] = gui.slider_float(
                "Coat roughness", _skin_params['clearcoat_roughness'][0], 0.0, 1.0)
            _skin_tex.material.clearcoat = _skin_params['clearcoat'][0]
            _skin_tex.material.clearcoat_roughness = _skin_params['clearcoat_roughness'][0]

            # ── Phase 5: sheen (live, no regen needed) ─────────────────────
            gui.text("-- Sheen (peach-fuzz) --")
            _skin_params['sheen'][0] = gui.slider_float(
                "Sheen", _skin_params['sheen'][0], 0.0, 1.0)
            _skin_params['sheen_roughness'][0] = gui.slider_float(
                "Sheen roughness", _skin_params['sheen_roughness'][0], 0.0, 1.0)
            _skin_tex.material.sheen = _skin_params['sheen'][0]
            _skin_tex.material.sheen_roughness = _skin_params['sheen_roughness'][0]

            # ── Phase 6: spectral skin tone (requires T-key regen) ─────────
            gui.text("-- Skin tone (press T to regen) --")
            _skin_params['melanin_amount'][0] = gui.slider_float(
                "Melanin", _skin_params['melanin_amount'][0], 0.0, 1.0)
            _skin_params['haemo_amount'][0] = gui.slider_float(
                "Haemoglobin", _skin_params['haemo_amount'][0], 0.0, 1.0)
            _skin_params['oxygenation'][0] = gui.slider_float(
                "Oxygenation", _skin_params['oxygenation'][0], 0.0, 1.0)
            # Show predicted base colour from the spectral model
            try:
                from utils.skin_texture_gen import spectral_skin_base_color
                import numpy as _np
                c = spectral_skin_base_color(
                    _skin_params['melanin_amount'][0],
                    _skin_params['haemo_amount'][0],
                    _skin_params['oxygenation'][0],
                )
                # sRGB for display (gamma encode)
                cs = _np.clip(1.055 * _np.power(_np.clip(c, 0, 1), 1/2.4) - 0.055, 0, 1)
                gui.text(f"  sRGB≈ ({cs[0]:.2f},{cs[1]:.2f},{cs[2]:.2f})  (press T to apply)")
            except Exception:
                pass

            gui.text("  Press T to regenerate textures (new seed)")

    # ── Nipple firmness ───────────────────────────────────────────────────────
    if _nipple_l is not None:
        gui.text("-- Nipple erection --")
        new_firmness = gui.slider_float("Firmness", _nipple_firmness[0], 0.0, 1.0)
        if new_firmness != _nipple_firmness[0]:
            _nipple_firmness[0] = new_firmness
            set_firmness(_nipple_l, new_firmness)
            refresh_hires(_nipple_l)
            _nipple_l_vp.from_numpy(_nipple_l.hires_verts)
            set_firmness(_nipple_r, new_firmness)
            refresh_hires(_nipple_r)
            _nipple_r_vp.from_numpy(_nipple_r.hires_verts)
        gui.text(f"  {'flat/relaxed' if _nipple_firmness[0] < 0.1 else 'firm/erect' if _nipple_firmness[0] > 0.9 else 'transitioning'}  ({_nipple_firmness[0]:.2f})")

    # ── Breathing ─────────────────────────────────────────────────────────────
    if _breath_ctrl is not None:
        gui.text("-- Breathing --")
        _breath_ctrl.enabled = gui.checkbox("Breathing on", _breath_ctrl.enabled)
        _breath_ctrl.params.rate_bpm = gui.slider_float(
            "Rate (bpm)", _breath_ctrl.params.rate_bpm, 0.0, 40.0)
        _breath_ctrl.params.chest_amplitude = gui.slider_float(
            "Chest amp (mm)",
            _breath_ctrl.params.chest_amplitude * 1000, 0.0, 25.0) / 1000.0
        _breath_ctrl.params.abdomen_amplitude = gui.slider_float(
            "Abdomen amp (mm)",
            _breath_ctrl.params.abdomen_amplitude * 1000, 0.0, 30.0) / 1000.0
        ph = _breath_ctrl.breath_phase(sim['frame'] / fps) if sim['frame'] > 0 else 0.0
        stage = "inhale" if ph > 0.05 else "exhale"
        gui.text(f"  {stage}  phase={ph:.2f}  "
                 f"{_breath_ctrl.params.rate_bpm:.1f} bpm")

    # ── Breast live reshape ───────────────────────────────────────────────────
    gui.text("-- Breast shape (Rebuild to apply) --")
    bp = torso._breast_build_params
    bp['breast_height'] = gui.slider_float(
        "Height (mm)", bp['breast_height'] * 1000, 20.0, 150.0) / 1000.0
    bp['breast_radius'] = gui.slider_float(
        "Radius (mm)", bp['breast_radius'] * 1000, 20.0, 120.0) / 1000.0
    bp['breast_k']      = gui.slider_float("Shape k",   bp['breast_k'],      0.3, 3.0)
    bp['breast_spread'] = gui.slider_float("Spread",     bp['breast_spread'],  0.1, 1.2)
    bp['breast_tilt']   = gui.slider_float("Tilt",       bp['breast_tilt'],    0.0, 0.8)
    if gui.button("Rebuild breasts"):
        torso.rebuild_breasts()   # uses stored _breast_build_params

tirender.add_gui_draw(gui_draw)

# ── Simulation control ────────────────────────────────────────────────────────
sim = {'paused': False, 'step_once': False, 'sim_rate': 1.0, 'frame': 0}

def sim_reset():
    torso.reset()
    skel.reset_pose()
    sim['frame'] = 0

def gui_draw_debug(gui):
    gui.text("── Simulation control ──")
    if gui.button("Resume" if sim['paused'] else "Pause"):
        sim['paused'] = not sim['paused']
    if gui.button("Step"):
        sim['paused'] = True;  sim['step_once'] = True
    if gui.button("Reset"):
        sim_reset()
    sim['sim_rate'] = gui.slider_float("Sim rate", sim['sim_rate'], 0.0, 2.0)
    status = 'PAUSED' if sim['paused'] else f"x{sim['sim_rate']:.2f}"
    gui.text(f"  frame={sim['frame']}  {status}")

tirender.add_gui_draw(gui_draw_debug)

# ── Keybind: T → regenerate skin textures with new seed ──────────────────────
if _wgpu and _SKIN_TEX_AVAILABLE:
    def _regen_skin_textures():
        global _skin_tex
        _skin_tex_seed[0] = random.randint(0, 9999)
        _skin_tex = _build_skin_textures(_skin_tex_seed[0])
        if _skin_tex is not None:
            _skin_tex.material.emissive_intensity = _emissive_intensity[0]
        print(f"[skin PBR] regenerated (seed={_skin_tex_seed[0]})")
    tirender.add_click_event('t', _regen_skin_textures)

# ── Nipple per-frame update ───────────────────────────────────────────────────

def _update_nipple_cage_from_breast(nipple, breast_verts_current, breast_faces):
    """Drive the nipple cage from current breast vertex positions.

    Base ring verts (cage indices 0..S-1) are positioned via their
    BarycentricBindingDefinitions.  Tip ring + centre are reconstructed
    from the current base centroid and a live EMA-smoothed outward normal.
    """
    _NORMAL_EMA_ALPHA = 0.20   # smoothing factor (higher = more responsive to breast shape)

    S = len(nipple.cage_bindings)  # == n_sides (4)
    h = float(_NIPPLE_H_RELAX + nipple.firmness * (_NIPPLE_H_FIRM  - _NIPPLE_H_RELAX))
    r_tip = float(_NIPPLE_R_TIP_RELAX + nipple.firmness * (_NIPPLE_R_TIP_FIRM - _NIPPLE_R_TIP_RELAX))

    # 1. Update base ring from barycentric bindings onto breast surface
    for b in nipple.cage_bindings:
        fi = b.anchor_bary_face
        w  = b.anchor_bary_uvw        # (3,)
        tri = breast_faces[fi]         # [v0, v1, v2]
        pos = (w[0] * breast_verts_current[tri[0]]
             + w[1] * breast_verts_current[tri[1]]
             + w[2] * breast_verts_current[tri[2]])
        nipple.cage_verts[b.skin_vertex_index] = pos

    # 2. Estimate current outward normal from the updated base ring
    base_ring  = nipple.cage_verts[:S]              # (S, 3)
    centroid   = base_ring.mean(axis=0)             # (3,)
    if S >= 3:
        d1 = (base_ring[2] - base_ring[0]).astype(np.float64)
        d2 = (base_ring[3 % S] - base_ring[1 % S]).astype(np.float64)
        est_n = np.cross(d1, d2)
        nlen  = np.linalg.norm(est_n)
        if nlen > 1e-9:
            est_n /= nlen
        else:
            est_n = nipple.apex_frame[:, 0].astype(np.float64)
    else:
        est_n = nipple.apex_frame[:, 0].astype(np.float64)

    # ── EMA smoothing of normal to prevent flickering ─────────────────────────
    # Initialise EMA state on first call
    if not hasattr(nipple, '_smooth_n'):
        nipple._smooth_n = nipple.apex_frame[:, 0].astype(np.float64).copy()

    prev_n = nipple._smooth_n
    # Guard against degenerate flips (prefer continuity with previous frame)
    if np.dot(est_n, prev_n) < 0:
        est_n = -est_n
    smooth_n = _NORMAL_EMA_ALPHA * est_n + (1.0 - _NORMAL_EMA_ALPHA) * prev_n
    nlen = np.linalg.norm(smooth_n)
    if nlen > 1e-9:
        smooth_n /= nlen
    else:
        smooth_n = prev_n.copy()
    nipple._smooth_n = smooth_n.copy()
    n_hat = smooth_n

    # ── Gram-Schmidt orthonormal frame from live n_hat ────────────────────────
    t_hat = nipple.apex_frame[:, 1].astype(np.float64)
    t_hat -= np.dot(t_hat, n_hat) * n_hat
    t_norm = np.linalg.norm(t_hat)
    if t_norm < 1e-8:
        hint  = np.array([0.0, 1.0, 0.0]) if abs(n_hat[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t_hat = hint - np.dot(hint, n_hat) * n_hat
        t_hat /= np.linalg.norm(t_hat) + 1e-12
    else:
        t_hat /= t_norm
    b_hat = np.cross(n_hat, t_hat)
    b_hat /= np.linalg.norm(b_hat) + 1e-12

    # 3. Reconstruct tip ring and centre from centroid + live normal
    angles = np.linspace(0, 2 * np.pi, S, endpoint=False)
    for i in range(S):
        nipple.cage_verts[S + i] = (centroid
            + h      * n_hat
            + r_tip  * np.cos(angles[i]) * t_hat
            + r_tip  * np.sin(angles[i]) * b_hat).astype(np.float32)
    nipple.cage_verts[2 * S] = (centroid + (h + r_tip * 0.2) * n_hat).astype(np.float32)


def _update_nipples():
    """Read skin-shell positions from simulation, update nipple hi-res mesh."""
    if _nipple_l is None:
        return
    all_verts = torso.skin_mesh.v_p.to_numpy().astype(np.float32)

    # ── Resolve cage-base bindings ───────────────────────────────────────────
    # If the nipple was bound to the outer skin shell use those live vertices;
    # otherwise fall back to the breast flesh vertices (legacy behaviour).
    if (_skin_outer_anchor_verts is not None
            and torso._skin_outer_vert_offset is not None):
        _skin_outer_n     = torso.n_skin_verts - torso.n_skin_inner_verts
        _skin_outer_start = torso._skin_outer_vert_offset
        anchor_verts_live = all_verts[_skin_outer_start : _skin_outer_start + _skin_outer_n]
        anchor_faces_live = torso._skin_outer_faces_np
    else:
        # Fallback: breast flesh
        n_l = torso.n_left_verts
        n_r = torso.n_right_verts
        anchor_verts_live = None   # signals per-nipple split below
        anchor_faces_live = None

    if anchor_verts_live is not None:
        # Both nipples share the same skin-shell anchor mesh
        _update_nipple_cage_from_breast(_nipple_l, anchor_verts_live, anchor_faces_live)
        _update_nipple_cage_from_breast(_nipple_r, anchor_verts_live, anchor_faces_live)
    else:
        n_l = torso.n_left_verts
        n_r = torso.n_right_verts
        _update_nipple_cage_from_breast(_nipple_l, all_verts[:n_l],
                                        torso._breast_l_faces_np)
        _update_nipple_cage_from_breast(_nipple_r, all_verts[n_l:n_l + n_r],
                                        torso._breast_r_faces_np)

    _nipple_l.hires_verts[:] = eval_hires_positions(
        _nipple_l.cage_verts, _nipple_l.cage_tets,
        _nipple_l.hires_tet_idx, _nipple_l.hires_bary)
    _nipple_l_vp.from_numpy(_nipple_l.hires_verts)

    _nipple_r.hires_verts[:] = eval_hires_positions(
        _nipple_r.cage_verts, _nipple_r.cage_tets,
        _nipple_r.hires_tet_idx, _nipple_r.hires_bary)
    _nipple_r_vp.from_numpy(_nipple_r.hires_verts)


# ── Main loop ─────────────────────────────────────────────────────────────────
import time as _time
_accum     = 0.0
_wall_prev = _time.time()

while tirender.window.running:
    tirender.handle_input()

    # Update skeleton and kinematic skin anchors once per frame
    # Breathing: apply chest expansion BEFORE skel.update() so ribcage
    # positions are correct when kinematic springs are resolved.
    # sim_time uses wall-clock frame count / fps so breathing rate is correct.
    # (Using frame * dt would give 1/substep × real time, making breathing ~6× too slow.)
    sim_time = sim['frame'] / fps
    if _breath_ctrl is not None:
        _breath_ctrl.apply_to_ribcage(skel, sim_time)
    skel.update()
    torso.update_kinematic_skin()
    # Breathing: drive abdominal skin targets AFTER kinematic skin update
    if _breath_ctrl is not None:
        _breath_ctrl.apply_abdomen(torso, sim_time)
    if torso.ligaments_l:
        torso.ligaments_l.update_anchors(skel.get_fascia_left_surface_anchors_np())
        torso.ligaments_r.update_anchors(skel.get_fascia_right_surface_anchors_np())

    wall_now   = _time.time()
    wall_delta = wall_now - _wall_prev
    _wall_prev = wall_now

    should_step = False
    if sim['step_once']:
        should_step = True;  sim['step_once'] = False
    elif not sim['paused']:
        _accum += wall_delta * sim['sim_rate']
        if _accum >= 1.0 / fps:
            _accum -= 1.0 / fps
            should_step = True

    if should_step:
        for _ in range(substep):
            torso.xpbd.make_prediction_pinned(torso.skin_mesh.v_invm)
            torso.xpbd.preupdate_cons()
            for _ in range(solve_step):
                torso.xpbd.update_cons()
            torso.xpbd.update_vel_pinned(torso.skin_mesh.v_invm)
        sim['frame'] += 1

    # Update nipple hi-res mesh to follow breast apex each frame
    _update_nipples()

    tirender.render()

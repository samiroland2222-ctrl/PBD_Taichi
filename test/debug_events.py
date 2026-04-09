"""Interactive event-pipeline debugger for the wgpu renderer.

Run from PBD_Taichi/:
    python test/debug_events.py

Three phases are tested in sequence (the window title tells you which one):

  Phase 1 – raw canvas only
    Tap keys, move mouse, scroll, click.
    Expected: every action prints a "CANVAS" line.

  Phase 2 – canvas + WgpuRenderer event forwarding
    Same actions.
    Expected: every action also prints a "RENDERER" line.

  Phase 3 – canvas + renderer + OrbitController (with explicit camera)
    Move mouse (no button), then hold left-button and drag slowly.
    Expected: "CONTROLLER" lines; camera position prints when it moves.

Each phase lasts ~15 seconds, then advances automatically.
Press Ctrl-C to abort early.
"""

import sys, time
sys.path.insert(0, '.')

import numpy as np
import pygfx
from rendercanvas.glfw import GlfwRenderCanvas

PHASE_DURATION = 15.0  # seconds per phase

# ---------------------------------------------------------------------------
# Shared scene (something visible so the window isn't just black)
# ---------------------------------------------------------------------------
def _make_scene():
    scene = pygfx.Scene()
    scene.add(pygfx.AmbientLight(intensity=1.0))
    geo = pygfx.box_geometry(1, 1, 1)
    mat = pygfx.MeshPhongMaterial(color=(0.5, 0.7, 0.9, 1))
    scene.add(pygfx.Mesh(geo, mat))
    return scene


# ===========================================================================
# Phase 1 – raw canvas event dispatch
# ===========================================================================
def phase1():
    print("\n" + "="*60)
    print("PHASE 1 – raw canvas events")
    print("  Tap keys, move mouse, scroll, left/right-click.")
    print("  You should see a CANVAS line for every action.")
    print("="*60)

    canvas = GlfwRenderCanvas(
        title="Phase 1 – raw canvas", size=(640, 480), update_mode="manual")
    renderer = pygfx.WgpuRenderer(canvas)
    scene = _make_scene()
    cam = pygfx.PerspectiveCamera(50, 640/480, depth_range=(0.01, 100))
    cam.local.position = np.array([0, 0, 3], dtype=float)
    cam.look_at(np.array([0, 0, 0], dtype=float))

    event_count = [0]

    def _log(ev):
        event_count[0] += 1
        t = ev.get("event_type", "?")
        # concise one-liner per event
        if t in ("pointer_move",):
            print(f"  CANVAS  pointer_move  ({ev.get('x',0):.0f}, {ev.get('y',0):.0f})")
        elif t == "wheel":
            print(f"  CANVAS  wheel  dy={ev.get('dy',0):.1f}")
        else:
            print(f"  CANVAS  {t}  {ev}")

    canvas.add_event_handler(_log,
        "key_down", "key_up",
        "pointer_down", "pointer_move", "pointer_up",
        "wheel", "resize",
    )

    def draw():
        renderer.render(scene, cam)

    canvas.request_draw(draw)

    t0 = time.time()
    while not canvas.get_closed() and time.time() - t0 < PHASE_DURATION:
        canvas._rc_gui_poll()
        canvas.force_draw()
        time.sleep(0.016)

    canvas.close()
    print(f"  → Phase 1 done: {event_count[0]} events received")
    return event_count[0]


# ===========================================================================
# Phase 2 – WgpuRenderer event forwarding
# ===========================================================================
def phase2():
    print("\n" + "="*60)
    print("PHASE 2 – WgpuRenderer event forwarding")
    print("  Same actions as Phase 1.")
    print("  You should see a RENDERER line for every CANVAS line.")
    print("="*60)

    canvas = GlfwRenderCanvas(
        title="Phase 2 – renderer events", size=(640, 480), update_mode="manual")
    renderer = pygfx.WgpuRenderer(canvas, enable_events=True)
    scene = _make_scene()
    cam = pygfx.PerspectiveCamera(50, 640/480, depth_range=(0.01, 100))
    cam.local.position = np.array([0, 0, 3], dtype=float)
    cam.look_at(np.array([0, 0, 0], dtype=float))

    canvas_count  = [0]
    renderer_count = [0]

    def _log_canvas(ev):
        canvas_count[0] += 1
        t = ev.get("event_type", "?")
        if t not in ("pointer_move",):
            print(f"  CANVAS    {t}")

    def _log_renderer(ev):
        renderer_count[0] += 1
        t = ev.get("event_type", "?")
        if t not in ("pointer_move", "before_render"):
            print(f"  RENDERER  {t}")

    canvas.add_event_handler(_log_canvas,
        "key_down", "key_up", "pointer_down", "pointer_move",
        "pointer_up", "wheel",
    )
    renderer.add_event_handler(_log_renderer,
        "key_down", "key_up", "pointer_down", "pointer_move",
        "pointer_up", "wheel", "before_render",
    )

    def draw():
        renderer.render(scene, cam)

    canvas.request_draw(draw)

    t0 = time.time()
    while not canvas.get_closed() and time.time() - t0 < PHASE_DURATION:
        canvas._rc_gui_poll()
        canvas.force_draw()
        time.sleep(0.016)

    canvas.close()
    print(f"  → Phase 2 done:  canvas={canvas_count[0]}  renderer={renderer_count[0]}")
    return canvas_count[0], renderer_count[0]


# ===========================================================================
# Phase 3 – OrbitController with explicit camera
# ===========================================================================
def phase3():
    print("\n" + "="*60)
    print("PHASE 3 – OrbitController (camera must be passed explicitly)")
    print("  Hold left-mouse and drag slowly to orbit.")
    print("  Scroll to zoom.")
    print("  You should see CONTROLLER lines + camera-position prints.")
    print("="*60)

    canvas = GlfwRenderCanvas(
        title="Phase 3 – OrbitController", size=(640, 480), update_mode="manual")
    renderer = pygfx.WgpuRenderer(canvas, enable_events=True)
    scene = _make_scene()
    cam = pygfx.PerspectiveCamera(50, 640/480, depth_range=(0.01, 100))
    cam.local.position = np.array([0, 0, 3], dtype=float)
    cam.look_at(np.array([0, 0, 0], dtype=float))

    # ── This is the critical line ──────────────────────────────────────
    # Without 'cam' as the first argument, controller.cameras == ()
    # and the camera is never updated regardless of mouse events.
    ctrl = pygfx.OrbitController(cam, target=(0, 0, 0))
    ctrl.register_events(renderer)
    print(f"  controller.cameras = {ctrl.cameras}  (must be non-empty)")

    ctrl_events = [0]

    def _log_ctrl(ev):
        ctrl_events[0] += 1
        t = ev.get("event_type", "?")
        if t not in ("pointer_move", "before_render"):
            print(f"  CONTROLLER via renderer: {t}")

    renderer.add_event_handler(_log_ctrl,
        "pointer_down", "pointer_up", "wheel", "before_render",
    )

    prev_pos = cam.local.position.copy()

    def draw():
        nonlocal prev_pos
        renderer.render(scene, cam)
        new_pos = cam.local.position.copy()
        if np.linalg.norm(new_pos - prev_pos) > 1e-5:
            print(f"  ✓ Camera moved → pos={new_pos.round(3)}")
            prev_pos = new_pos.copy()

    canvas.request_draw(draw)

    t0 = time.time()
    while not canvas.get_closed() and time.time() - t0 < PHASE_DURATION:
        canvas._rc_gui_poll()
        canvas.force_draw()
        time.sleep(0.016)

    canvas.close()
    print(f"  → Phase 3 done: {ctrl_events[0]} renderer events received")


# ===========================================================================
if __name__ == "__main__":
    n1 = phase1()
    if n1 == 0:
        print("\n⚠ Phase 1 got ZERO events – the canvas is not receiving "
              "any input from _rc_gui_poll(). This is the root cause.")
    else:
        print(f"\n✓ Phase 1 OK – {n1} events reached the canvas handler.")

    c2, r2 = phase2()
    if c2 > 0 and r2 == 0:
        print("\n⚠ Canvas got events but renderer got NONE – "
              "WgpuRenderer.enable_events() is not forwarding correctly.")
    elif c2 > 0 and r2 > 0:
        print(f"\n✓ Phase 2 OK – renderer forwarding works.")

    phase3()
    print("\nAll phases done.")


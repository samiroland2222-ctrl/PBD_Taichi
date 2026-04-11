"""pygfx / wgpu-py backend for TaichiRenderer3D.
Provides ``WgpuRenderer3D`` with the same call-site API as
``TaichiRenderer3D`` so that simulation scripts require no changes when
the ``RENDERER_BACKEND`` environment variable is set to ``wgpu`` (or when
``backend='wgpu'`` is passed to the ``TaichiRenderer3D`` constructor).
Scene proxy
-----------
Every render-draw callback receives a ``WgpuSceneProxy`` that mirrors
the ``ti.ui.Scene`` drawing API::
    def my_draw(scene):
        scene.mesh(v_p, f_i, color=(0.7, 0.3, 0.1))
        scene.lines(verts, 2.0, idx, color=(1.0, 1.0, 1.0))
        scene.particles(pts, 0.02, color=(0.0, 0.7, 0.7))
First call: pygfx objects are created and added to the pygfx scene.
Subsequent calls: only the mutable buffers (vertex positions, optional
vertex colours) are refreshed each frame.
"""
from __future__ import annotations

# ── ONE-TIME GLFW LIBRARY UNIFICATION ────────────────────────────────────────
# imgui_bundle ships its own libglfw.3.dylib; the Python `glfw` package (used
# by rendercanvas) ships a second copy.  On macOS, loading both into the same
# process causes duplicate Objective-C class definitions
# (GLFWWindow, GLFWWindowDelegate, GLFWContentView, …).  The OS picks one
# implementation at random → event callbacks on the rendercanvas window
# silently die, and the process eventually segfaults.
#
# imgui_bundle._glfw_set_search_path() sets PYGLFW_LIBRARY so that the `glfw`
# Python package reuses imgui_bundle's already-loaded library instead of its
# own.  This MUST be called before `glfw` (and therefore `rendercanvas.glfw`)
# is imported for the first time.
try:
    import imgui_bundle as _ib_early
    _ib_early._glfw_set_search_path()
    del _ib_early
except (ImportError, AttributeError):
    pass
# ─────────────────────────────────────────────────────────────────────────────

import time
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pygfx
import glfw as _glfw                                # unified libglfw (via PYGLFW_LIBRARY)
from rendercanvas.glfw import GlfwRenderCanvas  # glfw now loads via PYGLFW_LIBRARY

try:
    from imgui_bundle import imgui, hello_imgui
    _IMGUI_AVAILABLE = True

    # Rendercanvas key-name → imgui.Key mapping.
    # Used for the imgui-window-focused keyboard path (see _init_imgui).
    _IMGUI_KEY_MAP: dict = {
        ' ':          imgui.Key.space,
        'Enter':      imgui.Key.enter,
        'Escape':     imgui.Key.escape,
        'Backspace':  imgui.Key.backspace,
        'Delete':     imgui.Key.delete,
        'Tab':        getattr(imgui.Key, 'tab', None),
        'ArrowUp':    imgui.Key.up_arrow,
        'ArrowDown':  imgui.Key.down_arrow,
        'ArrowLeft':  imgui.Key.left_arrow,
        'ArrowRight': imgui.Key.right_arrow,
        'Home':       imgui.Key.home,
        'End':        imgui.Key.end,
        'Insert':     imgui.Key.insert,
        'PageUp':     imgui.Key.page_up,
        'PageDown':   imgui.Key.page_down,
        **{f'F{i}': getattr(imgui.Key, f'f{i}', None) for i in range(1, 25)},
    }

    def _to_imgui_key(key: str):
        """Map a rendercanvas key-name string to an imgui.Key value, or None."""
        # Single letter (rendercanvas emits lowercase)
        if len(key) == 1 and key.isalpha():
            return getattr(imgui.Key, key.lower(), None)
        return _IMGUI_KEY_MAP.get(key)

except ImportError:
    _IMGUI_AVAILABLE = False
    _IMGUI_KEY_MAP: dict = {}

    def _to_imgui_key(key: str):  # type: ignore[misc]
        return None

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_np_verts(field) -> np.ndarray:
    """Return a C-contiguous float32 (N, 3) array."""
    if isinstance(field, np.ndarray):
        arr = field.astype(np.float32, copy=False)
    else:
        arr = field.to_numpy().astype(np.float32)
    return np.ascontiguousarray(arr)
def _to_np_indices(field) -> np.ndarray:
    """Return a C-contiguous int32 flat 1-D array."""
    if isinstance(field, np.ndarray):
        arr = field.astype(np.int32, copy=False)
    else:
        arr = field.to_numpy().astype(np.int32)
    return np.ascontiguousarray(arr.ravel())
def _rgba(c) -> tuple:
    """Expand (r, g, b) to (r, g, b, 1.0)."""
    if len(c) == 3:
        return (float(c[0]), float(c[1]), float(c[2]), 1.0)
    return tuple(float(x) for x in c)
# ---------------------------------------------------------------------------
# WgpuSceneProxy
# ---------------------------------------------------------------------------
class WgpuSceneProxy:
    """Drop-in for ti.ui.Scene for the wgpu backend.
    Caches pygfx world-objects created on the first invocation of each
    draw-callback.  On every subsequent call only the mutable buffers
    (positions, colours) are updated.
    """
    def __init__(self, scene: pygfx.Scene, force_two_sided: bool = False):
        self._scene = scene
        self._force_two_sided = force_two_sided
        # id(taichi_field) -> (world_obj, geometry)
        self._meshes: dict = {}
        # id(taichi_field) -> (line_obj, geometry, indices_np)
        self._lines: dict = {}
        # id(taichi_field) -> (pts_obj, geometry, has_vertex_color)
        self._points: dict = {}
    # ------------------------------------------------------------------ mesh
    def mesh(self, vertices, indices, color=(0.5, 0.5, 0.5),
             show_wireframe=False, two_sided=False, **kwargs):
        verts_np = _to_np_verts(vertices)
        key = id(vertices)
        if key not in self._meshes:
            idx_np = _to_np_indices(indices).reshape(-1, 3).astype(np.int32)
            geo = pygfx.Geometry(
                positions=verts_np.copy(),
                indices=idx_np,
            )
            rgba = _rgba(color)
            mat = pygfx.MeshPhongMaterial(
                color=rgba,
                wireframe=bool(show_wireframe),
                side="both",
            )
            obj = pygfx.Mesh(geo, mat)
            self._scene.add(obj)
            self._meshes[key] = (obj, geo)
        else:
            _obj, geo = self._meshes[key]
            geo.positions.data[:] = verts_np
            geo.positions.update_range()
    # ----------------------------------------------------------------- lines
    def lines(self, vertices, width, indices, color=(0.5, 0.5, 0.5), **kwargs):
        """Draw indexed line segments.
        LineSegmentMaterial treats consecutive pairs of positions as
        segments (no explicit index support in pygfx.Line).  We gather
        positions using the indices on every frame.
        """
        verts_np = _to_np_verts(vertices)
        key = id(vertices)
        if key not in self._lines:
            idx_np = _to_np_indices(indices)
            gathered = np.ascontiguousarray(verts_np[idx_np])
            geo = pygfx.Geometry(positions=gathered)
            rgba = _rgba(color)
            mat = pygfx.LineSegmentMaterial(
                thickness=max(1.0, float(width)),
                color=rgba,
            )
            obj = pygfx.Line(geo, mat)
            self._scene.add(obj)
            self._lines[key] = (obj, geo, idx_np)
        else:
            _obj, geo, idx_np = self._lines[key]
            geo.positions.data[:] = np.ascontiguousarray(verts_np[idx_np])
            geo.positions.update_range()
    # --------------------------------------------------------------- particles
    def particles(self, vertices, radius, per_vertex_color=None,
                  color=(0.0, 0.7, 0.7), **kwargs):
        verts_np = _to_np_verts(vertices)
        key = id(vertices)
        size = float(radius) * 2.0
        if key not in self._points:
            if per_vertex_color is not None:
                col3 = _to_np_verts(per_vertex_color)
                alpha = np.ones((col3.shape[0], 1), dtype=np.float32)
                col4 = np.ascontiguousarray(np.concatenate([col3, alpha], axis=1))
                geo = pygfx.Geometry(positions=verts_np.copy(), colors=col4.copy())
                mat = pygfx.PointsGaussianBlobMaterial(
                    size=size, size_space="world", color_mode="vertex")
                has_vc = True
            else:
                rgba = _rgba(color)
                geo = pygfx.Geometry(positions=verts_np.copy())
                mat = pygfx.PointsGaussianBlobMaterial(
                    size=size, size_space="world", color=rgba)
                has_vc = False
            obj = pygfx.Points(geo, mat)
            self._scene.add(obj)
            self._points[key] = (obj, geo, has_vc)
        else:
            _obj, geo, has_vc = self._points[key]
            geo.positions.data[:] = verts_np
            geo.positions.update_range()
            if has_vc and per_vertex_color is not None:
                col3 = _to_np_verts(per_vertex_color)
                alpha = np.ones((col3.shape[0], 1), dtype=np.float32)
                geo.colors.data[:] = np.ascontiguousarray(
                    np.concatenate([col3, alpha], axis=1))
                geo.colors.update_range()
# ---------------------------------------------------------------------------
# _FakeWindow  –  keeps ``while tirender.window.running:`` working
# ---------------------------------------------------------------------------
class _FakeWindow:
    def __init__(self, renderer: "WgpuRenderer3D"):
        self._r = renderer
    @property
    def running(self) -> bool:
        return not self._r.canvas.get_closed()
# ---------------------------------------------------------------------------
# WgpuGui  –  Taichi ti.ui.Gui API  →  Dear ImGui
# ---------------------------------------------------------------------------

class WgpuGui:
    """Adapts Taichi's ``ti.ui.Gui`` call API to Dear ImGui.

    Passed to every ``gui_draw`` callback registered via
    ``add_gui_draw()``.  Provides the same method signatures as the
    Taichi GUI object so existing simulation scripts need no changes.

    Supported widgets
    -----------------
    ``text(label)``                            – static label
    ``slider_float(label, val, lo, hi)``       – returns new float value
    ``checkbox(label, val)``                   – returns new bool value
    ``button(label)``                          – returns True when clicked
    ``sub_window(name, x, y, w, h)``           – context manager (imgui window)
    """

    def text(self, label: str) -> None:
        imgui.text(str(label))

    def slider_float(self, label: str, value: float,
                     minimum: float, maximum: float) -> float:
        _changed, new_val = imgui.slider_float(
            label, float(value), float(minimum), float(maximum))
        return new_val

    def checkbox(self, label: str, value: bool) -> bool:
        _changed, new_val = imgui.checkbox(label, bool(value))
        return new_val

    def button(self, label: str) -> bool:
        return imgui.button(label)

    @contextmanager
    def sub_window(self, name: str,
                   x: float = 0, y: float = 0,
                   w: float = 0, h: float = 0):
        """Maps to ``imgui.begin`` / ``imgui.end``."""
        imgui.begin(name)
        try:
            yield self
        finally:
            imgui.end()


# ---------------------------------------------------------------------------
# WgpuRenderer3D
# ---------------------------------------------------------------------------

class WgpuRenderer3D:
    """pygfx/wgpu equivalent of TaichiRenderer3D.

    Identical call-site API; select via RENDERER_BACKEND=wgpu or
    by passing backend='wgpu' to TaichiRenderer3D().

    Camera navigation (OrbitController)
    ------------------------------------
    Left-mouse drag     orbit around lookat point
    Right-mouse drag    pan
    Scroll wheel        zoom
    p                   print camera position / lookat

    GUI panel
    ---------
    A separate Dear ImGui window ("Simulation Controls") is opened
    automatically.  All ``add_gui_draw`` callbacks are rendered there,
    followed by the built-in X-slice panel.
    """

    def __init__(self, title: str, res, fps: float,
                 cameraPos, cameraLookat, vertColor: bool = False) -> None:
        w, h = int(res[0]), int(res[1])

        # Canvas + renderer
        self.canvas = GlfwRenderCanvas(
            title=title, size=(w, h), update_mode="manual")

        # GlfwRenderCanvas sets glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API) and
        # does NOT reset it afterwards.  hello_imgui creates its own OpenGL window
        # later; if it inherits GLFW_NO_API the glfwMakeContextCurrent call fails.
        # Reset all hints to GLFW defaults here so the slate is clean.
        _glfw.default_window_hints()

        self.renderer = pygfx.WgpuRenderer(self.canvas, show_fps=False, enable_events=True)

        # Scene & lighting
        self._scene = pygfx.Scene()
        self._scene.add(pygfx.AmbientLight(intensity=0.8))
        self._point_light = pygfx.PointLight("#ffffff", intensity=3.0)
        self._scene.add(self._point_light)

        # Camera
        aspect = w / h
        self.camera = pygfx.PerspectiveCamera(
            50, aspect, depth_range=(0.001, 1000.0))
        cam_pos  = np.asarray(cameraPos,    dtype=float)
        cam_look = np.asarray(cameraLookat, dtype=float)
        self.camera.local.position = cam_pos
        self.camera.look_at(cam_look)
        self._scene.add(self.camera)

        # ── OrbitController ───────────────────────────────────────────
        # Must pass the camera explicitly – without it, controller.cameras
        # is an empty tuple and all pointer/scroll events are silently dropped.
        # Left-drag: orbit  |  Right-drag: pan  |  Scroll: zoom
        self.controller = pygfx.OrbitController(
            self.camera, target=cam_look.tolist())
        self.controller.register_events(self.renderer)

        # Register draw callback
        self.canvas.request_draw(self._draw_cb)

        # Compatibility shim: ``while tirender.window.running:``
        self.window = _FakeWindow(self)

        # Frame timing
        self.fps       = fps
        self.frame_dt  = 1.0 / fps
        self.prev_time = time.time()
        self.frame     = 0
        self.time      = 0.0

        # Draw lists
        self.scene_render_list: list[Callable] = []
        self.gui_list:          list[Callable] = []
        self.keyboard_input:    dict[str, Callable] = {}

        # X-slice
        self.clip_plane_enabled = False
        self.clip_x             = 0.0
        self._z_near_default    = 0.001

        # Persistent proxy (owns per-object pygfx caches)
        self._proxy = WgpuSceneProxy(self._scene)

        # Camera-change tracking (print on move, like Taichi renderer)
        self._cam_pos_prev = cam_pos.copy()

        # ── WASD / Q / C fly-navigation ───────────────────────────────
        # Keys currently held down (lowercase). Populated by the canvas
        # key_down / key_up handlers below; consumed each frame inside
        # _apply_keyboard_movement().
        self._keys_down: set[str] = set()
        self._move_speed: float = 1.0   # world-units per second

        def _on_key_down(event):
            k = event.get("key", "")
            if k:
                self._keys_down.add(k.lower())
            # 'p' → print camera info (canvas-window focus path)
            if k == "p":
                self._print_camera()

        def _on_key_up(event):
            k = event.get("key", "")
            if k:
                self._keys_down.discard(k.lower())

        self.canvas.add_event_handler(_on_key_down, "key_down")
        self.canvas.add_event_handler(_on_key_up, "key_up")

        # ── imgui GUI panel ───────────────────────────────────────────
        self._imgui_initialized = False
        if _IMGUI_AVAILABLE:
            self._init_imgui(fps)


    # ------------------------------------------------------------------ imgui

    def _init_imgui(self, fps: float) -> None:
        """Initialise hello_imgui ManualRender in a side-panel window."""
        _gui = WgpuGui()

        def _imgui_frame() -> None:
            # ── Keyboard shortcuts (hello_imgui-window focus path) ────────
            # GLFW only fires key callbacks on the OS-focused window.
            # After startup the hello_imgui window holds keyboard focus until
            # the user clicks the 3-D viewport.  Check key presses via
            # imgui's own IO so shortcuts fire regardless of which window
            # happens to be focused.  The canvas.add_event_handler path
            # covers the same shortcuts when the rendercanvas window is focused.
            #
            # NOTE: do NOT gate on io.want_capture_keyboard here.
            # hello_imgui enables ImGuiConfigFlags_NavEnableKeyboard by default,
            # which sets want_capture_keyboard=True even when no widget is
            # editing text, silently blocking every shortcut.  Use
            # imgui.is_any_item_active() instead – it is True only when a
            # text-input (or similar) widget actually has keyboard focus.
            if not imgui.is_any_item_active():
                # 'p' → print camera info
                if imgui.is_key_pressed(imgui.Key.p, False):
                    self._print_camera()
                # All add_click_event() registered shortcuts
                for _ks, _fn in list(self.keyboard_input.items()):
                    _ik = _to_imgui_key(_ks)
                    if _ik is not None and imgui.is_key_pressed(_ik, False):
                        _fn()

            # ── User GUI callbacks ────────────────────────────────────
            if self.gui_list:
                imgui.set_next_window_pos(
                    (0, 0), imgui.Cond_.first_use_ever)
                imgui.set_next_window_size(
                    (370, 700), imgui.Cond_.first_use_ever)
                imgui.begin("Controls")
                for fn in self.gui_list:
                    fn(_gui)
                imgui.end()

            # ── X-slice panel (always visible) ────────────────────────
            imgui.set_next_window_pos(
                (0, 710), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(
                (370, 130), imgui.Cond_.first_use_ever)
            imgui.begin("X-Slice")
            _c, self.clip_plane_enabled = imgui.checkbox(
                "Enable X-slice", self.clip_plane_enabled)
            if self.clip_plane_enabled:
                _c, self.clip_x = imgui.slider_float(
                    "clip_x", self.clip_x, -0.5, 0.5)
                imgui.text(f"  near clip @ x = {self.clip_x:.3f} m")
            else:
                imgui.text("  (disabled – near clip = default)")
            imgui.end()

        hello_imgui.manual_render.setup_from_gui_function(
            _imgui_frame,
            window_title="Simulation Controls",
            window_size_auto=False,
            window_size=[380, 860],
            fps_idle=fps,
        )
        self._imgui_initialized = True

    # ------------------------------------------------------------------ helpers

    def _print_camera(self) -> None:
        pos    = self.camera.world.position
        target = np.asarray(self.controller.target, dtype=float)
        print(f"Camera position : {pos}")
        print(f"Camera lookat   : {target}")

    def _apply_keyboard_movement(self) -> None:
        """Translate camera + orbit target with WASD / E / C each frame.

        Key bindings
        ------------
        W / S   move forward / backward along the horizontal view direction
        A / D   strafe left / right
        E / Q   move up / down along world-Y
        """
        _NAV = {'w', 'a', 's', 'd', 'e', 'q'}

        # --- collect active keys from both input sources ----------------
        # Source 1: canvas key_down/key_up events (rendercanvas window focused)
        active: set[str] = self._keys_down & _NAV

        # Source 2: imgui.is_key_down() (imgui window focused)
        if _IMGUI_AVAILABLE and self._imgui_initialized and not imgui.is_any_item_active():
            for k in _NAV:
                ik = _to_imgui_key(k)
                if ik is not None and imgui.is_key_down(ik):
                    active.add(k)

        if not active:
            return

        speed = self._move_speed * self.frame_dt

        cam_pos = np.array(self.camera.world.position, dtype=float)
        target  = np.array(self.controller.target,     dtype=float)

        # Forward = horizontal direction from camera toward target
        fwd = target - cam_pos
        fwd[1] = 0.0
        fwd_len = np.linalg.norm(fwd)
        if fwd_len > 1e-8:
            fwd /= fwd_len

        world_up = np.array([0.0, 1.0, 0.0], dtype=float)

        # Right = cross(forward, world_up)
        right = np.cross(fwd, world_up)
        right_len = np.linalg.norm(right)
        if right_len > 1e-8:
            right /= right_len

        delta = np.zeros(3, dtype=float)
        if 'w' in active: delta += fwd      * speed
        if 's' in active: delta -= fwd      * speed
        if 'd' in active: delta += right    * speed
        if 'a' in active: delta -= right    * speed
        if 'e' in active: delta += world_up * speed
        if 'q' in active: delta -= world_up * speed

        if np.linalg.norm(delta) < 1e-10:
            return

        self.camera.local.position = cam_pos + delta
        self.controller.target = (target + delta).tolist()

    # ------------------------------------------------------------------ draw

    def _draw_cb(self) -> None:
        """Registered with the canvas; invoked by canvas.force_draw()."""
        self._point_light.local.position = self.camera.world.position.copy()
        self._proxy._force_two_sided = self.clip_plane_enabled
        for draw_fn in self.scene_render_list:
            draw_fn(self._proxy)
        self.renderer.render(self._scene, self.camera)

    # ------------------------------------------------------------------ API

    def add_click_event(self, key: str, func: Callable) -> None:
        self.keyboard_input[key] = func

        def _on_key(event):
            if event.get("key") == key:
                func()

        self.canvas.add_event_handler(_on_key, "key_down")

    def add_gui_draw(self, gui_draw_call: Callable) -> None:
        self.gui_list.append(gui_draw_call)

    def clear_scene_render_draw(self) -> None:
        self.scene_render_list.clear()

    def add_scene_render_draw(self, scene_render_draw_call: Callable) -> None:
        self.scene_render_list.append(scene_render_draw_call)

    def handle_input(self) -> None:
        """No-op: events are handled via the canvas event system."""

    def render(self) -> None:
        """Pump events, render one frame (3-D + GUI), pace to target FPS."""
        # ── imgui event-capture guard ─────────────────────────────────
        # Disable the OrbitController only while Dear ImGui has an active
        # text-input widget (is_any_item_active).  Keyboard-navigation mode
        # sets want_capture_keyboard=True even with no text field, so we
        # must NOT use that flag here.
        if _IMGUI_AVAILABLE and self._imgui_initialized:
            self.controller.enabled = not imgui.is_any_item_active()

        # ── Camera-change detection ───────────────────────────────────
        new_pos = self.camera.world.position.copy()
        if np.linalg.norm(new_pos - self._cam_pos_prev) > 1e-5:
            self._print_camera()
            self._cam_pos_prev = new_pos

        # ── 3-D viewport ─────────────────────────────────────────────
        # _process_events() = _rc_gui_poll() + _events.flush()
        # The flush step is critical: it dispatches the queued pointer/key
        # events to the pygfx renderer and on to the OrbitController.
        # Calling only _rc_gui_poll() fills the queue but never empties it,
        # so the OrbitController (and all canvas event handlers) stay deaf.
        self.canvas._process_events()  # pump GLFW events AND flush to subscribers
        self._apply_keyboard_movement() # translate camera for held WASD/E/C keys
        self.canvas.force_draw()        # draw + present

        # ── imgui GUI panel ───────────────────────────────────────────
        if self._imgui_initialized:
            hello_imgui.manual_render.render()

        # ── Frame pacing ──────────────────────────────────────────────
        spent = time.time() - self.prev_time
        if spent < self.frame_dt:
            time.sleep(self.frame_dt - spent)
        self.prev_time = time.time()
        self.frame += 1
        self.time = self.frame / self.fps

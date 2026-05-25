import os
import taichi as ti
from pxr import Usd, UsdGeom
from geom import gmesh
import numpy as np
import time

# ---------------------------------------------------------------------------
# Backend routing
# ---------------------------------------------------------------------------
# Select the rendering backend via:
#   • environment variable:  RENDERER_BACKEND=wgpu
#   • constructor argument:  TaichiRenderer3D(..., backend='wgpu')
# The default remains the Taichi UI backend ('taichi').
_ENV_BACKEND = os.environ.get("RENDERER_BACKEND", "taichi").lower()


def _make_renderer(title, res, fps, cameraPos, cameraLookat,
                   vertColor=False, backend=None):
    """Factory that returns the appropriate renderer instance.

    Parameters
    ----------
    backend : str | None
        ``'wgpu'`` – use the pygfx/wgpu backend (``WgpuRenderer3D``).
        ``'taichi'`` or ``None`` – use the Taichi UI backend (default).
        If ``None`` the ``RENDERER_BACKEND`` environment variable is
        consulted, falling back to ``'taichi'``.
    """
    chosen = (backend or _ENV_BACKEND).lower()
    if chosen == "wgpu":
        from utils.renderer_wgpu import WgpuRenderer3D
        return WgpuRenderer3D(title, res, fps, cameraPos, cameraLookat,
                              vertColor=vertColor)
    # Taichi UI backend constructed inline (see TaichiRenderer3D below)
    return None  # sentinel: caller builds Taichi renderer as normal


class _SceneProxy:
    """Thin wrapper around a ``ti.ui.Scene`` that can force ``two_sided=True``
    on every ``mesh()`` call.  All other attribute accesses are forwarded
    transparently to the real scene object.

    Used by ``TaichiRenderer3D`` to make the X-slice cross-section visible:
    when the near clip plane cuts through geometry the back-faces are exposed,
    so they must be lit to see the interior layers.
    """

    def __init__(self, scene, force_two_sided: bool = False):
        # Store on __dict__ directly to avoid triggering our own __setattr__
        object.__setattr__(self, '_scene', scene)
        object.__setattr__(self, '_force_two_sided', force_two_sided)

    def mesh(self, vertices, indices, *args, **kwargs):
        if object.__getattribute__(self, '_force_two_sided'):
            kwargs['two_sided'] = True
        return object.__getattribute__(self, '_scene').mesh(
            vertices, indices, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_scene'), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_scene'), name, value)


class TaichiRenderer3D:

  def __new__(cls, title: str, res, fps, cameraPos, cameraLookat,
              vertColor=False, backend=None):
    """Return a WgpuRenderer3D when the wgpu backend is selected."""
    chosen = (backend or _ENV_BACKEND).lower()
    if chosen == "wgpu":
      from utils.renderer_wgpu import WgpuRenderer3D
      return WgpuRenderer3D(title, res, fps, cameraPos, cameraLookat,
                            vertColor=vertColor)
    return super().__new__(cls)

  def __init__(self,
               title: str,
               res,
               fps,
               cameraPos,
               cameraLookat,
               vertColor=False,
               backend=None) -> None:
    # Skip Taichi UI initialisation when the wgpu backend was chosen
    # (__new__ already returned a WgpuRenderer3D in that case).
    if not isinstance(self, TaichiRenderer3D):
      return
    self.window = ti.ui.Window(title, res)
    self.gui = self.window.get_gui()
    self.canvas = self.window.get_canvas()
    self.scene = ti.ui.Scene()
    self.camera = ti.ui.Camera()
    self.camera.position(cameraPos[0], cameraPos[1], cameraPos[2])
    self.camera.lookat(cameraLookat[0], cameraLookat[1], cameraLookat[2])

    self.frame = 0
    self.time = 0.0
    self.prev_time = time.time()
    self.fps = fps
    self.frame_dt = 1.0 / fps
    self.vertColor = vertColor
    self.keyboard_input = {}
    self.gui_list = []
    self.scene_render_list = []

    # ── Topological X-slice ────────────────────────────────────────────
    # When enabled, the near clip plane is recomputed every frame so that
    # it passes through (clip_x, 0, 0) in world space, letting you see
    # a cross-section through all layers at that X position.
    self.clip_plane_enabled = False
    self.clip_x             = 0.0
    self._z_near_default    = 0.001   # metres – used when slice is off

    # ── Surface-normals overlay ─────────────────────────────────────────
    # Taichi UI has no efficient line-drawing for CPU arrays; the toggle
    # is wired to the GUI but make_normals_draw_callback() returns a no-op.
    self.show_surface_normals  = False
    self.surface_normals_scale = 0.01

    def print_camera_info():
      print("Camera position: ", self.camera.curr_position)
      print("Camera look at: ", self.camera.curr_lookat)

    self.add_click_event('p', print_camera_info)

  def add_click_event(self, key: str, func):
    self.keyboard_input[key] = func

  def add_gui_draw(self, gui_draw_call):
    self.gui_list.append(gui_draw_call)

  def clear_scene_rendeer_draw(self):
    self.scene_render_list.clear()

  def add_scene_render_draw(self, scene_render_draw_call):
    self.scene_render_list.append(scene_render_draw_call)

  # ── X-slice clip plane ─────────────────────────────────────────────────────
  def _update_clip_plane(self):
    """Adjust the near clip plane every frame.

    When ``clip_plane_enabled`` is True the near clip plane is moved so it
    passes through ``(clip_x, 0, 0)`` in world space along the current view
    direction, creating a topological cross-section through all layers.
    When disabled the near clip plane is reset to ``_z_near_default``.
    """
    if not self.clip_plane_enabled:
      self.camera.z_near(self._z_near_default)
      return

    cam_pos  = np.array(self.camera.curr_position, dtype=np.float64)
    cam_look = np.array(self.camera.curr_lookat,   dtype=np.float64)
    view_dir = cam_look - cam_pos
    norm = float(np.linalg.norm(view_dir))
    if norm < 1e-8:
      self.camera.z_near(self._z_near_default)
      return
    view_dir /= norm

    # Signed distance from the camera to the world point (clip_x, 0, 0)
    # projected onto the view direction.
    world_pt = np.array([self.clip_x, 0.0, 0.0], dtype=np.float64)
    z = float(np.dot(world_pt - cam_pos, view_dir))
    # z_near must be strictly positive; if the slice point is behind the
    # camera we fall back to the default (nothing gets clipped).
    self.camera.z_near(max(z, self._z_near_default))

  def handle_input(self):
    if self.window.get_event(ti.ui.PRESS):
      if self.window.event.key in self.keyboard_input:
        self.keyboard_input[self.window.event.key]()

  def make_normals_draw_callback(self, get_verts, get_faces):
    """Stub: Taichi UI cannot efficiently draw CPU-side line arrays.

    The wgpu backend (``WgpuRenderer3D``) provides a full implementation.
    This version returns a no-op so call-sites work on both backends.
    """
    def _noop(scene):
      pass
    return _noop

  def render(self):
    old_cam_pos = self.camera.curr_position
    old_cam_lookat = self.camera.curr_lookat
    old_cam_up = self.camera.curr_up
    self.camera.track_user_inputs(self.window,
                                  movement_speed=0.01,
                                  hold_key=ti.ui.RMB)
    if (old_cam_pos - self.camera.curr_position).norm() > 0.00001 or (old_cam_lookat - self.camera.curr_lookat).norm() > 0.00001 or (old_cam_up - self.camera.curr_up).norm() > 0.00001:
        print("Camera position: ", self.camera.curr_position)
        print("Camera look at: ", self.camera.curr_lookat)
        print("Camera up: ", self.camera.curr_up)
    self._update_clip_plane()
    self.scene.set_camera(self.camera)
    self.scene.ambient_light((0.8, 0.8, 0.8))
    self.scene.point_light(pos=self.camera.curr_position, color=(1, 1, 1))
    self.scene.ambient_light([0.2, 0.2, 0.2])

    for scene_render_draw in self.scene_render_list:
      scene_render_draw(_SceneProxy(self.scene, self.clip_plane_enabled))

    self.canvas.scene(self.scene)

    if len(self.gui_list) > 0:
      with self.gui.sub_window('gui', 0.0, 0.0, 0.3, 0.4):
        for gui_draw in self.gui_list:
          gui_draw(self.gui)

    # ── Overlays panel (X-slice + surface normals) ─────────────────────
    with self.gui.sub_window('Overlays', 0.0, 0.41, 0.3, 0.20):
      self.gui.text("── X-Slice ──")
      self.clip_plane_enabled = self.gui.checkbox(
          "Enable X-slice", self.clip_plane_enabled)
      if self.clip_plane_enabled:
        self.clip_x = self.gui.slider_float(
            "clip_x", self.clip_x, -0.5, 0.5)
        self.gui.text(f"  near clip @ x = {self.clip_x:.3f} m")
      else:
        self.gui.text("  (disabled – near clip = default)")

      self.gui.text("── Surface Normals ──")
      self.show_surface_normals = self.gui.checkbox(
          "Show surface normals", self.show_surface_normals)
      if self.show_surface_normals:
        self.surface_normals_scale = self.gui.slider_float(
            "Normal scale", self.surface_normals_scale, 0.001, 0.1)
        self.gui.text(f"  (not supported in Taichi UI backend)")
      else:
        self.gui.text("  (disabled)")

    self.window.show()

    spent_time = time.time() - self.prev_time
    if spent_time < self.frame_dt:
      time.sleep(self.frame_dt - spent_time)
    self.prev_time = time.time()

    self.frame += 1
    self.time = self.frame / self.fps


class TaichiRender2D:

  def __init__(self, title: str, res, fps) -> None:
    self.window = ti.ui.Window(name=title, res=res)
    self.gui = self.window.get_gui()
    self.canvas = self.window.get_canvas()
    self.frame = 0
    self.time = 0.0
    self.prev_time = time.time()
    self.fps = fps
    self.frame_dt = 1.0 / fps
    self.keyboard_input = {}
    self.gui_list = []
    self.canvas_render_list = []

  def add_canvas_render_draw(self, canvas_render_draw_call):
    self.canvas_render_list.append(canvas_render_draw_call)

  def render(self):

    for canvas_render_draw in self.canvas_render_list:
      canvas_render_draw(self.canvas)

    if len(self.gui_list) > 0:
      with self.gui.sub_window('gui', 0.0, 0.0, 0.3, 0.4):
        for gui_draw in self.gui_list:
          gui_draw(self.gui)

    self.window.show()

    spent_time = time.time() - self.prev_time
    if spent_time < self.frame_dt:
      time.sleep(self.frame_dt - spent_time)
    self.prev_time = time.time()

    self.frame += 1
    self.time = self.frame / self.fps


class USDMeshRenderer:

  def __init__(self, filepath, totalframes, fps) -> None:
    self.stage = Usd.Stage.CreateNew(filepath)
    UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
    self.stage.SetStartTimeCode(1)
    self.stage.SetEndTimeCode(totalframes)
    self.stage.SetTimeCodesPerSecond(fps)

    self.totalframes = totalframes
    self.fps = fps
    self.frame = 0
    self.time = 0.0
    self.mesh_prims = []
    self.mesh_verts = []

    self.rootXform = UsdGeom.Xform.Define(self.stage, '/root')

  def render(self):
    if self.frame >= self.totalframes:
      return
    for i in range(len(self.mesh_prims)):
      meshGeom = UsdGeom.Mesh(self.stage.GetPrimAtPath(self.mesh_prims[i]))
      meshGeom.GetPointsAttr().Set(value=self.mesh_verts[i].to_numpy(),
                                   time=self.frame + 1)
    self.frame += 1
    self.time = self.frame / self.fps

  def add_ground(self, size):
    groundGeom = UsdGeom.Mesh.Define(self.stage, '/root/ground')
    groundPoints = np.array([[-size[0] / 2.0, 0.0, -size[1] / 2.0],
                             [-size[0] / 2.0, 0.0, size[1] / 2.0],
                             [size[0] / 2.0, 0.0, size[1] / 2.0],
                             [size[0] / 2.0, 0.0, -size[1] / 2.0]])
    groundGeom.GetPointsAttr().Set(groundPoints)
    groundGeom.GetFaceVertexIndicesAttr().Set(np.array([0, 1, 2, 0, 2, 3]))
    groundGeom.GetFaceVertexCountsAttr().Set(np.array([3, 3]))

  def add_dynamic_mesh(self,
                       vert: ti.MatrixField,
                       face: ti.MatrixField,
                       meshname='mesh'):
    primpath = '/root/' + meshname
    self.mesh_prims.append(primpath)
    self.mesh_verts.append(vert)
    self.meshGeom = UsdGeom.Mesh.Define(self.stage, primpath)
    self.meshGeom.GetPointsAttr().Set(vert.to_numpy())
    self.meshGeom.GetFaceVertexIndicesAttr().Set(face.to_numpy())
    self.meshGeom.GetFaceVertexCountsAttr().Set(vert.shape[0] * [3])
    self.meshGeom.GetSubdivisionSchemeAttr().Set('none')

  def save(self):
    self.stage.Save()


def get_visibility_switch(item):

  def visibility_switch():
    item.visible = not item.visible

  return visibility_switch


def get_wireframe_switch(item):

  def wireframe_switch():
    item.wireframe = not item.wireframe

  return wireframe_switch
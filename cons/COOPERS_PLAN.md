# Implementation Plan: Restructuring Cooper's Ligaments and Clavipectoral Fascia

## Overview
Currently, Cooper's ligaments are anchored directly to the clavicle bone. This plan aims to improve the biomechanical accuracy of the model by introducing the clavipectoral fascia as an intermediate deformable surface, anchoring the ligaments to this fascia instead, and providing a foundation for more realistic dendritic branching of the ligaments.

## Phase 1: Ribcage Visual Geometry
**Goal:** Add visual geometry for the ribcage to serve as a visual and eventual collision/attachment reference, following a reasonably accurate biological template.

**Tasks:**
1. **Locate or Generate Ribcage Data:**
   - In `geom/anatomy.py`, add a procedural generation function for a simplified ribcage (similar to `_bone_capsule_verts_local`).
2. **Integrate into Skeleton Proxy:**
   - Update the `Skeleton` class in `anatomy.py` to include the ribcage matrix and vertices.
   - The ribcage will be statically attached to the `chest` root.
3. **Rendering:**
   - Add a method to visualize the new ribcage geometry in the main rendering loop (likely in the main simulation scripts like `test/deform_3d.py`).
4. **Unit Tests:**
   - Write a test in a new file (e.g., `test/test_anatomy.py` or existing test) ensuring the ribcage vertices are generated and correctly transformed to the world space of the chest.

## Phase 2: Clavipectoral Fascia Deformable Surface
**Goal:** Model the left and right clavipectoral fascia separately as deformable quadrilateral surfaces.

**Tasks:**
1. **Model the Quad Surfaces:**
   - Create a new mesh generation function in `geom/anatomy.py` for a 2D grid/quad sheet representing the fascia.
   - The top edge corresponds to the clavicle length.
   - The bottom edge runs horizontally across the chest surface under the breast.
2. **Kinematic Binding:**
   - **Top Edge:** Bind the top edge vertices to the clavicle's coordinate frame so they track clavicle pitch/yaw exactly like the current clavicle surface anchors do.
   - **Bottom Edge:** Keep the bottom edge fixed to the chest/ribcage root frame.
   - **Interior Vertices:** Implement a simple interpolation (e.g., linear blend skinning - LBS) for the interior vertices between the top and bottom edges so the fascia cleanly deforms as the clavicle moves. Since we have an existing `lbs.py`, investigate if it can be reused, else implement a simple distance-based blend in `Skeleton.update()`.
3. **Expose Anchors:**
   - Add new methods to the `Skeleton`: `get_fascia_left_surface_anchors_np()` and `get_fascia_right_surface_anchors_np()`. These will replace the clavicle anchor getters.
4. **Integration Tests:**
   - Test that moving the clavicle (pitch/yaw) updates the fascia interior vertices correctly without dislocating the bottom edge.

## Phase 3: Re-anchor Cooper's Ligaments
**Goal:** Anchor the Cooper's ligaments to the clavipectoral fascia surface rather than the clavicle bone.

**Tasks:**
1. **Update `coopers.py` Initialization:**
   - In `build_coopers()`, change the source of the anchors from `get_clavicle_*_surface_anchors_np()` to the new `get_fascia_*_surface_anchors_np()`.
2. **Adjust Spacing and Attachment Logic:**
   - Ensure `max_attach_dist` and the anchor selection logic still work cleanly with the new fascia sheet, which is much closer to the breast mesh than the clavicle was.
   - Verify that the normal orientation or surface sampling of the fascia quad provides evenly spread anchors.
3. **Update Anchor Pos Kernel:**
   - In `CoopersLigaments.update_anchors()`, make sure it consumes the dynamically updated fascia points from the Skeleton each frame.
4. **Visualisation Check:**
   - Ensure the rendered ligaments (`get_render_draw()`) now visually start from the new fascia sheet.
5. **Integration Tests:**
   - Run existing breast simulation scripts (e.g. `deform_3d.py`) and visually verify the breast deforms naturally with the new anchor points when the clavicle moves.

## Phase 4: Dendritic Branching of Ligaments (Planned)
**Goal:** Model the Cooper's ligaments with dendritic branching (fanning out following the internal edges of the tetrahedra).

**Future Tasks:**
- Modify `build_coopers()` to select fewer starting anchors on the fascia.
- Instead of direct distance springs to the surface, traverse the breast tetrahedral mesh (`f_i` or `elem`) from the deep surface near the fascia towards the outer skin.
- Create branching spring constraints along the internal edges of these selected tetrahedra.
- Update the XPBD solver constraints and structure to handle a multi-segment network instead of a single long spring.


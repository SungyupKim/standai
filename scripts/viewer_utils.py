"""Small helpers shared by the MuJoCo viewers."""
import mujoco


def hide_opponent(model, viewer=None):
    """The fighting opponent ('b', blue) of the shared combat model is parked at x = 6 m and just crouches; none of the get-up policies use it. Move its geoms into
    hidden visualization group 5 (and make them transparent so they cast no shadow); with a live viewer, also switch that group off in its options."""
    for g in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "").startswith("b_"):
            model.geom_group[g] = 5
            model.geom_rgba[g, 3] = 0.0
    if viewer is not None:
        viewer.opt.geomgroup[5] = 0

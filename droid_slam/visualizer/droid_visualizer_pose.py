import moderngl
import moderngl_window
import numpy as np
import torch
from align import align_pose_fragements
from lietorch import SE3
from moderngl_window.opengl.vao import VAO

from .camera import OrbitDragCameraWindow

CAM_POINTS = 0.05 * np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
).astype("f4")

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)

CAM_SEGMENTS = []
for i, j in CAM_LINES:
    CAM_SEGMENTS.append(CAM_POINTS[i])
    CAM_SEGMENTS.append(CAM_POINTS[j])
CAM_SEGMENTS = np.stack(CAM_SEGMENTS, axis=0)


def merge_depths_and_poses(depth_video1, depth_video2):
    t1 = depth_video1.counter.value
    t2 = depth_video2.counter.value

    poses1 = depth_video1.poses[:max(t1, t2)].clone()
    poses2 = depth_video2.poses[:max(t1, t2)].clone()

    if t2 <= 0:
        return poses1
    if t2 >= t1:
        return poses2

    dP, s = align_pose_fragements(
        poses1[max(0, t2 - 16): t2],
        poses2[max(0, t2 - 16): t2],
    )

    poses1[..., :3] *= s
    poses2[t2:] = (dP * SE3(poses1[t2:])).data

    return poses2


class DroidVisualizer(OrbitDragCameraWindow):
    title = "DroidVisualizer"
    _depth_video1 = None
    _depth_video2 = None

    _refresh_rate = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = False

        self.cam_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_cam;
                void main() {
                    gl_Position = m_proj * m_cam * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                uniform vec3 color;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )

        self.cam_prog["color"].value = (0, 0, 0)

        n = len(self._depth_video1.poses)
        cam_segments = np.tile(CAM_SEGMENTS, (n, 1)).astype("f4")
        self.cam_buffer = self.ctx.buffer(cam_segments.tobytes())

        self.cams = self.ctx.vertex_array(
            self.cam_prog,
            [(self.cam_buffer, "3f", "in_position")],
        )

        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.75
        self.camera.zoom = 1.0

        self.count = 0

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.cam_prog["m_proj"].write(self.camera.projection.matrix)
        self.cam_prog["m_cam"].write(self.camera.matrix)

        t = self._depth_video1.counter.value
        if t > 12 and self.count % self._refresh_rate == 0:
            if self._depth_video2 is not None:
                poses = merge_depths_and_poses(self._depth_video1, self._depth_video2)
                poses = poses[:t]
            else:
                poses = self._depth_video1.poses[:t]

            cam_pts = torch.from_numpy(CAM_SEGMENTS).cuda()
            cam_pts = SE3(poses[:, None]).inv() * cam_pts[None]
            cam_pts = cam_pts.reshape(-1, 3).cpu().numpy()

            self.cam_buffer.write(cam_pts)

        self.count += 1
        self.cams.render(mode=moderngl.LINES)


def visualization_fn(depth_video1, depth_video2=None):
    config = DroidPoseVisualizer
    config._depth_video1 = depth_video1
    config._depth_video2 = depth_video2
    moderngl_window.run_window_config(config, args=["-r", "True"])

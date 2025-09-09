import lietorch
import numpy as np
import torch
from collections import OrderedDict
from torch.multiprocessing import Process
import time
from depth_video import DepthVideo
from droid_backend import DroidBackend
from droid_frontend import DroidFrontend
from droid_net import DroidNet
from motion_filter import MotionFilter
from trajectory_filler import PoseTrajectoryFiller
from droid_slam.ros.trajectory_publisher import TrajectoryPublisher
import rospy

class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)

        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # ROS2 integration
        if getattr(args, "ros", False):
            rospy.init_node('droid_publisher_node', anonymous=True)
            self.ros_enabled = True
            self.pose_publisher = TrajectoryPublisher()  # supports pose+path
        else:
            self.ros_enabled = False

        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(target=visualization_fn, args=(self.video, None))
            self.visualizer.start()

        # from visualizer.droid_visualizer_pose import visualization_fn
        # self.visualizer = Process(target=visualization_fn, args=(self.video, None))
        # self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

    def load_weights(self, weights):
        """ load trained model weights """

        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

            # if self.ros_enabled:
            #     self.pose_publisher.publish_latest()

    def terminate(self, stream=None):

        """Terminate the visualization process and return trajectory estimates."""
        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        traj_est =camera_trajectory.inv().data.cpu().numpy()

        if self.ros_enabled:
            for T in traj_est:
                self.pose_publisher.publish_poses(T)
            rospy.sleep(0.001)

        if self.ros_enabled:
            rospy.signal_shutdown('Finished')
        return traj_est

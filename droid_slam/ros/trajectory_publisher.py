import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import copy


class TrajectoryPublisher:
    def __init__(self):
        # self.video = video
        self.pose_pub = rospy.Publisher('/droid/pose', PoseStamped, queue_size=100)
        self.path_pub = rospy.Publisher('/droid/trajectory', Path, queue_size=100)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

    def publish_poses(self, pose7d):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()

        msg.pose.position.x = float(pose7d[0])
        msg.pose.position.y = float(pose7d[1])
        msg.pose.position.z = float(pose7d[2])

        msg.pose.orientation.x = float(pose7d[3])
        msg.pose.orientation.y = float(pose7d[4])
        msg.pose.orientation.z = float(pose7d[5])
        msg.pose.orientation.w = float(pose7d[6])

        self.pose_pub.publish(msg)

        # Deep copy before appending to avoid pointer overwrite issue
        self.path_msg.header.stamp = msg.header.stamp
        self.path_msg.poses.append(msg)
        self.path_pub.publish(self.path_msg)

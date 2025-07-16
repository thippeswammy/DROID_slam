import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class TrajectoryPublisher:
    def __init__(self, video):
        self.video = video
        self.pose_pub = rospy.Publisher('/droid/pose', PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher('/droid/path', Path, queue_size=10)
        self.path_msg = Path()
        self.video = video
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

    def publish_latest(self, tstamp):
        if self.video.counter.value == 0:
            return

        pose7d = self.video.poses[self.video.counter.value - 1].cpu().numpy()

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'

        msg.pose.position.x = float(pose7d[0])
        msg.pose.position.y = float(pose7d[1])
        msg.pose.position.z = float(pose7d[2])

        msg.pose.orientation.x = float(pose7d[3])
        msg.pose.orientation.y = float(pose7d[4])
        msg.pose.orientation.z = float(pose7d[5])
        msg.pose.orientation.w = float(pose7d[6])

        self.pose_pub.publish(msg)
        self.path_msg.header.stamp = msg.header.stamp
        self.path_msg.poses.append(msg)
        self.path_pub.publish(self.path_msg)

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class PosePublisher(Node):
    def __init__(self, video):
        super().__init__('droid_pose_publisher')
        self.pose_pub = self.create_publisher(PoseStamped, '/droid/pose', 10)
        self.video = video

    def publish_pose(self, tstamp):
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

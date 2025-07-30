import torch

class CudaTimer:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()
        
    def __exit__(self, type, value, traceback):
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed = self.start.elapsed_time(self.end)
            print(self.name, elapsed)
'''project worksapce is '/home/sdv_edge2/ws_slam/DROID-SLAM'
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/align.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/cuda_timer.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/depth_video.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/droid.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/droid_async.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/droid_backend.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/droid_frontend.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/droid_net.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/factor_graph.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/logger.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/motion_filter.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/trajectory_filler.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/visualization.py
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/visualizer
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/ros
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/modules
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/geom
/home/sdv_edge2/ws_slam/DROID-SLAM/droid_slam/data_readers
/home/sdv_edge2/ws_slam/DROID-SLAM/demo.py

i runs (.venv) sdv_edge2@sdv-edge2:~/ws_slam/DROID-SLAM$ python demo.py --ros
'''
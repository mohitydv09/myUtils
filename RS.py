import cv2
import time
import threading
import numpy as np
import pyrealsense2 as rs

class RealSenseCamera:
    def __init__(self, visualization=False):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.visualization = visualization
        self.intrinsics = None  
        self.depth_scale = None
        self.rgbd_frame = None
        self.color_frame = None
        self.depth_frame = None
        self.aling = None
        self.imu_data = None
        self.running = False
        self.thread = threading.Thread(target=self._update_frames, daemon=True)

    def start(self):
        """Start the camera and the thread."""
        self.running = True
        self.pipeline.start(self.config)
        self.aling = rs.align(rs.stream.color)
        
        rs_intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = {
            "width": rs_intrinsics.width,
            "height": rs_intrinsics.height,
            "ppx": rs_intrinsics.ppx,
            "ppy": rs_intrinsics.ppy,
            "fx": rs_intrinsics.fx,
            "fy": rs_intrinsics.fy,
            "coeffs": rs_intrinsics.coeffs
        }
        self.depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        self.thread.start()

    def stop(self):
        """Stop the camera and the thread."""
        self.running = False
        self.thread.join()
        self.pipeline.stop()

    def _update_frames(self):
        """Threaded function to update frames continuously."""
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.aling.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if color_frame and depth_frame:
                # Convert frames to numpy arrays
                self.color_frame = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                self.depth_frame = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)

                self.rgbd_frame = np.dstack((self.color_frame, self.depth_frame))

                # Show the live feed
                if self.visualization:
                    images = np.hstack((self.color_frame, cv2.applyColorMap(cv2.convertScaleAbs(self.depth_frame, alpha=0.03), cv2.COLORMAP_JET)))
                    cv2.imshow("RealSense - Color", images)
                    cv2.waitKey(1)

    def get_color_frame(self):
        """Get the current color frame."""
        return self.color_frame

    def get_depth_frame(self):
        """Get the current depth frame."""
        return self.depth_frame
    
    def get_available_streams(self):
        """Get the available streams."""
        context = rs.context()
        devices = context.query_devices()
        for device in devices:
            for sensor in device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    print(profile.stream_name(), profile.format(), profile.fps())
    
    def save_current_rgb_frame(self, filename):
        """Capture the current RGB frame. Filename should include the extension .jpg or .png."""
        cv2.imwrite(filename, self.color_frame)
        print(f"Saved Current RGB Frame as {filename}")

    def save_current_rgbd_frame(self, filename):
        """Capture the current RGBD frame."""
        color_frame, depth_frame = self.rgbd_frame[:,:,0:3], self.rgbd_frame[:,:,3]
        cv2.imwrite(f"{filename}_color.png", color_frame.astype(np.uint8))
        cv2.imwrite(f"{filename}_depth.png", depth_frame)
        print(f"Saved Current Frames as {filename}_color.png and {filename}_depth.png")

if __name__ == "__main__":
    camera = RealSenseCamera(visualization=True)
    camera.start()

    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        camera.stop()



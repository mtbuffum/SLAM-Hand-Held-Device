import cv2

class Camera:
    def __init__(self, camera_index: int = 1, width: int = 640, height: int = 480, fps: int=30):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
    
    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame
    
    def release(self):
        self.cap.release()
        
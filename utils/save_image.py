import cv2
import numpy as np

def save_numpy_frames_to_video(frames, output_path, fps=30):
    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if frame.shape[-1] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    out.release()
    cv2.destroyAllWindows()
    print(f"video saved to {output_path}")

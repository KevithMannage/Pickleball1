import cv2
import numpy as np
import math

class CourtDetector:
    """
    Base class for modular court detection. 
    Designed to be replaced by SegmentationCourtDetector in the future.
    """
    def process(self, frame):
        """
        Input:
            frame (BGR image)

        Output:
            overlay_frame (frame with drawn court lines)
            court_lines (list of detected lines)
            court_corners (optional list of key points)
        """
        raise NotImplementedError("Subclasses must implement the process method.")

class OpenCVCourtDetector(CourtDetector):
    """
    OpenCV implementation of the CourtDetector using traditional computer vision methods.
    """
    def __init__(self):
        # Parameters for Hough Lines
        self.hough_threshold = 80
        self.min_line_length = 100
        self.max_line_gap = 30
        
    def _get_angle(self, line):
        x1, y1, x2, y2 = line
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def process(self, frame):
        # Create a copy of the frame so we don't mutate the original directly before returning
        overlay_frame = frame.copy()
        court_lines = []
        court_corners = []
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # 4. Detect lines using HoughLinesP
        # Ignore very short or weak edges
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.hough_threshold, 
            minLineLength=self.min_line_length, 
            maxLineGap=self.max_line_gap
        )
        
        if lines is not None:
            # 5. Filter lines
            # - Keep only long lines
            # - (Future enhancement): Remove duplicates using angle + distance threshold
            # - (Future enhancement): Separate horizontal and vertical lines
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Basic filter: only keep sufficiently long lines
                if length >= self.min_line_length:
                    court_lines.append((x1, y1, x2, y2))
                    
            # 7. Draw detected court lines on a copy of the frame (green color, thickness 2)
            for x1, y1, x2, y2 in court_lines:
                cv2.line(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        # 6. Estimate court boundary using intersection of main lines 
        # (This is currently returning empty, but the interface is ready for the geometric math)

        # 8. Return: frame with overlays, list of filtered lines, estimated corner points (if possible)
        return overlay_frame, court_lines, court_corners

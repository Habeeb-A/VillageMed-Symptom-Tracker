"""
VillageMed AI Symptom Tracker
License: MIT
Author: Habeeb Abdulfatah
Description: Real-time facial analysis for health monitoring
"""

import cv2
import mediapipe as mp
import time
import random

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#helper functions
def calculate_eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR) to detect eye openness
    Higher EAR = more open eyes
    """
    # Get eye landmark points
    points = [landmarks[i] for i in eye_indices]
    
    # Calculate vertical distances
    vertical1 = abs(points[1].y - points[5].y)
    vertical2 = abs(points[2].y - points[4].y)
    
    # Calculate horizontal distance
    horizontal = abs(points[0].x - points[3].x)
    
    # Calculate EAR
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def detect_emotion(ear_left, ear_right):
    """
    Detect basic emotion based on eye openness
    This is a simplified approach for the prototype
    """
    avg_ear = (ear_left + ear_right) / 2
    
    if avg_ear > 0.25:
        return "Happy/Alert"
    elif avg_ear > 0.20:
        return "Neutral"
    else:
        return "Stressed/Tired"

def estimate_vitals(ear_left, ear_right, emotion):
    """
    Estimate vital signs based on eye contact concentration
    Note: These are simulated estimates for prototype purposes
    """
    avg_ear = (ear_left + ear_right) / 2
    
    # Simulated vital signs based on alertness level
    if emotion == "Happy/Alert":
        temp = round(36.5 + random.uniform(0, 0.5), 1)
        heart_rate = random.randint(70, 85)
        respiratory = random.randint(14, 18)
    elif emotion == "Neutral":
        temp = round(36.8 + random.uniform(0, 0.4), 1)
        heart_rate = random.randint(75, 90)
        respiratory = random.randint(16, 20)
    else:  # Stressed/Tired
        temp = round(37.0 + random.uniform(0, 0.8), 1)
        heart_rate = random.randint(85, 100)
        respiratory = random.randint(18, 24)
    
    return temp, heart_rate, respiratory

#Main Program
def main():
    """Main function to run the symptom tracker"""
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # For FPS calculation
    prev_time = 0
    frame_count = 0
    screenshot_count = 0
    
    # Eye landmark indices (MediaPipe specific)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    # Log file
    log_file = open("emotion_log.txt", "w")
    log_file.write("Timestamp,Emotion,Temperature,Heart_Rate,Respiratory_Rate\n")
    
    print("Starting VillageMed Symptom Tracker...")
    print("Press 's' to take screenshot")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = face_mesh.process(rgb_frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        
        # If face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
                
                # Calculate eye aspect ratios
                landmarks = face_landmarks.landmark
                ear_left = calculate_eye_aspect_ratio(landmarks, LEFT_EYE)
                ear_right = calculate_eye_aspect_ratio(landmarks, RIGHT_EYE)
                
                # Detect emotion
                emotion = detect_emotion(ear_left, ear_right)
                
                # Estimate vitals
                temp, hr, rr = estimate_vitals(ear_left, ear_right, emotion)
                
                # Display information on screen
                y_offset = 30
                cv2.putText(frame, f"Emotion: {emotion}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(frame, f"Temp: {temp}°C", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(frame, f"Heart Rate: {hr} bpm", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                cv2.putText(frame, f"Respiratory: {rr} /min", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Log every 2 seconds
                if frame_count % (int(fps * 2) if fps > 0 else 30) == 0:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{timestamp},{emotion},{temp},{hr},{rr}\n")
                    log_file.flush()
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('VillageMed Symptom Tracker', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and screenshot_count < 5:
            filename = f"screenshot_{screenshot_count + 1}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
            screenshot_count += 1
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print("Program ended. Check emotion_log.txt for results.")

if __name__ == "__main__":
    main()
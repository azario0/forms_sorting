import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define shapes and their corresponding holes
shapes = ['circle', 'square', 'triangle']
shape_colors = {
    'circle': (0, 255, 0),
    'square': (0, 0, 255),
    'triangle': (255, 0, 0)
}

def draw_shape(frame, shape, position):
    if shape == 'circle':
        cv2.circle(frame, position, 30, shape_colors[shape], -1)
    elif shape == 'square':
        cv2.rectangle(frame, (position[0]-30, position[1]-30), (position[0]+30, position[1]+30), shape_colors[shape], -1)
    elif shape == 'triangle':
        pts = np.array([(position[0], position[1]-30), (position[0]-30, position[1]+30), (position[0]+30, position[1]+30)], np.int32)
        cv2.fillPoly(frame, [pts], shape_colors[shape])

def draw_holes(frame):
    hole_positions = [(100, 400), (300, 400), (500, 400)]
    for shape, position in zip(shapes, hole_positions):
        draw_shape(frame, shape, position)
        cv2.putText(frame, shape, (position[0]-30, position[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def check_grab(hand_landmarks, shape_position, frame):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # Calculate the average position
    hand_center_x = (index_tip.x + thumb_tip.x + middle_tip.x) / 3
    hand_center_y = (index_tip.y + thumb_tip.y + middle_tip.y) / 3
    
    hand_pos = (int(hand_center_x * frame.shape[1]), int(hand_center_y * frame.shape[0]))
    
    distance = np.sqrt((hand_pos[0] - shape_position[0])**2 + (hand_pos[1] - shape_position[1])**2)
    
    return distance < 50  # Increased grab radius

def main():
    cap = cv2.VideoCapture(0)
    current_shape = random.choice(shapes)
    shape_position = (300, 200)
    grabbed = False
    score = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        draw_holes(frame)
        if not grabbed:
            draw_shape(frame, current_shape, shape_position)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if check_grab(hand_landmarks, shape_position, frame):
                    grabbed = True
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    shape_position = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                    draw_shape(frame, current_shape, shape_position)
                elif grabbed:
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    shape_position = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                    draw_shape(frame, current_shape, shape_position)
                    
                    # Check if shape is placed in the correct hole
                    hole_positions = [(100, 400), (300, 400), (500, 400)]
                    for hole_position, shape_name in zip(hole_positions, shapes):
                        if np.sqrt((shape_position[0] - hole_position[0])**2 + (shape_position[1] - hole_position[1])**2) < 50:
                            if shape_name == current_shape:
                                print(f"Correct! {current_shape} placed in the right hole.")
                                score += 1
                                current_shape = random.choice(shapes)
                                shape_position = (300, 200)
                                grabbed = False
                            break

        # Display score
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Shape Sorting Game', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
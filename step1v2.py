import cv2
import numpy as np

# Load the JPEG image
filepath = "C:\\Users\\aarus\\Downloads\\CT-abdomen-400x267.jpg"
img = cv2.imread(filepath)
base_img = img.copy()  # Copy of the original image to use as a base for redrawing

# Initialize global variables
pos_points = [[]]  # List of lists to hold positive polylines
neg_points = [[]]  # List of lists to hold negative polylines
current_phase = "positive"  # Start with collecting positive points

# Function to redraw the entire image with points, lines, coordinates, and instructions
def redraw_image():
    global img, base_img, pos_points, neg_points, current_phase

    img = base_img.copy()  # Reset the image to the original without drawings

    # Draw all positive polylines
    for polyline in pos_points:
        for i, point in enumerate(polyline):
            cv2.circle(img, point, 3, (0, 255, 0), -1)  # Green for positive points
            cv2.putText(img, str(point), (point[0] + 5, point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            if i > 0:
                cv2.line(img, polyline[i - 1], point, (0, 255, 0), 2)

    # Draw all negative polylines
    for polyline in neg_points:
        for i, point in enumerate(polyline):
            cv2.circle(img, point, 3, (0, 0, 255), -1)  # Red for negative points
            cv2.putText(img, str(point), (point[0] + 5, point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            if i > 0:
                cv2.line(img, polyline[i - 1], point, (0, 0, 255), 2)

    # Draw instructions
    instructions = "Left-click to draw. 'N' to switch. 'P' for new pos line." "\n" "'M' for new neg line. 'Q' to quit."
    phase_instruction = "Drawing " + ("Positive (Green)" if current_phase == "positive" else "Negative (Red)") + " Polylines"

    # Split the instructions into lines
    instruction_lines = instructions.split('\n')

    # Starting Y position for the first line
    y0 = 15

    # Loop through each line and draw it
    for i, line in enumerate(instruction_lines):
        # Adjust Y position for each line (15 pixels between lines as an example)
        y = y0 + i * 15
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw the phase instruction below the last instruction line
    cv2.putText(img, phase_instruction, (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


    cv2.imshow('image', img)

# Mouse callback function for drawing points and lines
def click_event(event, x, y, flags, param):
    global pos_points, neg_points, current_phase

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        if current_phase == "positive":
            pos_points[-1].append(point)  # Add to the last list of positive polylines
        else:
            neg_points[-1].append(point)  # Add to the last list of negative polylines

        redraw_image()  # Redraw the image with the new point

# Function to start a new polyline
def start_new_polyline(polyline_type):
    global pos_points, neg_points
    if polyline_type == "positive":
        pos_points.append([])  # Start a new list for a new positive polyline
    else:
        neg_points.append([])  # Start a new list for a new negative polyline
    print(f"Started a new {'positive' if polyline_type == 'positive' else 'negative'} polyline.")
    redraw_image()  # Redraw the image to update the instructions and visible points

# Function to switch between positive and negative points collection
def switch_phase():
    global current_phase
    current_phase = "negative" if current_phase == "positive" else "positive"
    print(f"Switched to {'negative' if current_phase == 'negative' else 'positive'} points collection.")
    redraw_image()  # Redraw the image to update the instructions and visible points

def main():
    cv2.imshow('image', img)  # Initial display
    cv2.setMouseCallback('image', click_event)
    redraw_image()  # Initial drawing of instructions

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Switch phases
            switch_phase()
        elif key == ord('p'):  # Start a new positive polyline
            start_new_polyline("positive")
        elif key == ord('m'):  # Start a new negative polyline
            start_new_polyline("negative")
        elif key == ord('q'):  # Quit
            break

    cv2.destroyAllWindows()
    
    # Keep each polyline as a separate array
    return pos_points, neg_points

if __name__ == "__main__":
    positive_polylines, negative_polylines = main()
    
    # Print each polyline separately
    print("Positive Polylines:")
    for polyline in positive_polylines:
        print(np.array(polyline, dtype=np.float32))
    
    print("Negative Polylines:")
    for polyline in negative_polylines:
        print(np.array(polyline, dtype=np.float32))

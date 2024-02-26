import cv2
import numpy as np
import os

# Load the JPEG image
#filepaths is all the files in a specific folder
folder = "slices_for_prompting"

filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
print(filepaths)
# filepaths = ["C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_0.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_1.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_2.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_3.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_4.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_5.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_6.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_7.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_8.png", "C:\\Users\\aarus\\Downloads\\slices_for_prompting/slice_9.png"]
  # Copy of the original image to use as a base for redrawing
listofdicts = []
for filepath in filepaths:

    img = cv2.imread(filepath)
    base_img = img.copy()
# Initialize global variables
    pos_points = [[]]  # List of lists to hold positive polylines
    pos_points_tosave = [[]]
    neg_points = [[]]  # List of lists to hold negative polylines
    neg_points_tosave = [[]]
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
                pos_points_tosave[-1].append((x, y,0))
            else:
                neg_points[-1].append(point)  # Add to the last list of negative polylines
                neg_points_tosave[-1].append((x, y,0))
            redraw_image()  # Redraw the image with the new point

    # Function to start a new polyline
    def start_new_polyline(polyline_type):
        global pos_points, neg_points
        if polyline_type == "positive":
            pos_points.append([])  # Start a new list for a new positive polyline
            pos_points_tosave.append([])
        else:
            neg_points.append([])  # Start a new list for a new negative polyline
            neg_points_tosave.append([])
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
        dict = {"img": filepath, "pos_polylines": pos_points_tosave, "neg_polylines": neg_points_tosave}
        return dict

# data = [
#     {'img': 'C:\\Users\\aarus\\Downloads\\CT-abdomen-400x267.jpg', 'pos_polylines': [[(322, 188), (197, 147)]], 'neg_polylines': [[(231, 200), (316, 129)]]},
#     {'img': 'C:\\Users\\aarus\\Downloads\\plot([99]).png', 'pos_polylines': [[(315, 128), (392, 179)]], 'neg_polylines': [[(383, 270), (319, 319)]]}


# Iterate through each dictionary in the list
        # for key in ['pos_polylines', 'neg_polylines']:
        #     # Iterate through each list of tuples (polylines) for the current key
        #     for i, polyline in enumerate(dict[key]):
        #         # Iterate through each tuple (coordinate point) in the polyline
        #         # and change it to (point[0], point[1], 0)
        #         dict[key][i] = [(point[0], point[1], 0) for point in polyline]

# print(data)


    if __name__ == "__main__":
        listofdicts.append(main())
print(listofdicts)

import json

filename = 'data.json'

# Write the list of dictionaries to the file in JSON format
with open(filename, 'w') as f:
    json.dump(listofdicts, f, indent=4)

print(f"Data has been saved to {filename}")
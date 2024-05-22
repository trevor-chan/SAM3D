import tkinter as tk
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
import json
path = r'C:\Users\aarus\Desktop\det\DET0001401_avg.nii'
path = r"C:\Users\aarus\Downloads\vol_6\vol_6"
import glob
import os
import tkinter as tk
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
import json
import glob
import os

class NiiImageEditor:
    def __init__(self, master, file_path, slice_axis=2, is_folder=True):
        self.master = master
        if "nii" in file_path:
            is_folder = False
        print(is_folder)
        if is_folder:
            self.nii_data = self.load_pngs_as_array(file_path)
        else:
            nii_file = nib.load(file_path)
            self.nii_data = nii_file.get_fdata()
        print(self.nii_data.shape)
        self.current_slice_index = 0
        self.pos_polylines = [[]]  # Each sublist represents a set of points forming a polyline
        self.neg_polylines = [[]]
        self.current_phase = 'positive'
        self.slices_with_points_x = set()
        self.slices_with_points_y = set()
        self.slices_with_points_z = set()

        self.master.title("NIfTI Image Editor")
        
        self.canvas = tk.Canvas(master, width=512, height=512)
        self.canvas.pack()

        self.max_slices = self.nii_data.shape[slice_axis] - 1
        self.slice_slider = tk.Scale(master, from_=0, to=self.max_slices, orient=tk.HORIZONTAL, command=self.update_image_from_scroll)
        self.slice_slider.pack(fill=tk.X, expand=True)
        self.slice_axis = slice_axis
        
        self.canvas.bind("<Button-1>", self.add_point)
        self.master.bind("<a>", lambda event: self.switch_phase())
        self.master.bind("<w>", lambda event: self.start_new_polyline("positive"))
        self.master.bind("<s>", lambda event: self.start_new_polyline("negative"))
        self.master.bind("<d>", lambda event: self.delete_point())
        self.master.bind("<q>", lambda event: self.on_close())

        self.status_label = tk.Label(master, text="Current Phase: Positive", bg="lightgray")
        self.status_label.pack(fill=tk.X)

        self.instructions_label = tk.Label(master, text="Instructions:\n1. Left-click to draw points.\n2. Press 'A' to switch phases.\n3. Press 'W' for new positive polyline.\n4. Press 'S' for new negative polyline.\n5. Press 'D' to delete.\n6. Press 'Q' to quit.", bg="lightgray")
        self.instructions_label.pack(fill=tk.X) 

        self.slice_buttons_frame = tk.Frame(master)
        self.slice_buttons_frame.pack(fill=tk.X)

        self.scale = 2
        self.shape = self.nii_data.shape[0]
        
        self.update_image()
        self.add_close_button()
        # In __init__, bind keys to switch axes
        self.master.bind("<x>", lambda event: self.switch_axis(0))
        self.master.bind("<y>", lambda event: self.switch_axis(1))
        self.master.bind("<z>", lambda event: self.switch_axis(2))

        # Bind the mouse wheel event to the on_mouse_wheel method
        self.master.bind("<MouseWheel>", self.on_mouse_wheel)  # For Windows and macOS
        self.master.bind("<Button-4>", self.on_mouse_wheel)    # For Unix/Linux scroll up
        self.master.bind("<Button-5>", self.on_mouse_wheel)    # For Unix/Linux scroll down

    def update_status_label(self):
        phase_text = "Positive" if self.current_phase == "positive" else "Negative"
        self.status_label.config(text=f"Current Phase: {phase_text}")

    def load_pngs_as_array(self, folder_path):
        # Find all PNG files in the folder, sorted alphabetically
        png_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        # Load, convert to grayscale, and stack images
        images = [np.array(Image.open(png).convert('L')) for png in png_files]
        if not images:
            raise ValueError("No PNG files found in the specified folder.")
        # Stack images into a 3D NumPy array
        return np.stack(images, axis=-1).astype(np.uint8)
    
    def update_image_from_scroll(self, value):
        new_slice_index = int(value)
        if new_slice_index != self.current_slice_index:
            self.current_slice_index = new_slice_index
            # Set the phase to positive automatically when switching slices
            self.current_phase = 'positive'
            self.update_status_label()  # Update the status label to reflect the change
            self.start_new_polyline(self.current_phase, force_new=True)
            self.update_image()

    def add_close_button(self):
        close_button = tk.Button(self.master, text="Close", command=self.on_close)
        close_button.pack()

    def on_close(self):
        self.save_points()
        self.master.destroy()

    def update_image(self):
        if self.slice_axis == 0:  # X-axis
            slice_2d = self.nii_data[self.current_slice_index, :, :]
        elif self.slice_axis == 1:  # Y-axis
            slice_2d = self.nii_data[:, self.current_slice_index, :]
        else:  # Z-axis or default
            slice_2d = self.nii_data[:, :, self.current_slice_index]
        img = Image.fromarray(slice_2d).convert("L")
        img = img.resize((self.shape * self.scale, self.shape * self.scale), Image.ANTIALIAS)
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
        self.draw_polylines()
        self.update_slice_buttons()

    def start_new_polyline(self, polyline_type, force_new=False):
        # Only add a new polyline if the last one contains points or if forced
        if polyline_type == "positive":
            if force_new or (self.pos_polylines[-1] and self.pos_polylines[-1][-1][self.slice_axis] == self.current_slice_index):
                self.pos_polylines.append([])
        else:
            if force_new or (self.neg_polylines[-1] and self.neg_polylines[-1][-1][self.slice_axis] == self.current_slice_index):
                self.neg_polylines.append([])
        self.current_phase = polyline_type

    def draw_polyline(self, polyline, color):
        # Filter points to only those in the current slice
        points_on_slice = [point for point in polyline if point[self.slice_axis] == self.current_slice_index]

        for i, point in enumerate(points_on_slice):
            # Convert 3D point to 2D drawing coordinates
            if self.slice_axis == 0:  # X-axis
                draw_x, draw_y = point[1] * self.scale, point[2] * self.scale
            elif self.slice_axis == 1:  # Y-axis
                draw_x, draw_y = point[0] * self.scale, point[2] * self.scale
            else:  # Z-axis
                draw_x, draw_y = point[0] * self.scale, point[1] * self.scale
            
            # Draw the point
            self.draw_point(draw_x, draw_y, color, point)

            # Connect points with lines if they are consecutive in the same slice
            if i > 0:
                prev_point = points_on_slice[i-1]
                if self.slice_axis == 0:
                    prev_x, prev_y = prev_point[1] * self.scale, prev_point[2] * self.scale
                elif self.slice_axis == 1:
                    prev_x, prev_y = prev_point[0] * self.scale, prev_point[2] * self.scale
                else:
                    prev_x, prev_y = prev_point[0] * self.scale, prev_point[1] * self.scale
                self.canvas.create_line(draw_y, draw_x, prev_y, prev_x, fill=color)

    def draw_polylines(self):
        self.canvas.delete("all")  # Clear existing canvas items
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)  # Redraw the image

        # Draw positive and negative polylines
        for polyline in self.pos_polylines:
            self.draw_polyline(polyline, "green")
        for polyline in self.neg_polylines:
            self.draw_polyline(polyline, "red")

    def draw_point(self, y, x, color, coordinates):
        radius = 5
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
        # Optionally display 3D coordinates as text next to the point
        text_offset = 10
        coordinates_text = f"({coordinates[0]}, {coordinates[1]}, {coordinates[2]})"
        self.canvas.create_text(x + text_offset, y + text_offset, text=coordinates_text, fill=color, font=("TkDefaultFont", 8))

    def delete_point(self):
        target_polylines = self.pos_polylines if self.current_phase == "positive" else self.neg_polylines
        if target_polylines[-1]:
            target_polylines[-1].pop()
            self.update_image()
            self.update_slice_buttons()

    def redraw_image(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
        self.draw_polylines()

    def switch_phase(self):
        self.current_phase = "negative" if self.current_phase == "positive" else "positive"
        self.update_status_label()

    def add_point(self, event):
        y, x = event.x, event.y
        # Translate 2D canvas coordinates (x, y) into 3D image coordinates based on the slicing axis
        if self.slice_axis == 0:  # X-axis slicing
            point = (self.current_slice_index, int(x / self.scale), int(y / self.scale))
            self.slices_with_points_x.add(self.current_slice_index)
        elif self.slice_axis == 1:  # Y-axis slicing
            point = (int(x / self.scale), self.current_slice_index, int(y / self.scale))
            self.slices_with_points_y.add(self.current_slice_index)
        else:  # Z-axis slicing, or default
            point = (int(x / self.scale), int(y / self.scale), self.current_slice_index)
            self.slices_with_points_z.add(self.current_slice_index)

        print(point)

        # Add the point to the appropriate polyline list based on the current phase
        if self.current_phase == 'positive':
            if not self.pos_polylines[-1] or self.pos_polylines[-1][-1][self.slice_axis] != self.current_slice_index:
                self.start_new_polyline("positive", force_new=True)
            self.pos_polylines[-1].append(point)
        else:  # 'negative' phase
            if not self.neg_polylines[-1] or self.neg_polylines[-1][-1][self.slice_axis] != self.current_slice_index:
                self.start_new_polyline("negative", force_new=True)
            self.neg_polylines[-1].append(point)

        self.update_image()
        self.update_slice_buttons()

    def switch_axis(self, axis):
        self.slice_axis = axis
        self.current_slice_index = 0  # Reset slice index or adjust as needed
        self.max_slices = self.nii_data.shape[self.slice_axis] - 1
        self.slice_slider.config(to=self.max_slices)  # Adjust slider range
        self.slice_slider.set(0)  # Reset slider position
        self.update_image()

    def save_points(self):
        # Filter out empty polylines from positive and negative polylines
        filtered_pos_polylines = [polyline for polyline in self.pos_polylines if polyline]
        filtered_neg_polylines = [polyline for polyline in self.neg_polylines if polyline]

        # Prepare the data dictionary to save, using the filtered polylines
        data_to_save = {
            "positive": filtered_pos_polylines,
            "negative": filtered_neg_polylines
        }

        try:
            with open("points.json", "w") as f:
                json.dump(data_to_save, f, indent=4)
        except Exception as e:
            print(f"Failed to save points: {e}")

    def update_slice_buttons(self):
        # Clear existing buttons
        for widget in self.slice_buttons_frame.winfo_children():
            widget.destroy()

        # Add buttons for each slice with points based on the current axis
        if self.slice_axis == 0:
            for slice_index in sorted(self.slices_with_points_x):
                btn = tk.Button(self.slice_buttons_frame, text=f"Slice {slice_index}", command=lambda idx=slice_index: self.go_to_slice(idx))
                btn.pack(side=tk.LEFT)
        elif self.slice_axis == 1:
            for slice_index in sorted(self.slices_with_points_y):
                btn = tk.Button(self.slice_buttons_frame, text=f"Slice {slice_index}", command=lambda idx=slice_index: self.go_to_slice(idx))
                btn.pack(side=tk.LEFT)
        else:
            for slice_index in sorted(self.slices_with_points_z):
                btn = tk.Button(self.slice_buttons_frame, text=f"Slice {slice_index}", command=lambda idx=slice_index: self.go_to_slice(idx))
                btn.pack(side=tk.LEFT)

    def go_to_slice(self, slice_index):
        self.current_slice_index = slice_index
        self.slice_slider.set(slice_index)
        self.update_image()

    def on_mouse_wheel(self, event):
        # Check the operating system
        if self.master.tk.call('tk', 'windowingsystem') == 'win32':
            # On Windows, event.delta gives 120 for scroll up and -120 for scroll down
            increment = -1 if event.delta > 0 else 1
        elif self.master.tk.call('tk', 'windowingsystem') == 'x11':
            # On Unix/Linux, use event.num; 4 for scroll up, 5 for scroll down
            if event.num == 4:
                increment = -1
            else:  # event.num == 5
                increment = 1
        else:
            # On macOS, event.delta gives positive for scroll up and negative for scroll down
            increment = -1 if event.delta > 0 else 1

        # Update the current slice index, ensuring it remains within valid bounds
        new_slice_index = self.current_slice_index + increment
        if 0 <= new_slice_index <= self.max_slices:
            self.current_slice_index = new_slice_index
            self.slice_slider.set(new_slice_index)  # Update the slider position
            
            # Set phase to positive automatically when scrolling
            self.current_phase = 'positive'
            self.update_status_label()  # Ensure this method updates the UI to reflect the phase change
            
            self.start_new_polyline('positive', force_new=True)
            self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = NiiImageEditor(root, path, slice_axis=2)
    root.mainloop()
    print(json.load(open("points.json")))

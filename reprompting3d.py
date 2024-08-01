import tkinter as tk
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
import json
import glob
import os
import utils

class NiiImageEditor:
    def __init__(self, master, file_path, slice_axis=2, is_folder=True, initial_points=None, savepath='tempdir'):
        self.master = master
        if "nii" in file_path:
            is_folder = False
        print(is_folder)
        if is_folder:
            # self.nii_data = utils.padtocube(self.load_pngs_as_array(file_path)) # self.load_pngs_as_array(file_path)
            self.nii_data = utils.padtocube(utils.load3dmatrix(file_path, 'png'))
        else:
            nii_file = nib.load(file_path)
            self.nii_data = nii_file.get_fdata()
            self.nii_data = utils.padtocube(self.nii_data)
        print(self.nii_data.shape)
        self.current_slice_index = 0
        self.pos_polylines = [[]]
        self.neg_polylines = [[]]
        self.current_phase = 'positive'
        self.slices_with_points = {
            0: set(),  # X-axis
            1: set(),  # Y-axis
            2: set()   # Z-axis
        }
        self.savepath = savepath + "/points.json"

        self.master.title("NIfTI Image Editor")
        
        self.canvas = tk.Canvas(master, width=512, height=512)
        self.canvas.pack()

        self.max_slices = self.nii_data.shape[slice_axis] - 1
        self.slice_slider = tk.Scale(master, from_=0, to=self.max_slices, orient=tk.HORIZONTAL, command=self.update_image_from_scroll)
        self.slice_slider.pack(fill=tk.X, expand=True)
        self.slice_axis = slice_axis
        
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.right_click_delete)
        self.master.bind("<a>", lambda event: self.switch_phase())
        self.master.bind("<w>", lambda event: self.start_new_polyline("positive"))
        self.master.bind("<s>", lambda event: self.start_new_polyline("negative"))
        self.master.bind("<d>", lambda event: self.delete_nearest_point())
        self.master.bind("<q>", lambda event: self.on_close())

        self.status_label = tk.Label(master, text="Current Phase: Positive", bg="lightgray")
        self.status_label.pack(fill=tk.X)

        self.instructions_label = tk.Label(master, text=(
            "Instructions:\n"
            "1. Left-click to draw points.\n"
            "2. Press 'A' to switch between positive and negative phases.\n"
            "3. Press 'W' to start a new positive polyline.\n"
            "4. Press 'S' to start a new negative polyline.\n"
            "5. Press 'D' to delete the nearest point to the cursor.\n"
            "6. Use mouse wheel or slider to navigate slices.\n"
            "7. Press 'X', 'Y', or 'Z' to switch viewing axis.\n"
            "8. Press 'Q' to quit and save."
        ), bg="lightgray", justify=tk.LEFT)
        self.instructions_label.pack(fill=tk.X, padx=5, pady=5)

        self.slice_buttons_frame = tk.Frame(master)
        self.slice_buttons_frame.pack(fill=tk.X)

        self.scale = 2
        self.shape = self.nii_data.shape[0]
        
        if initial_points:
            self.load_initial_points(initial_points)
        
        self.update_image()
        self.add_close_button()
        self.master.bind("<x>", lambda event: self.switch_axis(0))
        self.master.bind("<y>", lambda event: self.switch_axis(1))
        self.master.bind("<z>", lambda event: self.switch_axis(2))

        self.master.bind("<MouseWheel>", self.on_mouse_wheel)
        self.master.bind("<Button-4>", self.on_mouse_wheel)
        self.master.bind("<Button-5>", self.on_mouse_wheel)

    def load_initial_points(self, initial_points):
        for phase in ['positive', 'negative']:
            if phase in initial_points:
                for polyline in initial_points[phase]:
                    if phase == 'positive':
                        self.pos_polylines.append(polyline)
                    else:
                        self.neg_polylines.append(polyline)
                    
                    for point in polyline:
                        self.slices_with_points[0].add(point[0])
                        self.slices_with_points[1].add(point[1])
                        self.slices_with_points[2].add(point[2])
        
        self.pos_polylines.append([])
        self.neg_polylines.append([])
        
        self.update_image()
        self.update_slice_buttons()

    def update_status_label(self):
        phase_text = "Positive" if self.current_phase == "positive" else "Negative"
        self.status_label.config(text=f"Current Phase: {phase_text}")

    def load_pngs_as_array(self, folder_path):
        png_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        images = [np.array(Image.open(png).convert('L')) for png in png_files]
        if not images:
            raise ValueError("No PNG files found in the specified folder.")
        return np.stack(images, axis=-1).astype(np.uint8)
    
    def update_image_from_scroll(self, value):
        new_slice_index = int(value)
        if new_slice_index != self.current_slice_index:
            self.current_slice_index = new_slice_index
            self.current_phase = 'positive'
            self.update_status_label()
            self.start_new_polyline(self.current_phase, force_new=True)
            self.update_image()

    def add_close_button(self):
        close_button = tk.Button(self.master, text="Close", command=self.on_close)
        close_button.pack()

    def on_close(self):
        self.save_points()
        self.master.destroy()

    def update_image(self):
        if self.slice_axis == 0:
            slice_2d = self.nii_data[self.current_slice_index, :, :]
        elif self.slice_axis == 1:
            slice_2d = self.nii_data[:, self.current_slice_index, :]
        else:
            slice_2d = self.nii_data[:, :, self.current_slice_index]
        img = Image.fromarray(slice_2d).convert("L")
        img = img.resize((self.shape * self.scale, self.shape * self.scale), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
        self.draw_polylines()
        self.update_slice_buttons()

    def start_new_polyline(self, polyline_type, force_new=False):
        if polyline_type == "positive":
            if force_new or (self.pos_polylines[-1] and self.pos_polylines[-1][-1][self.slice_axis] == self.current_slice_index):
                self.pos_polylines.append([])
        else:
            if force_new or (self.neg_polylines[-1] and self.neg_polylines[-1][-1][self.slice_axis] == self.current_slice_index):
                self.neg_polylines.append([])
        self.current_phase = polyline_type

    def draw_polyline(self, polyline, color):
        points_on_slice = [point for point in polyline if point[self.slice_axis] == self.current_slice_index]
        points_on_slice_in_polyline = [point for point in polyline]
        draw_segments = points_on_slice == points_on_slice_in_polyline

        # This takes points in a polyline, omits points that are not on the same slice as current (self.slice_axis - looks for current index)
        # Then, it draws a polyline segment between consecutive points in the new list
        # However, we want to know if the points were consecutive in the old list, and only draw if so

        for i, point in enumerate(points_on_slice):
            if self.slice_axis == 0:
                draw_x, draw_y = point[1] * self.scale, point[2] * self.scale
            elif self.slice_axis == 1:
                draw_x, draw_y = point[0] * self.scale, point[2] * self.scale
            else:
                draw_x, draw_y = point[0] * self.scale, point[1] * self.scale
            
            self.draw_point(draw_x, draw_y, color, point)

            if i > 0 and draw_segments:
                prev_point = points_on_slice[i-1]
                if self.slice_axis == 0:
                    prev_x, prev_y = prev_point[1] * self.scale, prev_point[2] * self.scale
                elif self.slice_axis == 1:
                    prev_x, prev_y = prev_point[0] * self.scale, prev_point[2] * self.scale
                else:
                    prev_x, prev_y = prev_point[0] * self.scale, prev_point[1] * self.scale
                self.canvas.create_line(draw_y, draw_x, prev_y, prev_x, fill=color)

    def draw_polylines(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)

        for polyline in self.pos_polylines:
            self.draw_polyline(polyline, "green")
        for polyline in self.neg_polylines:
            self.draw_polyline(polyline, "red")

    def draw_point(self, y, x, color, coordinates):
        radius = 5
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
        text_offset = 10
        coordinates_text = f"({coordinates[0]}, {coordinates[1]}, {coordinates[2]})"
        self.canvas.create_text(x + text_offset, y + text_offset, text=coordinates_text, fill=color, font=("TkDefaultFont", 8))

    def delete_point(self):
        target_polylines = self.pos_polylines if self.current_phase == "positive" else self.neg_polylines
        if target_polylines[-1]:
            point = target_polylines[-1].pop()
            # Remove the point from slices_with_points if it's the last point on that slice
            for axis in range(3):
                if not any(p[axis] == point[axis] for polyline in self.pos_polylines + self.neg_polylines for p in polyline):
                    self.slices_with_points[axis].discard(point[axis])
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
        if self.slice_axis == 0:
            point = (self.current_slice_index, int(x / self.scale), int(y / self.scale))
        elif self.slice_axis == 1:
            point = (int(x / self.scale), self.current_slice_index, int(y / self.scale))
        else:
            point = (int(x / self.scale), int(y / self.scale), self.current_slice_index)

        # print(point)

        # Add the point to all relevant sets
        self.slices_with_points[0].add(point[0])
        self.slices_with_points[1].add(point[1])
        self.slices_with_points[2].add(point[2])

        if self.current_phase == 'positive':
            if not self.pos_polylines[-1] or self.pos_polylines[-1][-1][self.slice_axis] != self.current_slice_index:
                self.start_new_polyline("positive", force_new=True)
            self.pos_polylines[-1].append(point)
        else:
            if not self.neg_polylines[-1] or self.neg_polylines[-1][-1][self.slice_axis] != self.current_slice_index:
                self.start_new_polyline("negative", force_new=True)
            self.neg_polylines[-1].append(point)

        self.update_image()
        self.update_slice_buttons()

    def switch_axis(self, axis):
        self.slice_axis = axis
        self.current_slice_index = 0
        self.max_slices = self.nii_data.shape[self.slice_axis] - 1
        self.slice_slider.config(to=self.max_slices)
        self.slice_slider.set(0)
        self.update_image()

    def save_points(self):
        filtered_pos_polylines = [polyline for polyline in self.pos_polylines if polyline]
        filtered_neg_polylines = [polyline for polyline in self.neg_polylines if polyline]

        data_to_save = {
            "positive": filtered_pos_polylines,
            "negative": filtered_neg_polylines
        }

        try:
            with open(self.savepath, "w") as f:
                json.dump(data_to_save, f, indent=4)
        except Exception as e:
            print(f"Failed to save points to {filename}: {e}")

    def update_slice_buttons(self):
        for widget in self.slice_buttons_frame.winfo_children():
            widget.destroy()

        slices_with_points = sorted(self.slices_with_points[self.slice_axis])

        for slice_index in slices_with_points:
            btn = tk.Button(self.slice_buttons_frame, text=f"Slice {slice_index}", command=lambda idx=slice_index: self.go_to_slice(idx))
            btn.pack(side=tk.LEFT)

    def go_to_slice(self, slice_index):
        self.current_slice_index = slice_index
        self.slice_slider.set(slice_index)
        self.update_image()

    def on_mouse_wheel(self, event):
        if self.master.tk.call('tk', 'windowingsystem') == 'win32':
            increment = -1 if event.delta > 0 else 1
        elif self.master.tk.call('tk', 'windowingsystem') == 'x11':
            if event.num == 4:
                increment = -1
            else:
                increment = 1
        else:
            increment = -1 if event.delta > 0 else 1

        new_slice_index = self.current_slice_index + increment
        if 0 <= new_slice_index <= self.max_slices:
            self.current_slice_index = new_slice_index
            self.slice_slider.set(new_slice_index)
            
            self.current_phase = 'positive'
            self.update_status_label()
            
            self.start_new_polyline('positive', force_new=True)
            self.update_image()

    def find_nearest_point(self, x, y):
        min_distance = float('inf')
        nearest_point = None
        nearest_polyline = None
        is_positive = True

        for polylines, is_pos in [(self.pos_polylines, True), (self.neg_polylines, False)]:
            for polyline in polylines:
                for point in polyline:
                    if self.slice_axis == 0:
                        px, py = point[1] * self.scale, point[2] * self.scale
                    elif self.slice_axis == 1:
                        px, py = point[0] * self.scale, point[2] * self.scale
                    else:
                        px, py = point[0] * self.scale, point[1] * self.scale
                    
                    distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        nearest_point = point
                        nearest_polyline = polyline
                        is_positive = is_pos

        return nearest_point, nearest_polyline, is_positive

    def delete_specific_point(self, point, polyline, is_positive):
        polyline.remove(point)
        if not polyline:
            if is_positive:
                self.pos_polylines.remove(polyline)
            else:
                self.neg_polylines.remove(polyline)
        
        # Remove the point from slices_with_points if it's the last point on that slice
        for axis in range(3):
            if not any(p[axis] == point[axis] for polyline in self.pos_polylines + self.neg_polylines for p in polyline):
                self.slices_with_points[axis].discard(point[axis])
        
        self.update_image()
        self.update_slice_buttons()

    def delete_nearest_point(self):
        x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        nearest_point, nearest_polyline, is_positive = self.find_nearest_point(y, x)
        if nearest_point:
            self.delete_specific_point(nearest_point, nearest_polyline, is_positive)

    def right_click_delete(self, event):
        nearest_point, nearest_polyline, is_positive = self.find_nearest_point(event.x, event.y)
        if nearest_point:
            self.delete_specific_point(nearest_point, nearest_polyline, is_positive)

def main(imgpath, savepath):
    root = tk.Tk()
    initial_points = None
    path = imgpath
    app = NiiImageEditor(root, path, slice_axis=2, initial_points=initial_points, savepath=savepath)
    root.mainloop()
    # if os.path.exists(f"{savepath}/points.json")
    # json.load(open(f"{savepath}/points.json"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path',)
    args = parser.parse_args()
    main(args.path)
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from sklearn import svm
import pickle
import random
from tkinter import ttk


class SVMTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("hBN Thickness SVM Trainer")
        
        # Frame for image display
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Canvas for displaying the image
        screen_height=root.winfo_screenheight()-200
        screen_width=root.winfo_screenwidth()-300
        self.canvas = tk.Canvas(self.image_frame, width=screen_width, height=screen_height)
        self.canvas.pack()


        

        # Frame for thickness buttons
        self.thickness_frame = tk.Frame(root)
        self.thickness_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Load image button
        self.btn_load_image = tk.Button(self.thickness_frame, text="Load Image", command=self.load_image)
        self.btn_load_image.pack(pady=10)
        print('test')
        
        # Background button
        self.btn_background = tk.Button(self.thickness_frame, text="Background", command=self.set_background)
        self.btn_background.pack(pady=2)

        # Thickness buttons
        thickness_labels = ["0-5 nm", "5-10 nm", "10-15 nm", "15-20 nm", "20-25 nm", "25-30 nm", "30-40 nm",
                            "40-50 nm", "50-60 nm", "60-70 nm", "70-80 nm", "80-100 nm", "larger than 100"]
        self.thickness_values = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
        
        self.thickness_buttons = []
        for label, value in zip(thickness_labels, self.thickness_values):
            button = tk.Button(self.thickness_frame, text=label, command=lambda v=value: self.set_thickness(v))
            button.pack(pady=2)
            self.thickness_buttons.append(button)
        
        # Train button
        self.btn_train = tk.Button(self.thickness_frame, text="Train SVM", command=self.train_svm)
        self.btn_train.pack(pady=5)
        
        # Load model button
        self.btn_load_model = tk.Button(self.thickness_frame, text="Load SVM Model", command=self.load_model)
        self.btn_load_model.pack(pady=5)

        # Predict button
        self.btn_predict = tk.Button(self.thickness_frame, text="Predict", command=self.enable_prediction)
        self.btn_predict.pack(pady=20)
        
        # Variables to store image and selections
        self.image = None
        self.photo = None
        self.thickness = None
        self.box_coords = []
        self.data = []
        self.model = None
        self.background_rgb = None
        self.predict_mode = False
        
        # Bind events for drawing box
        self.canvas.bind("<ButtonPress-1>", self.start_box)
        self.canvas.bind("<B1-Motion>", self.draw_box)
        self.canvas.bind("<ButtonRelease-1>", self.end_box)
        self.canvas.bind("<Motion>", self.Motion)
        
        
        # Variables for image movement and zoom
        self.image_id = None
        self.start_x = 0
        self.start_y = 0
        self.current_scale = 1.0  # Initial scale of the image
        self.movement_enabled = tk.BooleanVar()
        
        # Checkbutton to enable image movement and zoom
        self.click_move = tk.Checkbutton(root, text='Click to Move/Zoom', variable=self.movement_enabled)
        self.click_move.pack(side='bottom', anchor='w', padx=10, pady=10)
        
        # Bind right-click and mouse wheel events for moving and zooming the image
        self.canvas.bind('<ButtonPress-3>', self.move_from)
        self.canvas.bind('<B3-Motion>',     self.move_to)
        self.canvas.bind("<MouseWheel>", self.zoom_image)

    def load_image(self):
        """Load and display an image."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.image = Image.open(file_path).convert('RGB')
            self.image.thumbnail((1280, 960), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(self.image)
            if self.image_id:
                self.canvas.delete(self.image_id)
            self.image_id = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        
        self.imscale = 1.0 
        self.delta = 1.3
        self.image = Image.open(file_path)
        self.width, self.height = self.image.size
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0) 
            

    def set_background(self):
        """Set the current mode to background."""
        self.thickness = -1
        self.predict_mode = False
        self.background_rgb = None

    def set_thickness(self, thickness):
        """Set the current thickness label for drawing."""
        if self.background_rgb is None:
            messagebox.showwarning("Warning", "Please select the background first.")
        else:
            self.thickness = thickness
            self.predict_mode = False
    
    def start_box(self, event):
        """Start drawing a box."""
        if self.image and (self.thickness is not None or self.predict_mode):
            self.box_coords = [(event.x, event.y)]
    
    def draw_box(self, event):
        """Update the box as the mouse moves."""
        if self.image and (self.thickness is not None or self.predict_mode) and self.box_coords:
            x0, y0 = self.box_coords[0]
            x1, y1 = event.x, event.y
            self.canvas.delete("box")
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", tag="box")
    
    def end_box(self, event):
        """Finish drawing the box and collect RGB data or predict thickness."""
        if self.image and (self.thickness is not None or self.predict_mode) and self.box_coords:
            x0, y0 = self.box_coords[0]
            x1, y1 = event.x, event.y
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            if self.predict_mode:
                self.predict_thickness(box)
            else:
                self.collect_rgb_data(box)
            self.canvas.delete("box")
            self.box_coords = []

    def collect_rgb_data(self, box):
        """Collect RGB data from the selected box and store it with the thickness label."""
        if self.image:
            cropped_image = self.image.crop(box)
            img_array = np.array(cropped_image)
            avg_rgb = np.mean(img_array, axis=(0, 1))

            if self.thickness == -1:  # Background
                self.background_rgb = avg_rgb
                messagebox.showinfo("Info", "Background set successfully.")
            else:
                contrast_rgb = (avg_rgb - self.background_rgb) / self.background_rgb
                self.data.append((*contrast_rgb, self.thickness))

    def predict_thickness(self, box):
        """Predict the thickness based on the RGB values in the selected box."""
        if self.model:
            cropped_image = self.image.crop(box)
            img_array = np.array(cropped_image)
            avg_rgb = np.mean(img_array, axis=(0, 1))
            contrast_rgb = (avg_rgb - self.background_rgb) / self.background_rgb
            prediction = self.model.predict([contrast_rgb])[0]
            x0, y0, x1, y1 = box
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=str(prediction), fill="blue", tag="prediction")
        else:
            messagebox.showerror("Error", "No model loaded for prediction.")

    def train_svm(self):
        """Train an SVM model on the collected data."""
        if not self.data:
            messagebox.showerror("Error", "No data collected for training.")
            return
        
        df = pd.DataFrame(self.data, columns=["R", "G", "B", "Thickness"])
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Training data saved to {file_path}.")
        
        # Train SVM model
        X = df[["R", "G", "B"]]
        y = df["Thickness"]
        model = svm.SVC()
        model.fit(X, y)
        
        # Save trained model
        model_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if model_path:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            messagebox.showinfo("Success", f"SVM model saved to {model_path}.")

    def load_model(self):
        """Load a pre-trained SVM model."""
        model_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if model_path:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.background_rgb = None
            messagebox.showinfo("Success", "SVM model loaded successfully. Please select the background again.")

    def enable_prediction(self):
        """Enable prediction mode."""
        if self.background_rgb is None:
            messagebox.showwarning("Warning", "Please select the background first.")
        else:
            self.predict_mode = True

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        if self.movement_enabled.get():
            self.canvas.scan_mark(event.x, event.y)
    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        if self.movement_enabled.get():
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            if self.image_id:
                self.canvas.delete(self.image_id)
            self.show_image()  # redraw the image

    def Motion(self, event):
        x_pos = event.x
        y_pos = event.y
        return [x_pos,y_pos]
    def zoom_image(self, event):
        if self.movement_enabled.get():
            if self.image_id:
                self.canvas.delete(self.image_id)
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            bbox = self.canvas.bbox(self.container)  # get image area
            if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
            else: return  # zoom only inside image area
            scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
            if event.num == 5 or event.delta == -120:  # scroll down
                i = min(self.width, self.height)
                if int(i * self.imscale) < 30: return  # image is less than 30 pixels
                self.imscale /= self.delta
                scale        /= self.delta
            if event.num == 4 or event.delta == 120:  # scroll up
                i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
                if i < self.imscale: return  # 1 pixel is bigger than the visible area
                self.imscale *= self.delta
                scale        *= self.delta
            self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
            self.show_image()
    def show_image(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection
        

            

if __name__ == "__main__":
    root = tk.Tk()
    app = SVMTrainerApp(root)
    root.mainloop()


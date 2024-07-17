import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing GUI")
        
        # Frame to hold the control buttons and labels
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button to load an image
        self.btn_load = tk.Button(control_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack()
        
        # Button to enable drawing mode
        self.btn_draw = tk.Button(control_frame, text="Enable Drawing", command=self.enable_drawing)
        self.btn_draw.pack()
        
        # Button to clear drawings
        self.btn_clear = tk.Button(control_frame, text="Clear Drawings", command=self.clear_drawings)
        self.btn_clear.pack()
        
        # Entry and button for pixel-to-micrometer ratio
        self.ratio_label = tk.Label(control_frame, text="Pixel to Micrometer Ratio:")
        self.ratio_label.pack()
        self.ratio_entry = tk.Entry(control_frame)
        self.ratio_entry.pack()
        self.ratio_button = tk.Button(control_frame, text="Confirm Ratio", command=self.confirm_ratio)
        self.ratio_button.pack()
        # click to move
        self.movement_enabled = tk.BooleanVar()
        self.click_move= tk.Checkbutton(control_frame,text='click to move',variable=self.movement_enabled )
        self.click_move.pack(side='bottom')

        # StringVar to hold the RGB values and contrast text
        self.rgb_values = tk.StringVar()
        self.rgb_values.set("Average RGB Values and Contrast:")
        self.label = tk.Label(control_frame, textvariable=self.rgb_values)
        self.label.pack()

        # StringVars to hold the individual R, G, B values
        self.r_value = tk.StringVar()
        self.g_value = tk.StringVar()
        self.b_value = tk.StringVar()

        # Labels to display the individual R, G, B values
        self.r_label = tk.Label(control_frame, textvariable=self.r_value)
        self.r_label.pack()
        
        self.g_label = tk.Label(control_frame, textvariable=self.g_value)
        self.g_label.pack()
        
        self.b_label = tk.Label(control_frame, textvariable=self.b_value)
        self.b_label.pack()
        
        # Variables to store line coordinates and drawing state
        self.line_coords = []
        self.original_line_coords = []
        self.image = None
        self.photo = None
        self.lines = []
        self.line_labels = []
        
        self.drawing_enabled = False
        self.drawing_active = False
        self.pixel_to_micrometer_ratio = 1.0  # Default ratio

        # Create a canvas to display the image with fixed size
        self.canvas = tk.Canvas(root, width=1280, height=960)
        self.canvas.pack(side=tk.LEFT)

        # Variables for image movement
        self.image_id = None
        self.start_x = 0
        self.start_y = 0
        self.current_scale = 1.0  # Initial scale of the image

        # Bind right-click and mouse wheel events for moving and zooming the image
        self.canvas.bind('<ButtonPress-3>', self.move_from)
        self.canvas.bind('<B3-Motion>',     self.move_to)
        self.canvas.bind("<MouseWheel>", self.zoom_image)

    def load_image(self):
        """Load an image and display it on the canvas."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.image = Image.open(file_path).convert('RGB')  # Open and convert image to RGB
            self.image.thumbnail((1280, 960), Image.Resampling.LANCZOS)  # Resize image to fit within 1280x960 while maintaining aspect ratio
            self.photo = ImageTk.PhotoImage(self.image)  # Convert image to PhotoImage
            if self.image_id:
                self.canvas.delete(self.image_id)
            self.image_id = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW, tags="image")  # Display image on canvas
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))  # Configure scroll region
        
        self.imscale = 1.0 
        self.delta = 1.3
        self.image = Image.open(file_path)
        self.width, self.height = self.image.size
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)

    def enable_drawing(self):
        """Enable drawing mode."""
        self.drawing_enabled = True  # Set drawing_enabled flag to True
        self.canvas.bind("<ButtonPress-1>", self.start_line)  # Bind mouse button press event
        self.canvas.bind("<B1-Motion>", self.draw_line)  # Bind mouse motion event
        self.canvas.bind("<ButtonRelease-1>", self.end_line)  # Bind mouse button release event

    def start_line(self, event):
        """Start a new line when the mouse button is pressed."""
        if self.drawing_enabled:
            self.drawing_active = True  # Set drawing_active flag to True
            self.current_line_start = (event.x / self.current_scale, event.y / self.current_scale)  # Store the starting coordinates of the line
            self.current_line_id = None  # Initialize current_line_id to None

    def draw_line(self, event):
        """Draw a line as the mouse moves."""
        if self.drawing_active:
            scaled_start = (self.current_line_start[0] * self.current_scale, self.current_line_start[1] * self.current_scale)
            current_coords = (event.x, event.y)
            if self.current_line_id:
                self.canvas.delete(self.current_line_id)  # Delete the previous line segment
            # Draw the new line segment
            self.current_line_id = self.canvas.create_line(scaled_start, current_coords, fill="red" if len(self.line_coords) % 2 == 0 else "blue", tags="line")

    def end_line(self, event):
        """Finish drawing the line when the mouse button is released."""
        if self.drawing_active:
            self.drawing_active = False  # Set drawing_active flag to False
            end_coords = (event.x / self.current_scale, event.y / self.current_scale)
            # Store the line coordinates
            self.line_coords.append([self.current_line_start, end_coords])
            self.original_line_coords.append([self.current_line_start, end_coords])
            self.lines.append(self.current_line_id)  # Store the line ID
            # Calculate the length of the line
            line_length = np.sqrt((end_coords[0] - self.current_line_start[0]) ** 2 + 
                                  (end_coords[1] - self.current_line_start[1]) ** 2)
            micrometer_length = line_length / self.pixel_to_micrometer_ratio
            # Add a text label showing the line number and length
            line_number = len(self.line_coords)
            scaled_start = (self.current_line_start[0] * self.current_scale, self.current_line_start[1] * self.current_scale)
            scaled_end = (end_coords[0] * self.current_scale, end_coords[1] * self.current_scale)
            label_id = self.canvas.create_text((scaled_start[0] + scaled_end[0]) // 2,
                                               (scaled_start[1] + scaled_end[1]) // 2,
                                               text=f"{line_number}: {micrometer_length:.2f} {'μm' if self.pixel_to_micrometer_ratio != 1 else 'px'}", fill="black", tags="label")
            self.line_labels.append(label_id)
            self.current_line_id = None  # Reset current_line_id to None
            if len(self.line_coords) == 2:
                self.calculate_rgb_and_contrast()  # Calculate RGB values and contrast if two lines are drawn

    def clear_drawings(self):
        """Clear all drawings from the canvas."""
        for line_id in self.lines:
            self.canvas.delete(line_id)  # Delete each line from the canvas
        for label_id in self.line_labels:
            self.canvas.delete(label_id)  # Delete each label from the canvas
        self.lines = []  # Clear the lines list
        self.line_labels = []  # Clear the labels list
        self.line_coords = []  # Clear the line coordinates list
        self.original_line_coords = []  # Clear the original line coordinates list
        self.rgb_values.set("Average RGB Values and Contrast:")  # Reset the RGB values text
        self.r_value.set("")  # Clear R value
        self.g_value.set("")  # Clear G value
        self.b_value.set("")  # Clear B value
        self.drawing_enabled = False  # Set drawing_enabled flag to False

    def calculate_rgb_and_contrast(self):
        """Calculate and display the average RGB values and contrast for the drawn lines."""
        img_array = np.array(self.image)  # Convert image to numpy array
        if len(self.line_coords) < 2:
            return  # Return if less than two lines are drawn
        
        line1_rgb = self.get_line_rgb(self.line_coords[0], img_array)  # Get RGB values for the first line
        line2_rgb = self.get_line_rgb(self.line_coords[1], img_array)  # Get RGB values for the second line
        
        # Calculate contrast between the two lines for each RGB channel
        contrast = [abs(line2_rgb[i] - line1_rgb[i]) / line1_rgb[i] for i in range(3)]
        # Set the result text
        result_text = (f"Average RGB Line 1 (background): {line1_rgb}\n"
                       f"Average RGB Line 2 (sample): {line2_rgb}\n"
                       f"Contrast: {contrast}")
        self.rgb_values.set(result_text)  # Update the RGB values and contrast text
        
        # Update the individual R, G, B values
        self.r_value.set(f"R values: Line 1 - {line1_rgb[0]}, Line 2 - {line2_rgb[0]}, Contrast - {contrast[0]:.2f}")
        self.g_value.set(f"G values: Line 1 - {line1_rgb[1]}, Line 2 - {line2_rgb[1]}, Contrast - {contrast[1]:.2f}")
        self.b_value.set(f"B values: Line 1 - {line1_rgb[2]}, Line 2 - {line2_rgb[2]}, Contrast - {contrast[2]:.2f}")

    def get_line_rgb(self, line_coords, img_array):
        """Get the average RGB values along a line."""
        # Generate x and y values along the line
        x_values = np.linspace(line_coords[0][0], line_coords[1][0], num=1000).astype(int)
        y_values = np.linspace(line_coords[0][1], line_coords[1][1], num=1000).astype(int)
        rgb_values = img_array[y_values, x_values]  # Get RGB values from the image array
        avg_rgb = np.mean(rgb_values, axis=0).astype(int)  # Calculate the average RGB values
        return avg_rgb  # Return the average RGB values

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

    def confirm_ratio(self):
        """Confirm the pixel-to-micrometer ratio and update the lengths of lines."""
        try:
            ratio = float(self.ratio_entry.get())
            if ratio > 0:
                self.pixel_to_micrometer_ratio = ratio
            else:
                self.pixel_to_micrometer_ratio = 1.0
        except ValueError:
            self.pixel_to_micrometer_ratio = 1.0
        
        # Update the lengths of existing lines
        for i, (start, end) in enumerate(self.original_line_coords):
            line_length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            micrometer_length = line_length / self.pixel_to_micrometer_ratio
            self.canvas.itemconfig(self.line_labels[i], text=f"{i + 1}: {micrometer_length:.2f} {'μm' if self.pixel_to_micrometer_ratio != 1 else 'px'}")

if __name__ == "__main__":
    root = tk.Tk()  # Create the main window
    app = ImageApp(root)  # Create an instance of the ImageApp class
    root.mainloop()  # Run the main event loop

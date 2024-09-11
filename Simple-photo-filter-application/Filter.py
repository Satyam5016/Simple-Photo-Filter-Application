import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

class PhotoFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Photo Filter Application")

        # Set background color
        self.root.configure(bg='lightblue')

        # Configure grid layout
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)
        root.grid_rowconfigure(3, weight=1)
        root.grid_rowconfigure(4, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)
        root.grid_columnconfigure(3, weight=1)

        self.label = Label(root, text="Choose an image and apply a filter", bg='lightblue', font=('Arial', 14))
        self.label.grid(row=0, column=0, columnspan=4, pady=10)

        self.upload_button = Button(root, text="Upload Image", command=self.upload_image, bg='lightgrey', font=('Arial', 12))
        self.upload_button.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

        self.grayscale_button = Button(root, text="Apply Grayscale", command=self.apply_grayscale, bg='lightgrey', font=('Arial', 12))
        self.grayscale_button.grid(row=1, column=1, padx=10, pady=5, sticky='ew')

        self.sepia_button = Button(root, text="Apply Sepia", command=self.apply_sepia, bg='lightgrey', font=('Arial', 12))
        self.sepia_button.grid(row=1, column=2, padx=10, pady=5, sticky='ew')

        self.edge_button = Button(root, text="Apply Edge Detection", command=self.apply_edge_detection, bg='lightgrey', font=('Arial', 12))
        self.edge_button.grid(row=1, column=3, padx=10, pady=5, sticky='ew')

        self.hsv_button = Button(root, text="Apply HSV", command=self.apply_hsv, bg='lightgrey', font=('Arial', 12))
        self.hsv_button.grid(row=2, column=0, padx=10, pady=5, sticky='ew')

        self.blur_button = Button(root, text="Apply Blur", command=self.show_blur_slider, bg='lightgrey', font=('Arial', 12))
        self.blur_button.grid(row=2, column=1, padx=10, pady=5, sticky='ew')

        self.contour_button = Button(root, text="Apply Contour", command=self.apply_contour, bg='lightgrey', font=('Arial', 12))
        self.contour_button.grid(row=2, column=2, padx=10, pady=5, sticky='ew')

        self.edge_enhanced_button = Button(root, text="Apply Edge Enhancement", command=self.apply_edge_enhancement, bg='lightgrey', font=('Arial', 12))
        self.edge_enhanced_button.grid(row=2, column=3, padx=10, pady=5, sticky='ew')

        self.sharpen_button = Button(root, text="Apply Sharpening", command=self.apply_sharpening, bg='lightgrey', font=('Arial', 12))
        self.sharpen_button.grid(row=3, column=0, padx=10, pady=5, sticky='ew')

        self.face_button = Button(root, text="Apply Face Detection", command=self.apply_face_detection, bg='lightgrey', font=('Arial', 12))
        self.face_button.grid(row=3, column=1, padx=10, pady=5, sticky='ew')

        self.save_button = Button(root, text="Save Image", command=self.save_image, bg='lightgrey', font=('Arial', 12))
        self.save_button.grid(row=3, column=2, padx=10, pady=5, sticky='ew')

        self.image = None
        self.tk_image = None
        self.blur_scale = None  # Slider for blur intensity

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.show_image(self.image)

    def show_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.label.config(image=self.tk_image)

    def apply_grayscale(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.show_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR))

    def apply_sepia(self):
        if self.image is not None:
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_image = cv2.transform(self.image, sepia_filter)
            sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
            self.show_image(sepia_image)

    def apply_edge_detection(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.show_image(edges_colored)

    def apply_hsv(self):
        if self.image is not None:
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # Convert HSV back to BGR for display
            self.show_image(hsv_image)

    def show_blur_slider(self):
        if self.image is not None:
            # Create a scale for selecting the blur intensity (odd numbers from 1 to 50)
            if not self.blur_scale:
                self.blur_scale = Scale(self.root, from_=1, to=50, orient=HORIZONTAL, length=300,
                                        label="Blur Intensity", command=self.apply_blur)
                self.blur_scale.grid(row=4, column=0, columnspan=4, padx=10, pady=5)
                self.blur_scale.set(15)  # Set default blur value

    def apply_blur(self, val):
        if self.image is not None:
            ksize = int(val)
            if ksize % 2 == 0:  # Ensure kernel size is odd
                ksize += 1
            blurred_image = cv2.GaussianBlur(self.image, (ksize, ksize), 0)
            self.show_image(blurred_image)

    def apply_contour(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = self.image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
            self.show_image(contour_image)

    def apply_edge_enhancement(self):
        if self.image is not None:
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            edge_enhanced_image = cv2.filter2D(self.image, -1, kernel)
            self.show_image(edge_enhanced_image)

    def apply_sharpening(self):
        if self.image is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened_image = cv2.filter2D(self.image, -1, kernel)
            self.show_image(sharpened_image)

    def apply_face_detection(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self.show_image(self.image)

    def save_image(self):
        if self.image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.image)

if __name__ == "__main__":
    root = Tk()
    app = PhotoFilterApp(root)
    root.mainloop()

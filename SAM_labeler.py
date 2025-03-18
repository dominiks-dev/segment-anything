import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QMenu, QAction, QFileDialog, QComboBox, QWidget
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QKeySequence
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

from SAM_labler_data import SamData

# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def cv_image_to_qimage(cv_image):
   height, width, channel = cv_image.shape
   bytes_per_line = 3 * width
   qimage = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
   return qimage
def numpy_to_qpixmap(np_img):
    if np_img.ndim == 3 and np_img.shape[2] == 3:
        # For RGB images
        h, w, ch = np_img.shape
        bytes_per_line = 3 * w
        q_img = QImage(np_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    elif np_img.ndim == 3 and np_img.shape[2] == 4:
        # For RGBA images
        h, w, ch = np_img.shape
        bytes_per_line = 4 * w
        q_img = QImage(np_img.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
    else:
        raise ValueError("Unsupported image format")
    return QPixmap.fromImage(q_img)

 

class ImageWidget(QLabel):
   # Create a custom signal to emit the clicked point
   clickedPoint = pyqtSignal(QPoint)
   startSam = pyqtSignal()
   display_second_screen = pyqtSignal()

   def __init__(self, img_path):
      super().__init__()
      self.setAlignment(Qt.AlignCenter)
      self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus for the QLabel

      # Load an example image for demonstration
      try: 
         self.image = QPixmap(img_path)
      except Exception as e: 
         print(e.args)
         return

      self.modified_image = self.image.copy()  # Create a copy to hold the modifications
      self.setPixmap(self.image)

   def resizeEvent(self, event):
      # Scale the image to fit the new size while maintaining aspect ratio
      self.modified_image = self.image.scaled(self.size(), Qt.KeepAspectRatio)
      self.setPixmap(QPixmap(self.modified_image))

   def mousePressEvent(self, event):
      if event.button() == Qt.LeftButton:
         # The current pixmap size
         pixmap_size = self.pixmap().size()

         # Calculate the scale factors
         scale_width = pixmap_size.width() / self.image.width()
         scale_height = pixmap_size.height() / self.image.height()

         # Get the click position in the widget's coordinates
         click_pos = event.pos()

         # Adjust for centering/margins (if any)
         # Assuming the image is centered in the label
         offset_x = (self.width() - pixmap_size.width()) / 2
         offset_y = (self.height() - pixmap_size.height()) / 2

         # Adjust the click position to the image's scale
         image_x = int((click_pos.x() - offset_x) / scale_width)
         image_y = int((click_pos.y() - offset_y) / scale_height)
         image_pos = QPoint(image_x, image_y)# Get the click position in the widget's coordinates
         click_pos = event.pos()

         # Emit the clicked point through a signal
         self.clickedPoint.emit(image_pos)        

         # Draw a point at the click position on the modified image
         painter = QPainter(self.modified_image)
         painter.setPen(QPen(Qt.red, 5))
         painter.drawPoint(image_pos)
         painter.end()
 
         # Update the displayed image with the modified image
         self.setPixmap(self.modified_image)

      elif event.button() == Qt.MiddleButton: 
         self.startSam.emit()
      
      elif event.button() == Qt.RightButton: 
         self.display_second_screen.emit()


   def keyPressEvent(self, event):
      if event.key() == Qt.Key_R:
         # Reset the modified image to the original image
         self.modified_image = self.image.copy() # TODO: check if that is okay
         self.setPixmap(self.image) 
         # Emit the clicked point through a signal
         point = QPoint(-1, -1)
         self.clickedPoint.emit(point) 

   def display_cv_image(self, cv_image):      
      # Convert OpenCV image to QImage
      qimage = cv_image_to_qimage(cv_image)

      # Display the QImage in the QLabel
      self.setPixmap(QPixmap.fromImage(qimage))


import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os 
""" 
class SecondWindow(QWidget): 
    def __init__(self): 
        super().__init__()
        self.w = 512
        self.h = 512
        self.resize(self.w, self.h)
        self.setWindowOpacity(0.8) 
        self.setStyleSheet('background-color: lightgreen') 
        
        # Create an image (black 512x512 RGB)
        self.image = np.zeros([512, 512, 3], dtype=np.uint8)

        # Convert the NumPy array to QPixmap
        pixmap = numpy_to_qpixmap(self.image)

        # Create a QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Move the window to the second screen
        self.move_to_second_screen()

    def move_to_second_screen(self):
        ### Moves the window to the second screen if available.
        app = QApplication.instance()
        screens = app.screens()  # Get all available screens
        if len(screens) > 1:
            second_screen = screens[1]  # Use the second screen
            screen_geometry = second_screen.geometry()
            x = screen_geometry.x() + (screen_geometry.width() - self.width()) // 2
            y = screen_geometry.y() + (screen_geometry.height() - self.height()) // 2
            self.move(x, y)
        else:
            print("Warning: No second screen detected, displaying on primary screen.")

    def update_image(self, new_image):
        ###Updates the displayed image dynamically.
        pixmap = numpy_to_qpixmap(new_image)
        self.image_label.setPixmap(pixmap)
"""

class MainWindow(QMainWindow):
   def __init__(self, image_folder= "C:/Users/zinst/Pictures"):
      super().__init__()
      self.img_folder = image_folder
      self.image_list = [f for f in os.listdir(self.img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

      if (len(self.image_list) == 0): 
         self.img_path = "sled.png"
      else: 
         self.img_path = os.path.join(self.img_folder, self.image_list[0]) 
         self.index= 0

      self.initUI()
      
      # Connect the custom signals from ImageWidget to a slot in MainWindow
      self.image_widget.clickedPoint.connect(self.handle_clicked_point)
      self.image_widget.startSam.connect(self.segment_points)
      # self.image_widget.display_second_screen.connect(self.display_img_second_screen) # 
      self.points=[]
      self.samData=SamData(self.img_path) # input default image 

      # if a second window is necessary
      # self.sw= SecondWindow()
      # self.sw.show() 
      

   def initUI(self):
      self.setWindowTitle("Image Viewer")
      self.setGeometry(100, 100, 500, 400)

      # Create a custom ImageWidget as the central widget
      self.image_widget = ImageWidget(self.img_path)
      self.setCentralWidget(self.image_widget)

      # Create a menu bar
      menubar = self.menuBar()

      # Create a QMenu as a dropdown menu
      dropdown_menu = QMenu("Dropdown Menu", self)

      # Add items to the dropdown menu
      dd_item_a = QAction("Small", self)
      dd_item_b = QAction("Middle", self)
      dd_item_c = QAction("Large", self)
      dropdown_menu.addAction(dd_item_a)
      dropdown_menu.addAction(dd_item_b)
      dropdown_menu.addAction(dd_item_c)

      # Connect the actions to their respective slots
      dd_item_a.triggered.connect(self.handle_dropdown_selection)
      dd_item_b.triggered.connect(self.handle_dropdown_selection)
      dd_item_c.triggered.connect(self.handle_dropdown_selection)

      # Add the dropdown menu to the menu bar
      menubar.addMenu(dropdown_menu)

      # Create the "File" menu
      file_menu = menubar.addMenu("File")

      # Create an action for loading an image
      load_action = QAction("Load Image", self)
      load_action.triggered.connect(self.load_image)
      file_menu.addAction(load_action)
      save_action = QAction("Save Labels", self)
      save_action.triggered.connect(self.save_labels)
      file_menu.addAction(save_action)

      # Create a dropdown menu (QComboBox) with three options
      dropdown_menu = QComboBox(self)
      dropdown_menu.addItem("Option A")
      dropdown_menu.addItem("Option B")
      dropdown_menu.addItem("Option C")
      dropdown_menu.activated[str].connect(self.handle_dropdown_selection)
      menubar.setCornerWidget(dropdown_menu)

      # Create a context menu for the image widget
      # self.image_widget.setContextMenuPolicy(Qt.CustomContextMenu)
      # self.image_widget.customContextMenuRequested.connect(self.show_context_menu)


   def display_img_second_screen(self): 
      # self.sw.update_image(new_numpy_image)
      pass


   def handle_clicked_point(self, point):
      # Receive the clicked point from ImageWidget and do something with it
      if (point.x() == -1 and point.y() == -1):
         self.points.clear()
         print("Cleared Points")
      else:
         # self.points.append((point))
         self.points.append((point.x(), point.y()))
         print("Added Point:", point.x(), point.y())

   def segment_points(self): 
      pts = self.points
      if (len(self.points)==0):
         return
      # set the points as input label and get the mask from sam 
      mask, score = self.samData.predict_point_inputs(self.points)

      # merge mask and original image
      alpha = 0.6
      beta = (1.0 - alpha) 
       
      image = cv2.imread(self.img_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

      # color = np.array([30/255, 144/255, 255/255, 0.6])
      # # h, w = mask['segmentation'].shape[-2:]
      # h, w = mask.shape[-2:]
      # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

      # Convert the binary mask to a 3-channel orange color image
      mask = mask.squeeze()  # Remove the singleton dimension if present
      height, width = mask.shape

      # Create an empty 3-channel image
      orange_mask_img = np.zeros((height, width, 3), dtype=np.uint8)

      # Set the orange color (in BGR format) in the regions where the mask is True
      # orange_color_image[mask] = (0, 165, 255) 
      orange_mask_img[mask] = (255, 165, 0) # in rgb

      cv_image = cv2.addWeighted(orange_mask_img, alpha, image, beta, 0.0)
      # pass the mask to the image display 
      self.image_widget.display_cv_image(cv_image)
      # self.sw.update_image(cv_image)


   def load_image(self):
      options = QFileDialog.Options()
      file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.gif);;All Files (*)", options=options)
      if file_path:
         self.image_widget.image = QPixmap(file_path)
         self.image_widget.modified_image = self.image_widget.image.copy()  # Create a copy to hold the modifications
         self.image_widget.setPixmap(self.image_widget.image)
         self.samData.encode_img(file_path)
   def save_labels(self):
      pass

   def handle_dropdown_selection(self):
      action = self.sender()
      if action:
         print("Selected option:", action.text)



if __name__ == "__main__":
   app = QApplication(sys.argv) 
   window = MainWindow('C:/Users/Anwender/Desktop/2025_03_06_Becher_sorted/Flash') # MainWindow("A:\Bilder") 
   window.show()
   
   sys.exit(app.exec_())

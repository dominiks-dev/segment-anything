import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QMenu, QAction, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QKeySequence
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 
def cv_image_to_qimage(cv_image):
   height, width, channel = cv_image.shape
   bytes_per_line = 3 * width
   qimage = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
   return qimage

class ImageWidget(QLabel):
   # Create a custom signal to emit the clicked point
   clickedPoint = pyqtSignal(QPoint)
   startSam = pyqtSignal()

   def __init__(self, default_img):
      super().__init__()
      self.setAlignment(Qt.AlignCenter)
      self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus for the QLabel

      # Load an example image for demonstration
      try: 
         self.image = QPixmap(default_img)
      except: 
         self.image=None
      self.modified_image = self.image.copy()  # Create a copy to hold the modifications
      self.setPixmap(self.image)

   def mousePressEvent(self, event):
      if event.button() == Qt.LeftButton:
         # Get the click position in the image coordinates
         click_pos = event.pos()
         image_pos = self.mapFrom(self, click_pos)

         # Emit the clicked point through a signal
         self.clickedPoint.emit(image_pos)        

         # Draw a point at the click position on the modified image
         painter = QPainter(self.modified_image)
         painter.setPen(QPen(Qt.red, 5))
         painter.drawPoint(image_pos)
 
         # Update the displayed image with the modified image
         self.setPixmap(self.modified_image)
      elif event.button() == Qt.MiddleButton: 
         self.startSam.emit()

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


from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch # has to be outside of sam class
import matplotlib.pyplot as plt
import cv2 
#TODO: refactor to seperate file
class SamData(): 

   def __init__(self, image_path) -> None:
      self.sam_checkpoint = "sam_vit_b_01ec64.pth"  # min model 0.37gb
      self.model_type = "vit_b"   
      # self.image_path = "A:/datasets/ProQuaOpt/2023-03-14 11_13_43_area.png"
      self.image_path = "A:/TrainingsDaten/ProQuaOpt/SnowFox/RGB/2023-03-14 11_13_43_area.png"

      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
      self.sam.to(device=self.device)

      self.predictor = SamPredictor(self.sam)
      if(image_path!=None):
         self.load_image_CV(self.image_path)
      

   def load_image_CV(self, image_path): 
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    self.image = image # DS: only temp TODO: remove

    # Pass loaded image through to predictor
    self.predictor.set_image(image)
   
   def predict_point_inputs(self, points):  
      input_points_orig = np.array([[500, 375], [1125, 625]])
      input_points = np.array(points) 
      # need shape (1, m, n)
      # input_points= np.expand_dims(input_points_dy, axis=0)
      # input_label = np.array([1, 1, 1])
      input_label = np.ones(input_points.shape[0])
      masks, scores, logits = self.predictor.predict(
         point_coords=input_points,
         point_labels=input_label,
         multimask_output=False, # only one mask if true
        ) 
      return (masks, scores) 

# setup basic variables
# sam_checkpoint = "sam_vit_h_4b8939.pth" # max model 2,5gb
# model_type = "vit_h" 
# sam_checkpoint = "sam_vit_l_0b3195.pth" # middle model 1,2gb
# model_type = "vit_l" 

class MainWindow(QMainWindow):
   def __init__(self, default_img):
      super().__init__()
      self.default_img= default_img
      self.initUI()
      # Connect the custom signal from ImageWidget to a slot in MainWindow
      self.image_widget.clickedPoint.connect(self.handle_clicked_point)
      self.image_widget.startSam.connect(self.segment_points)
      self.points=[]
      self.samData=SamData(default_img) # input default image

   def initUI(self):
      self.setWindowTitle("Image Viewer")
      self.setGeometry(100, 100, 500, 400)

      # Create a custom ImageWidget as the central widget
      self.image_widget = ImageWidget(self.default_img)
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
      self.image_widget.setContextMenuPolicy(Qt.CustomContextMenu)
      self.image_widget.customContextMenuRequested.connect(self.show_context_menu)

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
      # set the points as input label and get the mask from sam 
      mask, score = self.samData.predict_point_inputs(self.points)

      # merge mask and original image
      alpha = 0.5
      beta = (1.0 - alpha) 
      im2 = self.samData.image
      # color = np.array([30/255, 144/255, 255/255, 0.6])
      # # h, w = mask['segmentation'].shape[-2:]
      # h, w = mask.shape[-2:]
      # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

      # Convert the binary mask to a 3-channel orange color image
      mask = mask.squeeze()  # Remove the singleton dimension if present
      height, width = mask.shape

      # Create an empty 3-channel image
      orange_color_image = np.zeros((height, width, 3), dtype=np.uint8)

      # Set the orange color (in BGR format) in the regions where the mask is True
      # orange_color_image[mask] = (0, 165, 255) 
      orange_color_image[mask] = (255, 165, 0) # in rgb

      cv_image = cv2.addWeighted(orange_color_image, alpha, im2, beta, 0.0)
      # pass the mask to the image display 
      self.image_widget.display_cv_image(cv_image)

   def load_image(self):
      options = QFileDialog.Options()
      file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.gif);;All Files (*)", options=options)
      if file_path:
         self.image_widget.image = QPixmap(file_path)
         self.image_widget.modified_image = self.image_widget.image.copy()  # Create a copy to hold the modifications
         self.image_widget.setPixmap(self.image_widget.image)
         self.samData.load_image_CV(file_path)
   def save_labels(self):
      pass

   def show_context_menu(self, pos):
      context_menu = QMenu(self)
      context_menu.addAction("Action 1")
      context_menu.addAction("Action 2")
      context_menu.addAction("Action 3")

      action = context_menu.exec_(self.image_widget.mapToGlobal(pos))

   def handle_dropdown_selection(self):
      action = self.sender()
      if action:
         print("Selected option:", action.text())

if __name__ == "__main__":
   app = QApplication(sys.argv)
   default_img="2023-03-14 11_13_43_area.png"
   window = MainWindow(default_img) 
   window.show()
   sys.exit(app.exec_())

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QMenu, QAction, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QKeySequence
from PyQt5.QtCore import Qt, QPoint

class ImageWidget(QLabel):
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
      # Create points list
      self.points = []

   def mousePressEvent(self, event):
      if event.button() == Qt.LeftButton:
         # Get the click position in the image coordinates
         click_pos = event.pos()
         image_pos = self.mapFrom(self, click_pos)

         # Draw a point at the click position on the modified image
         painter = QPainter(self.modified_image)
         painter.setPen(QPen(Qt.red, 5))
         painter.drawPoint(image_pos)

         self.points.append(image_pos)

         # Update the displayed image with the modified image
         self.setPixmap(self.modified_image)

   def keyPressEvent(self, event):
      if event.key() == Qt.Key_R:
         # Reset the modified image to the original image
         self.modified_image = self.image.copy() # TODO: check if that is okay
         self.setPixmap(self.image)
         self.points.clear()


class MainWindow(QMainWindow):
   def __init__(self, default_img):
      super().__init__()
      self.default_img= default_img
      self.initUI()

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

   def load_image(self):
      options = QFileDialog.Options()
      file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.gif);;All Files (*)", options=options)
      if file_path:
         self.image_widget.image = QPixmap(file_path)
         self.image_widget.modified_image = self.image_widget.image.copy()  # Create a copy to hold the modifications
         self.image_widget.setPixmap(self.image_widget.image)
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

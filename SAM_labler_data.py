from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch # has to be outside of sam class
import matplotlib.pyplot as plt
import cv2 

class SamData(): 

   def __init__(self, image_path) -> None:
      self.sam_checkpoint = "sam_vit_b_01ec64.pth"  # min model 0.37gb
      self.model_type = "vit_b"   
      # check if there are images in the default path - if not load default 
      if(image_path==None):
         self.image_path = "sled.png"
      else: 
         self.image_path=image_path

      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
      self.sam.to(device=self.device)

    #   self.image_resized=None
      self.predictor = SamPredictor(self.sam)      

      self.encode_img(self.image_path) 
         

   def encode_img(self, image_path): 
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    # Pass loaded image through to predictor
    self.predictor.set_image(image)
    # self.image_resized = self.predictor.transform.apply_image(image)
   
   def predict_point_inputs(self, points):  
      # input_points_orig = np.array([[500, 375], [1125, 625]])
      input_points = np.array(points) 
      # need shape (1, m, n)
      # input_points= np.expand_dims(input_points_dy, axis=0)
      # input_label = np.array([1, 1, 1])
      if (len(input_points) == 0): 
         return
      input_label = np.ones(input_points.shape[0]) # only one class
      masks, scores, logits = self.predictor.predict(
         point_coords=input_points,
         point_labels=input_label,
         multimask_output=False, # only one mask if true
        ) 
      return (masks, scores) 


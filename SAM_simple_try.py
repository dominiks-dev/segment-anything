
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch 
import matplotlib.pyplot as plt
import cv2

# setup basic variables
# sam_checkpoint = "sam_vit_h_4b8939.pth" # max model 2,5gb
# model_type = "vit_h" 
# sam_checkpoint = "sam_vit_l_0b3195.pth" # middle model 1,2gb
# model_type = "vit_l" 

sam_checkpoint = "sam_vit_b_01ec64.pth"  # min model 0.37gb
model_type = "vit_b"  

image_path = "G:/Pytorch/KI/truck1.png"

# device = "cuda"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load image - with numpy or cv
from PIL import Image 

def load_image_PIL(image_path):
    # Open the image file
    img = Image.open(image_path)
    # Convert the image to a NumPy array
    img_array = np.array(img)
    return img_array
def load_image_CV(image_path): 
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    asdf = mask['segmentation']
    h, w = mask['segmentation'].shape[-2:]
    mask_image = mask['segmentation'].reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# B) or generate masks for an entire image:
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)
image = load_image_CV(image_path) 

masks = mask_generator.generate(image)

# from pycocotools import mask as mask_utils
# mask = mask_utils.decode(annotation["segmentation"])

print("masks len: ", len(masks), "  mask shape = ", masks[0]['segmentation'].shape)
 

plt.figure(figsize=(10,10))
plt.imshow(image)
for mask in masks: 
    show_mask(mask, plt.gca())
# show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show() 



# A) to generate a mask

# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image("G:/Pytorch/KI/truck.jpg")
# masks, _, _ = predictor.predict(<input_prompts>)

# set parameters for input: 
input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

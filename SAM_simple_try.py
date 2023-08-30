
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

# image_path = "G:/Pytorch/KI/truck1.png"
# image_path = "A:/datasets/ProQuaOpt/2023-03-14 11_13_43_area.png"
image_path = "A:/TrainingsDaten/ProQuaOpt/SnowFox/RGB/2023-03-14 11_13_43_area.png"
image_path = "A:/TrainingsDaten/ProQuaOpt/SnowFox/RGB/NIO_shortshot/5.bmp"
# image_path = "A:/datasets/ProQuaOpt/test10__2023-06-20 09_27_20.png"

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

def show_mask_dict(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    # asdf = mask['segmentation']
    h, w = mask['segmentation'].shape[-2:]
    # h, w = mask.shape[-2:]
    mask_image = mask['segmentation'].reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)  
def show_mask(mask, ax, random_color=True):
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
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def resize_CV(image, width=512, height=512, percent= 100.0): 
    # resizing in percentage has prio
    if (percent != 100): 
        width= int(image.shape[1]*percent/100)
        height = int(image.shape[0]*percent/100)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
image = load_image_CV(image_path) 
image=resize_CV(image, percent=50)

# B) or generate masks for an entire image:
segEntireImage=False
if (segEntireImage):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) 
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.92,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )
    
    # DS: memory explodes if large image is used (5536*3692) - even 2768*1846 to big
    # "Tried to allocate 14.62 GiB (GPU 0; 8.00 GiB total capacity; 1.17 GiB already allocated; 3.77 GiB free; 1.90 GiB reserved in total by PyTorch)"
    masks = mask_generator.generate(image) 

    # from pycocotools import mask as mask_utils
    # mask = mask_utils.decode(annotation["segmentation"])
    print("masks len: ", len(masks), "  mask shape = ", masks[0]['segmentation'].shape)    

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    for mask in masks: 
        show_mask_dict(mask, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show() 


segInputLabel=True
# A) to generate a mask
if (segInputLabel):
    # sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    predictor.set_image(image)
    # masks, _, _ = predictor.predict(<input_prompts>)

    # set parameters for input: 
    input_point = np.array([[488, 607], [1125, 625]])
    input_label = np.array([1, 1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     mask_input=mask_input[None, :, :],
    #     multimask_output=False,
    # )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()  
        # DS: TODO: paint points into image
        plt.imsave(f"out_{i}.png", image)
        # cv2.imwrite(f"out_{i}.png", image)

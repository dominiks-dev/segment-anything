import torch
import os


from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

cwd = os.getcwd()
print("current directory = ", cwd)  # DS: change the path in the launch.json file
print("Loading model...")

model_type = "vit_l"
checkpoint = "sam_vit_l_0b3195.pth"
output = model_type + ".onnx"

sam = sam_model_registry[model_type](checkpoint=checkpoint)

# optional sam config parameters
return_single_mask = True  # DS: if time try if multiples are possible and take the "best" one
gelu_approximate = False
use_stability_score = False
return_extra_metrics = False

onnx_model = SamOnnxModel(
    model=sam,
    return_single_mask=return_single_mask,
    use_stability_score=use_stability_score,
    return_extra_metrics=return_extra_metrics,
)


# Assuming that `model` is your PyTorch model.
# Make sure the model is in evaluation mode.
onnx_model = onnx_model.eval()

# Define the size of your input tensor. The size should be the same as the input size your model expects.
# For example, if your model takes in 3-channel images of size 224x224, the input size would be (1, 3, 224, 224) for a single image.
input_tensor = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(onnx_model,               # model being run
                  input_tensor,        # model input (or a tuple for multiple inputs)
                  "model.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=17,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],    # the model's input names
                  output_names = ['output'])  # the model's output names

import onnx

# Load the ONNX model
model = onnx.load("vit_b.onnx")

# Get the name and shape of the input
for i in model.graph.input:
    print(i.name, end = ' ')
    for dim in i.type.tensor_type.shape.dim:
        # Check if the dimension is fixed or dynamic
        if (dim.dim_value != 0):
            print(dim.dim_value, end = ', ')
        else:
            print('dynamic', end = ', ')

print("end")
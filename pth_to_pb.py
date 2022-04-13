
import torch
import tensorflow
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import onnx
from onnx_tf.backend import prepare

PATH = "./model.pth"
base_classes = ["", "unweighted", "weighted"]
num_classes = len(base_classes)+1

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

dummy_input = torch.autograd.Variable(torch.randn(1, 3, 360, 540)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(model, dummy_input, "model.onnx")
print("SAVED ONNX MODEL")

# Load the ONNX file
model = onnx.load('output/mnist.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

print('inputs:', tf_rep.inputs)
# tf_rep.export_graph('model.pb')
print("SAVED TF MODEL")

import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os

os.chdir("/users/salar/desktop/ftc-nn")

cap = cv2.VideoCapture("vid.mp4")
PATH = "./model.pth"
base_classes = ["", "unweighted", "weighted"]
box_colors = [(), (255,0,0), (0,255,0)]
num_classes = len(base_classes)+1
score_threshold = 0.3

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()


while True:
    _, frame = cap.read()   
    # USE THE FOLLOWING + Z-SCORE NORMALIZE WHEN USING TENSORFLOW
        # width, height, _ = frame.shape
        # frame = cv2.resize(frame, (216, 324)) # change image dimensions to 32*32 
        # frame_t = frame.transpose((2,0,1)) # flip image dimensions to match model input
        # frame_t = torch.tensor(frame_t) # convert numpy frame to tensor
        # frame_t = frame_t.view(1,3,1920,1080).float() # add extra dimension at front because model needs 4 dimensions (since it can take multiple photos at once)

    PILimage = Image.fromarray(frame).convert("RGB")
    # ToTensor flips image dimensions to (channel, width, height) and does minimax normalization
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    frame_t = t(PILimage).unsqueeze(0)

    with torch.no_grad():
        prediction = model(frame_t)

    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy().astype(int)
        label = prediction[0]["labels"][element]
        score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals= 4)
        if score > score_threshold:
            cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), box_colors[label], 1)
            cv2.putText(frame, str(base_classes[label]), (boxes[0], boxes[1]), cv2.FONT_HERSHEY_PLAIN, 1, box_colors[label], 2)
  
    cv2.imshow("Freight Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break


# frame_t = frame_t[0].permute(1, 2, 0).int()  # change dimensions back to original so plt can display
# frame_t = frame_t[:,:,[2,1,0]] # switch channels from RGB to BGR so plt can display it correctly
# plt.imshow(frame_t) # display last frame in video capture
# plt.show()

cap.release()
cv2.destroyAllWindows()
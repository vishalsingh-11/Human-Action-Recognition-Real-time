import cv2
import glob

import torch
import torchvision
import cv2
import numpy as np
import csv
import pathlib
from timesformer.models.vit import TimeSformer
from torchvision import transforms
from PIL import Image
import time
import os


class ActionRecognition(object):
    def __init__(self, src=0):
        self.no_of_preds = 2
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Utilizing",self.device,"for execution")

    def initialize_model(self, model_path):
        # Initializing NTU RGB+D model
        model = TimeSformer(img_size=224, num_classes=60, num_frames=16,
                            attention_type='divided_space_time', pretrained_model=model_path)
        model.eval()
        model.to(self.device)
        # prepare the translation dictionary label-action
        rows = []
        with open('NTU_labels.csv', 'r',) as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                rows.append(row)
        rows.pop(0)
        idx_to_class = {}
        for i in rows:
            idx_to_class[int(i[0].split(',')[0])] = i[0].split(',')[1]
        return model, idx_to_class

    def webcam_inference(self, model, idx_to_class):
        frame_count = 0
        stack = []
        transform = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[
                                 0.225, 0.225, 0.225]),
        ])
        softmax = torch.nn.Softmax()

        # Start looping on frames received from webcam
        vs = cv2.VideoCapture(0)
        softmax = torch.nn.Softmax()
        nn_output = torch.tensor(np.zeros((1, 60)), dtype=torch.float32).cuda()

        while True:
            # read each frame and prepare it for feedforward in nn (resize and type)
            ret, orig_frame = vs.read()

            if ret is False:
                print("Camera disconnected or not recognized by computer")
                break

            old_frame = orig_frame.copy()
            frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame).view(1, 3, 224, 224).cuda()

            # add frame to the stack and remove oldest frame if necessary
            if len(stack) < 16:
                stack.append(frame)
            else:
                stack.pop(0)
                stack.append(frame)

            # feed the stacked frames to the neural network
            if len(stack) == 16:
                nn_input = torch.stack(stack, dim=2)
                nn_output = model(nn_input)

            # vote for class with 25 consecutive frames
            if frame_count % 10 == 0:
                nn_output = softmax(nn_output)
                nn_output = nn_output.data.cpu().numpy()
                preds = nn_output.argsort()[0][-self.no_of_preds:][::-1]
                pred_classes = []
                for pred in preds:
                # Filtering to remove actions when nothing is available in frame
                    if nn_output[0, pred] > 0.2:
                        pred_classes.append(
                            (idx_to_class[pred+1], nn_output[0, pred]))

                
                # reset the process
                nn_output = torch.tensor(
                    np.zeros((1, 60)), dtype=torch.float32).cuda()

            # Display the resulting frame and the classified actions
            if len(pred_classes) > 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                y0, dy = 300, 40
                for i in range(len(pred_classes)):
                    y = y0 + i * dy
                    cv2.putText(orig_frame, '{} - {:.2f}'.format(pred_classes[i][0], pred_classes[i][1]),
                                (5, y), font, 1, (0, 0, 255), 2)
            cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                'Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Webcam', orig_frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = r'pretrained_models/checkpoint_epoch_00015.pyth'
    action_object = ActionRecognition()
    model, label_dict = action_object.initialize_model(model_path)
    action_object.webcam_inference(model, label_dict)

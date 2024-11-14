import numpy as np
import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from random import randrange

from default_settings import GeneralSettings
from tracker.boost_track import BoostTrack

import utils


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = self.preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 5))


    def preproc(self, image, input_size, mean, std, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r


# Instantiate an object detector
model = YOLO('best.pt', 'detect')

# Check for GPU availability
device = select_device('0')
# Devolve the processing to selected devices
model.to(device)

# Load the video file
cap = cv2.VideoCapture('Sample.mp4')

# Get the frame rate, frame width, and frame height
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the MP4 codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

# Instantiate a tracker
tracker = None
results = {}
frame_count = 0
total_time = 0
video_name = "custom_video"
tag = "custom_test"
preproc=ValTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

# Define a color list for track visualization
colors = {}

# Process each frame of the video
while cap.isOpened():
    # Load the frame
    ret, frame = cap.read()
    if not ret:
        break
    track_img, _ = preproc(frame, (frame_height, frame_width), (frame_height, frame_width))
    track_img = torch.from_numpy(track_img).cuda()
    track_img = track_img.unsqueeze(0)

    if frame_count == 0:
        # if tracker is not None:
        #     tracker.dump_cache()

        tracker = BoostTrack(video_name=video_name)

    # Detect people in the frame
    prediction = model.predict(frame, imgsz=(frame_height,frame_width), conf=0.1, iou=0.45,
                                half=False, device=device, max_det=99, classes=0,
                                verbose=False)

    # Exclude additional information from the predictions
    prediction_results = prediction[0].boxes.cpu().numpy()
    dets = np.concatenate((prediction_results.xyxy, np.expand_dims(prediction_results.conf, axis=1)), axis=1)

    # Update the tracker with the latest detections
    targets = tracker.update(dets, track_img, frame, tag)
    tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])

    frame_count += 1

    # Skip additional analysis if the tracker is not currently tracking anyone
    if len(tlwhs) == 0:
        continue

    # Visualize tracks
    for idx, (track_id, bbox) in enumerate(zip(ids, tlwhs)):
        # Define a new color for newly detected tracks
        if track_id not in colors:
            colors[track_id] = (randrange(255), randrange(255), randrange(255))

        color = colors[track_id]

        # Extract the bounding box coordinates
        x0, y0, w, h = map(int, bbox)
        x1 = x0 + w
        y1 = y0 + h

        # Draw the bounding boxes on the frame
        annotated_frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        # Put the track label on the frame alongside the bounding box
        cv2.putText(annotated_frame, str(track_id), (x0, y0-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    # Write the frame to the output video file
    out.write(annotated_frame)

# Release everything when done
cap.release()
out.release()

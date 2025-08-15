import WebGUI
import Frequency
import HAL

import numpy as np
import cv2
import onnxruntime
# ONNX Runtime utilizes the DLLs installed by PyTorch
onnxruntime.preload_dlls() 
# alternate add: import pytorch

# CONSTANT
IMAGE_SIZE = (240, 640) 
MODEL_PATH = "workspace/code/model.onnx"

# enable cuda

try:
    session = onnxruntime.InferenceSession(MODEL_PATH,providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("[INFO] ONNX model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load ONNX model: {e}")
    session = None

if session is None:
    exit(0)


# Preprocessing function
def preprocess_image(img):
    if img.shape[0] < 240 or img.shape[1] < 640:
        img = cv2.resize(img, (640, 480))

    cropped_img = cv2.resize(img[-240 : , :], (IMAGE_SIZE[1], IMAGE_SIZE[0]))

    # BGR to RGB and normalize
    img_rgb = cropped_img[:, :, ::-1].astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))  # HWC to CHW
    img_rgb = np.expand_dims(img_rgb, axis=0)  # Add batch dimension

    return img_rgb

while True:
    Frequency.tick()

    # GET IMAGE FROM HAL API
    raw_img = HAL.getImage()

    cropped_img = preprocess_image(raw_img)

    output = session.run([output_name], {input_name: cropped_img})[0]
    v, w = output[0]

    HAL.setV(v)
    HAL.setW(w)

    WebGUI.showImage(raw_img)

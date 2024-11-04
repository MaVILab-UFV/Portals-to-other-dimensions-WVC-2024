import torch
from torchvision import transforms
from inferencer import Inferencer
from pasticheModel import PasticheModel
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import json
from ultralytics import YOLO
import json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def style_image(img, inference, filter=5):
    img_PIL = Image.fromarray(img)
    img_PIL.thumbnail((256, 256), Image.LANCZOS)

    styled_img = inference.eval_image(img_PIL, 1, filter, 0.0)
    styled_img = np.array(styled_img)
    styled_img = cv2.cvtColor(styled_img, cv2.COLOR_RGB2BGR)
    styled_img = cv2.resize(styled_img, (img.shape[1], img.shape[0]))
    return styled_img

def portal_img(img, model, inference, style=5):
    results = model(img, verbose=False)
    res_json = json.loads(results[0].to_json())
    if len(res_json) == 0:
        return img
    height, width, _ = img.shape


    result = max(res_json, key=lambda x: x['confidence'], default=None)
    index = res_json.index(result)

    if result['class'] != 0:
        styled_img = style_image(img, inference, style)
        #get segmentation mask
        mask_door = (results[index].masks.data[index].cpu().numpy() * 255).astype(np.uint8) 
        mask_door = cv2.resize(mask_door.astype(np.uint8), (img.shape[1], img.shape[0]))

        #create convex hull of the door mask
        _, threshold = cv2.threshold(mask_door, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))

        convex_hull_image = np.zeros_like(mask_door)
        cv2.drawContours(convex_hull_image, [hull], 0, 255, thickness=cv2.FILLED)

        #create the final image
        interior = cv2.bitwise_xor(mask_door, convex_hull_image)
        white_part = cv2.bitwise_and(styled_img, styled_img, mask=interior)
        black_part = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(interior))

        result = cv2.bitwise_or(black_part, white_part)

        return result
    else:
        return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save-video', action='store_true', help='Save the video')
    parser.add_argument('--style', type=int, help='Choose which style to apply')
    parser.add_argument('--model-path', type=str, default=256, help='Image size for style transfer')
    args = parser.parse_args()

    cv2.namedWindow('Real Time', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real Time', 2400, 1800) 

    model = YOLO('best.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_styles = 16
    model_save_dir = args.model_path

    pastichemodel = PasticheModel(num_styles)

    inference = Inferencer(pastichemodel,device,256)
    inference.load_model_weights(model_save_dir)

    cap = cv2.VideoCapture(0)

    if args.save_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter('output_video_path.mp4', codec, fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = portal_img(frame, model, inference, args.style)
        if args.save_video:
            output_video.write(frame)
        cv2.imshow('Real Time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if args.save_video:
        output_video.release()
    cv2.destroyAllWindows()
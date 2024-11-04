import functools
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from PIL import Image
from ultralytics import YOLO
import argparse
import torch 
import json

def tensor_to_image(tensor):
    """Converts a tensor to a PIL Image."""
    tensor = (tensor * 255).numpy().astype(np.uint8)  # Convert tensor to uint8 directly
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    return tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)

@functools.lru_cache(maxsize=None)
def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads an image and prepares it for processing."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img_shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = tf.reduce_max(img_shape)
    scale = 512 / long_dim

    new_shape = tf.cast(img_shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]  # Add batch dimension

    return img

def preprocess_image(image, max_dim):
    """Preprocesses the image for style transfer."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    img = img.astype(np.float32) / 255.0  # Normalize pixel values

    img_tensor = tf.convert_to_tensor(img)
    img_shape = tf.cast(tf.shape(img_tensor)[:-1], tf.float32)
    long_dim = tf.reduce_max(img_shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(img_shape * scale, tf.int32)
    img_tensor = tf.image.resize(img_tensor, new_shape)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

    return img_tensor

def style_image(img, hub_module, style_path):
    """Applies style transfer to an image."""
    content_image = preprocess_image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 512)
    style_image_tensor = load_image(style_path)
    
    stylized_image = hub_module(content_image, style_image_tensor)[0]
    styled_img = tensor_to_image(stylized_image)

    # print(type(styled_img))

    styled_img = cv2.cvtColor(styled_img, cv2.COLOR_RGB2BGR)
    styled_img = cv2.resize(styled_img, (img.shape[1], img.shape[0]))


    return styled_img

def portal_img(img, model, inference, style_path):
    results = model(img, verbose=False)
    res_json = json.loads(results[0].to_json())
    if len(res_json) == 0:
        return img
    height, width, _ = img.shape


    result = max(res_json, key=lambda x: x['confidence'], default=None)
    index = res_json.index(result)

    if result['class'] != 0:
        styled_img = style_image(img, inference, style_path)
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
    parser.add_argument('--style-path', type=str, help='Load the directory of image style')
    args = parser.parse_args()

    cv2.namedWindow('Real Time', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real Time', 2400, 1800) 

    model = YOLO('best.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

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

        frame = portal_img(frame, model, hub_module, args.style_path)
        if args.save_video:
            output_video.write(frame)
        cv2.imshow('Real Time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if args.save_video:
        output_video.release()
    cv2.destroyAllWindows()
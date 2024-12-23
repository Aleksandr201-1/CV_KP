import argparse
import io
from PIL import Image
import time
import cv2
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_file

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

sr_image = None
scaled_image = None
to_save = None

app = Flask(__name__)

def preprocess_image(image):
    #hr_image = tf.image.decode_image(image)
    hr_image = np.asarray(image)
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'png', quality=100)
    img_io.seek(0)
    img = base64.b64encode(img_io.getvalue()).decode('ascii')
    img_tag = f'<img src="data:image/png;base64,{img}" class="img-fluid" style="height:500px;"/>'
    return img_tag

def scale_image(image, scale):
    """Scales down images using bicubic downsampling."""
    image_size = [image.shape[1], image.shape[0]]  # width, height
    image = tf.squeeze(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8))
    
    lr_image = np.asarray(
        Image.fromarray(image.numpy())
        .resize([int(image_size[0] * scale), int(image_size[1] * scale)], Image.Resampling.BICUBIC)
    )
    
    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image

@app.route('/download_image')
def download_file():
    temp = io.BytesIO()
    to_save.save(temp, format="png")
    temp.seek(0)
    return send_file(temp, download_name="SR_Image.png", as_attachment=True)

@app.route("/", methods=["GET", "POST"])
def predict():
    global sr_image
    global scaled_image
    global to_save
    timeS = ''
    max_h = 500
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", sr_image=sr_image, scaled_image=scaled_image)
        file = request.files["file"]
        if not file:
            return render_template("index.html", sr_image=sr_image, scaled_image=scaled_image)

        img_bytes = file.read()
        file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lr_image = preprocess_image(img)

        start = time.time()
        sr_image = model(lr_image)
        sr_image = tf.squeeze(sr_image)
        timeS = str(round(time.time() - start, 2))

        scaled_image = scale_image(img, 4)
        scaled_image = tf.squeeze(scaled_image)
        scaled_image = tf.clip_by_value(scaled_image, 0, 255)
        scaled_image = Image.fromarray(tf.cast(scaled_image, tf.uint8).numpy())
        scaled_image = serve_pil_image(scaled_image)

        if not isinstance(sr_image, Image.Image):
            sr_image = tf.clip_by_value(sr_image, 0, 255)
            sr_image = Image.fromarray(tf.cast(sr_image, tf.uint8).numpy())
            to_save = sr_image
        sr_image = serve_pil_image(sr_image)
    return render_template("index.html", sr_image=sr_image, scaled_image=scaled_image, message=timeS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app for image super-resolution")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = tf.saved_model.load('./model/')
    app.run(host="0.0.0.0", port=args.port, debug=False)

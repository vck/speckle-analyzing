import os 
import numpy as np
from time import strftime
from skimage import feature, color
from skimage.data import imread
from scipy import misc, fftpack
from PIL import Image
from flask import (
    Flask,
    render_template,
    request)

# server config 

DEV_MACHINE = 'X86_64'
PROD_MACHINE = 'armv7l'
PROD_MODE = False
DEBUG = True
UPLOAD_FOLDER = "static/"

# scikit image reader
reader = misc.imread

# flask object
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def img2frq(image):
  img = imread(image, as_grey=True)
  fft = fftpack.fft2(img)
  fft_data = fft.argmax(axis=1).tolist()
  return fft_data


def blob_counter(img, min_sigma=2, max_sigma=8, treshold=0.0001, overlap=0.6):
   img = imread(img, as_grey=True)
   bw = img.mean(axis=2)
   blobs_dog = [(x[0],x[1],x[2]) for x in feature.blob_dog(-bw, min_sigma=2, max_sigma=8, threshold=0.0001, overlap=0.6)]
   blobs_dog = set(blobs_dog)
   return str(len(blobs_dog))


def count_blob(img):
   img = reader(img)
   bw = img.mean(axis=2)
   blobs_dog = [(x[0], x[1], x[2]) for x in feature.blob_dog(-bw, min_sigma=2, max_sigma=8, threshold=0.0001, overlap=0.6)]
   blobs_dog = set(blobs_dog)
   return str(len(blobs_dog))


def generate_filename():
    timestamps = strftime("%m%d%Y-%H%M%S") + '.jpg'
    return timestamps


# drats router 
@app.route("/", methods=["GET", "POST"])
def capture_image():

    if PROD_MODE == False:
        if request.method == "POST":
            image = request.files['file']
            if image:
               image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
               filepath = "static/"+image.filename
               num_blob = count_blob(filepath)
               histogram = Image.open(filepath).histogram()
               fft = img2frq(filepath)
               minx, maxx, std, mean = np.min(fft), np.max(fft), np.std(fft), np.mean(fft)
               data = [[fft[i], i] for i in range(len(fft))]
               return render_template('dev.html', filename=image.filename, num_blob=num_blob, 
					histogram=histogram, minx=minx, maxx=maxx, std=std, mean=mean, data=data)
	
        return render_template("dev.html")


@app.route('/plot')
def plot():
  return render_template("plot.html")




if __name__ == '__main__':
    app.run('0.0.0.0', port=8000, debug=DEBUG)


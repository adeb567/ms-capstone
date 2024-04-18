import numpy as np
from PIL import Image
from os import listdir
from os.path import join, exists
from keras.models import model_from_json
## local libs
from utils.data_utils import read_and_resize, preprocess, deprocess

## test funie-gan
checkpoint_dir  = '/content/drive/MyDrive/4th_Semester/Capstone/Code/FUnIE/models/gen_p/'
model_name_by_epoch = "model_15320_" 

model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"

# sanity
assert (exists(model_h5) and exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)
# load weights into new model
funie_gan_generator.load_weights(model_h5)
print("\nLoaded data and model")

 
# get the path/directory
input_dir = "/home/amitabha/capstone/instance_version/val/"
resize_dir = "/home/amitabha/capstone/instance_version/resize_val_f/"
output_dir = "/home/amitabha/capstone/instance_version/enhanced_val_f/"

for image_path in listdir(input_dir): 
    input_path = join(input_dir, image_path)
    inp_img = read_and_resize(input_path, (256, 256))

    resize_path = join(resize_dir, image_path)
    resized_img = np.array(inp_img, dtype=np.uint8) 
    Image.fromarray(resized_img).save(resize_path)
    
    im = preprocess(inp_img)
    im = np.expand_dims(im, axis=0) # (1,256,256,3)

    # generate enhanced image
    gen = funie_gan_generator.predict(im)
    gen_img = deprocess(gen)[0]

    # save output image
    output_path = join(output_dir, image_path)
    Image.fromarray(gen_img).save(output_path)

import argparse
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
import numpy as np
import json
#Main  
parser = argparse.ArgumentParser(description = "Welcome to the FLOWER prediction script")
parser.add_argument('image_path',help="enter the image_Path:", default="./test_images/wild_pansy.jpg",type=str)
parser.add_argument('model',help="enter the Model Path", default ='./my_model.h5',type=str)
parser.add_argument('--top_k', help="enter a number",type = int, default = 5)
parser.add_argument('--category_names', help="enter the json file",default = "label_map.json",type=str)
#parse arguments
args = parser.parse_args()

model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer': hub.KerasLayer})

image = Image.open(args.image_path)
image = np.asarray(image)
image = tf.convert_to_tensor(image,tf.float32)
image = tf.image.resize(image,(224,224))
image /= 255
image = image.numpy()
image = np.expand_dims(image,axis=0)

ps=model.predict(image)
ps=ps[0].tolist()
t_values, t_indices= tf.math.top_k(ps,k=args.top_k)
probs= t_values.numpy().tolist()
classes= t_indices.numpy()
classes+=1
classes=classes.tolist()
print(probs, classes)

if args.category_names!=None:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    cl_name = [class_names[str(value)] for value in classes]
    print(cl_name)
        



    
 
    

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from shutil import copy2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input 

predictions = []
f = open("classes1.txt","r")
classes = []
for line in f:
    if '+' in line:
        index = line.find("+")
        nline = line[0:index]+"_"+line[index+1:-1]
        classes.append(nline)
        
    else:
        classes.append(line[0:-1])
               
for a in classes:
    path = "D:\DL3 Dataset\categories\\"+a
    if not os.path.exists(path):
        os.makedirs(path)       
        
model = InceptionResNetV2(weights='imagenet')
target_size = (299, 299)

def predict(model, img, target_size):
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  list1 = (decode_predictions(preds, top=3)[0])
  return list1[0]

for i in tqdm(range(12473,12601)):
    img_path = "train_img1\Image-" + str(i) + ".jpg"
    img = Image.open(img_path)
    predicted_class = predict(model, img, target_size)
    name = predicted_class[1].lower()
    predictions.append(name)
    flag = False     
    for l in classes:
        if (name.find(l)!=-1):
            copy2(img_path,"D:\DL3 Dataset\categories\\"+l)
            flag = True
            break
    
    
    if not flag:
        copy2(img_path,"D:\DL3 Dataset\categories\other")
    
    

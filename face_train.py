import cv2
import os
from PIL import Image 
import pickle
import numpy as np
face_dict = {}
def get_images_and_labels(path):
    global face_dict
    names=os.listdir("dataset")
    image_paths=[]
    new_path="dataset"
    for name in names:
        image_paths.append([os.path.join(new_path+"\\"+name, f) for f in os.listdir(new_path+"\\"+name)])
    print(image_paths)
 # images will contains face images
    images = []
 # labels will contains the label that is assigned to the image
    labels = []
    nbr = 0
    i = 0
    j = 0
    for name in names:
        for image_path in image_paths[j]:
             # Read the image and convert to grayscale
            image_pil = Image.open(image_path).convert('L')
             # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            face_dict[j] = name
            print(face_dict)
            #nbr=int(''.join(str(ord(c)) for c in nbr))
            print(image_path)
             # Detect the face in the image
            faceCascade = cv2.CascadeClassifier('face.xml')
            faces = faceCascade.detectMultiScale(image)
             # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(j)
                cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                cv2.waitKey(10)
            i = i+1
        j = j+1
    return images,labels

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#file_rename(path,names)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer\\trainer.yml')
images, labels = get_images_and_labels('dataset\\ayush')
#cv2.imshow('test',images[0])
#cv2.waitKey(1)
save_obj(face_dict,"trainer\\trained_faces")
recognizer.train(images, np.array(labels))
recognizer.save('trainer\\trainer.yml')
cv2.destroyAllWindows()

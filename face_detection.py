import cv2
import sys
import os

# cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier("face.xml")
name = "ayush"
try:
    os.mkdir("dataset")
except:
    pass
try:
    os.mkdir("dataset"+"\\"+name)
except:
    pass
video_capture = cv2.VideoCapture(0)
new_path = "dataset\\ayush"
i = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        i=i+1
        cv2.imwrite(new_path+"\\"+name +'_'+ str(i) + ".jpg", gray[y:y+h,x:x+w])
        print("file_saved")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(i)
    # Display the resulting frame
    try:
        cv2.imshow('Video', frame)
    except:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if i+1>50:
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

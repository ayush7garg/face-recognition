import cv2
import pickle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer\\trainer.yml')
cascadePath = "face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
nbr_predicted, conf=0,0
with open("trainer\\trained_faces.pkl",'rb') as f:
	face_dict = pickle.load(f)
# face_dict=load_obj("trainer\\trained_faces")
cam = cv2.VideoCapture(0)
while True:
	ret, im = cam.read()
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
	for(x,y,w,h) in faces:
		nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
		print(round(100 - conf, 2))
		cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)

		label = face_dict[nbr_predicted]
		if(round(100 - conf, 2)>30):
			cv2.putText(im,str(label),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
		else :
			cv2.putText(im,str("UNKNOWN"),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

		cv2.putText(im,"press esc to exit",(x-45,y-40),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0),2)

	cv2.imshow('im',im)
	k = cv2.waitKey(100) & 0xff
	if k == 27:
		break
cam.release()
cv2.destroyAllWindows()
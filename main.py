

import cv2

import dlib

import face_recognition

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   #yüz tanıma için

eyeCascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

videoCapture = cv2.VideoCapture(0)              #video akışı için VideoCapture nesnesi oluşturur

detector=dlib.get_frontal_face_detector()
trump=face_recognition.load_image_file("trump.jpg") #kişi tanımlama için
trump_enc=face_recognition.face_encodings(trump)[0]

while True:                                 # sonsuz döngü oluşturarak videonun sürekli olmasını sağlar
    ret, frame = videoCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #renkli görüntüyü okuyabilmek için gri tona dönüştürür



    face_loc=face_recognition.face_locations(frame)
    face_encoding=face_recognition.face_encodings(frame,face_loc)


    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:                        #tespit edilen yüzlerin koordinatlarını döngü içinde işlemek için kullanılır.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Yüz bölgesi içinde göz algılama
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 1) # parantezden sonraki 1 çerceve çizgisinin kalınlıgını belirliyor

    for face in face_encoding:
        sonuc=face_recognition.compare_faces([trump_enc],face)
        print(sonuc)

    # Gülümsemeleri algıla
    smiles = smile_cascade.detectMultiScale(frame, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

    # Algılanan gülümsemeleri işle
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, 'Smile', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    cv2.imshow('Video', frame)          # işlenen video çerçevesini ekranda gösterir

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()                           #Açılmış olan tüm OpenCV pencelerini kapatır.

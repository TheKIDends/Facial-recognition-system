# Bước 4: Sử dụng dữ liệu đã train để nhận diện tên người trên webcam

import cv2 # thư viện opencv
import numpy as np # thư viện toán học
import os # thư viện truy cập vào hệ dẫn, hệ thống để lấy đường dẫn các thư mục
import sqlite3 # thư viện thao tác với database
from PIL import Image # trích xuất ảnh trong thư mục

# thư viện để nhận diện khuôn mặt ở đâu trên camera
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# thư viện để nhận dạng xem khuôn mặt đấy là ai
recognizer = cv2.face.LBPHFaceRecognizer_create()

# đọc dữ liệu từ file đã train
recognizer.read('D:\\TheKIDends\\IT\\AI\\Facial recognition system\\Facial-recognition-system\\recognizer\\tranningData.yml')

# Hàm lấy thông tin của người trong database thông qua ID
def getProfile(id):
    # kết nối đến database
    conn = sqlite3.connect('D:\\TheKIDends\\IT\\AI\\Facial recognition system\\Facial-recognition-system\\database\\data.db')

    # truy vấn các bản ghi trong ID
    query = "SELECT * FROM people WHERE ID=" + str(id)

    # lấy bản ghi trong query
    cusror = conn.execute(query)

    # biến lưu giá trị lấy được trong database
    profile = None

    for row in cusror:
        profile = row
    
    # đóng kết nối với database
    conn.close()
     
    return profile

# nhận diện xem khuôn mặt đang ở đâu trên camera (tương tự như getDataFromWebcam)
# truy cập webcam
cap = cv2.VideoCapture(0)

fontface = cv2.FONT_HERSHEY_SIMPLEX

# while để hiển thị liên tục cho đến khi mình thoát
while (True):
    # biến ret trả về true nếu truy cập thành công
    # biến frame là data lấy được từ webcam
    ret, frame = cap.read()

    # chuyển về ảnh trắng xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # sau khi lấy được ảnh xám kết hợp với thư viện khuôn mặt
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        # vẽ hình vuông bao quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)

        # cắt ảnh trên webcam ra để so sánh với tập dữ liệu
        roi_gray = gray[y: y + h, x: x + w]

        # hàm dự đoán nhận diện khuôn mặt, trả về ID và độ chính xác
        id, confidence = recognizer.predict(roi_gray)

        if confidence < 40:
            profile = getProfile(id)

            # Nếu tìm thấy người (dự đoán được tên người)
            if profile != None:
                # hiển thị tên
                # tham số 1: hình ảnh để hiển thị chữ lên
                # tham số 2: tên người (profile có 2 giá trị là ID và Name -> profile[1] = Name)
                # tham số 3: tọa độ chữ hiển thị
                # tham số 4: font chữ hiển thị 
                # tham số 5: (fontScale) tỷ lệ font chữ  
                # tham số 6: chỉ số màu BGR
                # tham số 7: độ dày chữ
                cv2.putText(frame, "" + str(profile[1]), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
        
        else:
            cv2.putText(frame, "Unknow", (x + 10, y + h + 30), fontface, 1, (0, 0, 255), 2)

    # hiện thị ảnh lên
    cv2.imshow('DETECTING FACE', frame)

    # điều kiện thoát chương trình 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()
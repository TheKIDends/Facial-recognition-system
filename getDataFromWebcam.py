# Quá trình: 
    # Nhận diện khuôn mặt trên webcam
    # Lưu dữ liệu vào database để train cho máy
    # Sử dụng 1 tấm ảnh khác để xem máy có nhận diện đúng không 

# Bước 1: Nhận diện khuôn mặt trên webcam. 

# thư viện opencv
import cv2

# thư viện khuôn mặt mặc định
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# truy cập webcam
cap = cv2.VideoCapture(0)

# while để hiển thị liên tục cho đến khi mình thoát
while (True):
    # biến ret trả về true nếu truy cập thành công
    # biến frame là data lấy được từ webcam
    ret, frame = cap.read()

    # cần chuyển về ảnh trắng xám để train
    # cvt là convert to
    # tham số 1: dữ liệu từ webcam
    # tham số 2: tham số để chuyển từ ảnh màu BGR thành ảnh xám GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # sau khi lấy được ảnh xám kết hợp với thư viện khuôn mặt
    faces = face_cascade.detectMultiScale(gray)

    # vẽ hình vuông bao quanh khuôn mặt
    for (x, y, w, h) in faces:
        # tham số 1: dữ liệu từ webcam
        # tham số 2: tọa độ điểm trái trên của hình vuông
        # tham số 3: tọa độ tịnh tiến trong không gian (tọa độ phải dưới)
        # tham số 3: màu hình vuông BGR 
        # tham số 4: độ dày hình vuông
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)

    # hiện thị ảnh lên
    # tham số 1: tiêu đề ảnh
    # tham số 2: dữ liệu từ webcam
    cv2.imshow('DETECTING FACE', frame)

    # điều kiện thoát chương trình 
    # hàm waitKey giúp chương trình không tắt luôn khi mới bật lên
    # 0xFF = ord('q') là sử dụng nút q để thoát khỏi chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()
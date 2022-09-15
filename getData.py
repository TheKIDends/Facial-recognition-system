import cv2 # thư viện opencv
import numpy as np # thư viện toán học
import sqlite3 # thư viện thao tác với database
import os # thư viện truy cập vào hệ dẫn, hệ thống để lấy đường dẫn các thư mục

# hàm truy cập đến database, thêm mới hoặc update bản ghi
def insertOrUpdate(id, name):
    # kết nối đến database
    # tham số: đường dẫn chứa database
    conn = sqlite3.connect('D:\TheKIDends\IT\AI\Facial recognition system\Facial-recognition-system\database\data.db')

    # truy vấn các bản ghi trong ID
    query = "SELECT * FROM people WHERE ID=" + str(id)

    # lấy bản ghi trong query
    cusror = conn.execute(query)

    # biến kiểm tra ID đã tồn tại chưa, nếu rồi thì update, nếu chưa thì insert
    # nếu có ID trong query rồi thì cho isRecordExist = 1, ngược lại vẫn giữ = 0
    isRecordExist = 0 

    # duyệt từng hàng trên bản ghi
    for row in cusror:
        # cứ có bản ghi cũ trong ID là chuyển về = 1
        isRecordExist = 1

    # nếu chưa có bản ghi
    if isRecordExist == 0:
        # thì insert vào database
        query = "INSERT INTO people(ID, name) VALUES(" + str(id) + ", '"+ str(name)+ "')"
    else:
        # ngược lại, update
        query = "UPDATE people SET Name='" + str(name) + "' WHERE ID=" + str(id)
    
    conn.execute(query)
    conn.commit()
    conn.close()


# load thư viện khuôn mặt mặc định
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# truy cập webcam
cap = cv2.VideoCapture(0)

# cho người dùng nhập từ bàn phím vào ID và Name
# để train cho 1 tập ảnh sao cho máy hiểu với tập ảnh này sẽ có giá trị là ID và Name nhập vào 
# Sau này mình lấy một ảnh mới thì nó có thể nhận diện được là ID vào Name nào

# insert database
id = input("Enter your ID:")
name = input("Enter your Name:")
insertOrUpdate(id, name)

# để đánh chỉ số hình ảnh trong tập ảnh 
sampleNum = 0 

# lấy tập ảnh từ webcam
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
    # tham số 1: dữ liệu ảnh xám
    # tham số 2: scaleFactor là tỉ lệ vùng chữ nhật được thu nhỏ sau mỗi lần quét (default là 1.1)
    # tham số 3: minNeighbors là số khung hình chữ nhật trùng nhau tối thiểu để được xem là 1 đối tượng (default là 3)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

     # vẽ hình vuông bao quanh khuôn mặt
    for (x, y, w, h) in faces:
        # tham số 1: dữ liệu từ webcam
        # tham số 2: tọa độ điểm trái trên của hình vuông
        # tham số 3: tọa độ tịnh tiến trong không gian (tọa độ phải dưới)
        # tham số 3: màu hình vuông BGR 
        # tham số 4: độ dày hình vuông
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2)

        # tạo folder để lưu ảnh vừa cắt được từ hình vuông 

        # kiểm tra xem trong đường dẫn của mình đã có folder đấy chưa
        if not os.path.exists('dataSet'):
            # nếu chưa thì tạo
            os.makedirs('dataSet')
        # ta sẽ lưu tên của ảnh theo định dạng (User.1.1, User.1.2,...) với 1 1 là ID và 1 2 là chỉ số

        # tăng chỉ số hình ảnh 
        sampleNum += 1
        
        # lưu ảnh
        # tham số 1: đường dẫn của ảnh
        # tham số 2: dữ liệu của ảnh cắt được theo hình vuông  (đây là tọa độ tịnh tiến)
        cv2.imwrite('dataSet/User.' + str(id) + '.' + str(sampleNum) + '.jpg', gray[y: y + h, x: x + w])

    # hiện thị ảnh lên
    # tham số 1: tiêu đề ảnh
    # tham số 2: dữ liệu từ webcam
    cv2.imshow('DETECTING FACE', frame)

    # hàm waitKey giúp chương trình không tắt luôn khi mới bật lên
    cv2.waitKey(1)

    # điều kiện thoát, khi số lượng hình ảnh đạt đủ
    if sampleNum >= 100:
        break

# giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()
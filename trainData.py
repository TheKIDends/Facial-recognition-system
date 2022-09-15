import cv2 # thư viện opencv
import numpy as np # thư viện toán học
import os # thư viện truy cập vào hệ dẫn, hệ thống để lấy đường dẫn các thư mục
from PIL import Image # trích xuất ảnh trong thư mục

# để train ảnh cần lấy đc ID của nó và lấy ra một bảng các dữ liệu của ảnh
# lấy ID từ chính tên của ảnh (User.<ID>.<INDEX>)

# thư viện mặc định của opencv để train cho nhận diện hình ảnh 
recognizer = cv2.face.LBPHFaceRecognizer_create()

# đường dẫn tới thư mục ảnh 
path = 'dataSet'

# hàm lấy ID và list các dữ liệu ảnh
def getImageWithID(path):

    # imagePaths là danh sách đường dẫn file ảnh trong thư mục 
    # os.listdir(path) để truy cập vào tất cả các file trong đường dẫn path
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  
    # print(imagePaths)  # ['dataSet\\User.1.1.jpg', 'dataSet\\User.1.2.jpg', ...]

    faces = [] # lưu dữ liệu ảnh
    IDs = [] # lưu list ID

    # duyệt từng đường dẫn trong danh sách
    for imagePath in imagePaths:

        # Image.open(): mở và load ảnh trong thư mục
        # convert: chuyển giữa 2 chế độ màu thông dụng là đen trắng (L) và có màu (RGB)
        faceImg = Image.open(imagePath).convert('L')

        # chuyển dữ liệu ảnh thành mảng các điểm ảnh 
        # tham số 1: dữ liệu đối tượng
        # tham số 2: (dtype) kiểu dữ liệu mong muốn cho bảng
        # int8 là kiểu dữ liệu số nguyên có dấu với độ lớn 8 bits
        faceNp = np.array(faceImg, dtype='uint8')
        # print(faceNp) # ví dụ 1 ảnh được chuyển thành array
        # [[ 61  64  61 ... 180 179 179]
        # [ 62  60  54 ... 183 177 175]
        # [ 60  56  47 ... 185 178 178]
        # ...
        # [  9   6   9 ...  11  11  11]
        # [  8   6  10 ...   8   8   8]
        # [  7   5   9 ...   8   7   7]]

        # ta cần cắt ID ra từ đường dẫn hình ảnh
        # đường dẫn có dạng 'dataSet\\User.<ID>.<INDEX>.jpg'
        ID = int (imagePath.split("\\")[1].split(".")[1])
        # imagePath.split("\\")[0] = dataSet
        # imagePath.split("\\")[1] = User.<ID>.<INDEX>.jpg
        #
        # imagePath.split("\\")[1].split(".")[0] = User
        # imagePath.split("\\")[1].split(".")[1] = <ID>
        # imagePath.split("\\")[1].split(".")[2] = <INDEX>
        # imagePath.split("\\")[1].split(".")[3] = jpg

        # sau khi lấy được dữ liệu ảnh và cả ID, ta lưu vào mảng
        faces.append(faceNp) 
        IDs.append(ID) 

        # hiển thị ảnh để biết khi nào train xong
        cv2.imshow('tranning', faceNp)
        cv2.waitKey(10)

    return faces, IDs
        
faces, IDs = getImageWithID(path)

# kiểm tra xem trong đường dẫn đã có folder tên recognizer chưa
if not os.path.exists('recognizer'):
    # nếu chưa thì tạo
    os.makedirs('recognizer')

# sử dụng lại biến recognizer để train
recognizer.train(faces, np.array(IDs))

# sau khi train xong, nó sẽ trả ra 1 file định dạng yml, ta sẽ lưu nó vào folder vừa tạo
recognizer.save('recognizer/tranningData.yml')

# giải phóng bộ nhớ
cv2.destroyAllWindows()
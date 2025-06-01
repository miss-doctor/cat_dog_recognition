import cv2


def detect_cats_dogs(image):
    # 加载猫脸和狗脸检测模型（请确保XML文件存在）
    cat_cascade = cv2.CascadeClassifier(r'E:\cat_dog_recognition\cat_dog_recognition\haarcascades\haarcascade_frontalcatface.xml')
    # dog_cascade = cv2.CascadeClassifier('haarcascade_frontaldogface.xml')  # 需下载对应的狗脸模型

    # 检查模型是否成功加载
    if cat_cascade.empty():
        print("错误: 猫脸检测模型未加载！")
        return image
    # if dog_cascade.empty():
    #     print("错误: 狗脸检测模型未加载！")
    #     return image

    # 转换为灰度图，提高检测效率
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测猫脸
    cats = cat_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # # 检测狗脸
    # dogs = dog_cascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )

    # 在原图上绘制检测结果
    for (x, y, w, h) in cats:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色框表示猫
        cv2.putText(image, 'Cat', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # for (x, y, w, h) in dogs:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色框表示狗
    #     cv2.putText(image, 'Dog', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return image

# 获取默认的摄像头设备
cap = cv2.VideoCapture(0)

# 创建窗口并设置固定尺寸（关键修改）
cv2.namedWindow("capture", cv2.WINDOW_AUTOSIZE)  # Use valid flag
TARGET_WIDTH, TARGET_HEIGHT = 800, 600           # Define fixed size
cv2.resizeWindow("capture", TARGET_WIDTH, TARGET_HEIGHT)  # Set window size

while True:
    flag, frame = cap.read()
    if flag is False:
        print("摄像头获取出现问题")
        break
    # 复制图像并进行猫狗检测
    result = detect_cats_dogs(frame.copy())
    
    # 调整图像到窗口尺寸（可选，但推荐保持显示比例）
    resized_result = cv2.resize(result, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # 显示结果
    cv2.imshow("capture", resized_result)  # Show resized image
    
    # 按空格键退出
    if cv2.waitKey(2) == ord(" "):
        break

cv2.destroyAllWindows()
cap.release()  # 释放摄像头资源
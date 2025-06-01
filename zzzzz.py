import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import datetime


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("摄像头应用")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # 设置中文字体支持
        self.font = ('SimHei', 10)

        # 摄像头相关变量
        self.cap = None
        self.is_camera_on = False
        self.current_frame = None

        # 识别相关变量
        self.dog_cascade = cv2.CascadeClassifier('dog_cascade.xml')
        self.cat_cascade = cv2.CascadeClassifier(r'E:\cat_dog_recognition\cat_dog_recognition\haarcascades\haarcascade_frontalcatface.xml')

        self.is_detecting_dog = False
        self.is_detecting_cat = False

        # 创建界面组件
        self.create_widgets()

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 创建主框架 - 使用网格布局管理
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 设置网格权重，让视频区域可以扩展，控制区域保持固定大小
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # 创建视频显示区域
        self.video_frame = tk.Label(main_frame, bg="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 创建控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew", pady=10)

        # 摄像头控制按钮
        self.camera_btn = tk.Button(control_frame, text="打开摄像头", font=self.font,
                                    command=self.toggle_camera, width=15)
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        # 拍照按钮
        self.capture_btn = tk.Button(control_frame, text="拍照", font=self.font,
                                     command=self.capture_image, width=15, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        # 录制按钮
        self.record_btn = tk.Button(control_frame, text="开始录制", font=self.font,
                                    command=self.toggle_recording, width=15, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # 狗脸识别按钮
        self.detect_dog_btn = tk.Button(control_frame, text="开始识别狗脸", font=self.font,
                                        command=self.toggle_dog_detection, width=15, state=tk.DISABLED)
        self.detect_dog_btn.pack(side=tk.LEFT, padx=5)

        # 猫脸识别按钮
        self.detect_cat_btn = tk.Button(control_frame, text="开始识别猫脸", font=self.font,
                                        command=self.toggle_cat_detection, width=15, state=tk.DISABLED)
        self.detect_cat_btn.pack(side=tk.LEFT, padx=5)

        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪")
        self.status_label = tk.Label(main_frame, textvariable=self.status_var, font=self.font, fg="blue")
        self.status_label.grid(row=2, column=0, sticky="s", pady=5)

        # 录制相关变量
        self.is_recording = False
        self.out = None

    def toggle_camera(self):
        if not self.is_camera_on:
            # 打开摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头")
                return

            self.is_camera_on = True
            self.camera_btn.config(text="关闭摄像头")
            self.capture_btn.config(state=tk.NORMAL)
            self.record_btn.config(state=tk.NORMAL)
            self.detect_dog_btn.config(state=tk.NORMAL)
            self.detect_cat_btn.config(state=tk.NORMAL)
            self.status_var.set("摄像头已开启")

            # 开始更新视频帧
            self.update_frame()
        else:
            # 关闭摄像头
            self.is_camera_on = False
            self.is_detecting_dog = False
            self.is_detecting_cat = False
            self.camera_btn.config(text="打开摄像头")
            self.capture_btn.config(state=tk.DISABLED)
            self.detect_dog_btn.config(text="开始识别狗脸")
            self.detect_cat_btn.config(text="开始识别猫脸")

            # 如果正在录制，停止录制
            if self.is_recording:
                self.toggle_recording()

            if self.cap:
                self.cap.release()
                self.cap = None

            self.status_var.set("摄像头已关闭")

    def update_frame(self):
        if self.is_camera_on:
            ret, frame = self.cap.read()
            if ret:
                # 转换为RGB格式
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 如果开启了识别
                if self.is_detecting_dog:
                    self.detect_dogs()
                if self.is_detecting_cat:
                    self.detect_cats()

                # 添加录制指示
                if self.is_recording:
                    cv2.putText(self.current_frame, "录制中...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # 调整图像大小（关键修改：限制最大显示尺寸为800x600，避免占满界面）
                img = Image.fromarray(self.current_frame)
                max_display_width = 800  # 自定义最大宽度
                max_display_height = 600  # 自定义最大高度
                img.thumbnail((max_display_width, max_display_height), Image.LANCZOS)  # 按固定尺寸缩放

                # 原逻辑注释：不再使用窗口实际尺寸动态缩放
                # max_width = self.video_frame.winfo_width()
                # max_height = self.video_frame.winfo_height()
                # if max_width > 0 and max_height > 0:
                #     img.thumbnail((max_width, max_height), Image.LANCZOS)

                img_tk = ImageTk.PhotoImage(image=img)

                # Fix: Use self.video_frame instead of self.video_label
                self.video_frame.img_tk = img_tk  # Correct attribute
                self.video_frame.config(image=img_tk)  # Correct attribute
                self.root.after(30, self.update_frame)

    def detect_dogs(self):
        if self.current_frame is not None:
            # 转换为灰度图进行检测
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)

            # 检测狗脸
            dogs = self.dog_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # 在检测到的狗脸周围绘制矩形
            for (x, y, w, h) in dogs:
                cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.current_frame, "dog", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 更新状态
            if len(dogs) > 0:
                self.status_var.set(f"检测到 {len(dogs)} 只狗")

    def detect_cats(self):
        if self.current_frame is not None:
            # 转换为灰度图进行检测
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)

            # 检测猫脸
            cats = self.cat_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # 在检测到的猫脸周围绘制矩形
            for (x, y, w, h) in cats:
                cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(self.current_frame, "cat", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # 更新状态
            if len(cats) > 0:
                self.status_var.set(f"检测到 {len(cats)} 只猫")

    def toggle_dog_detection(self):
        if not self.is_detecting_dog:
            # 检查是否加载了分类器
            if self.dog_cascade.empty():
                messagebox.showerror("错误", "无法加载狗脸识别模型，请确保dog_cascade.xml文件存在")
                return

            self.is_detecting_dog = True
            self.detect_dog_btn.config(text="停止识别狗脸")
            self.status_var.set("正在识别狗脸...")
        else:
            self.is_detecting_dog = False
            self.detect_dog_btn.config(text="开始识别狗脸")
            if not self.is_detecting_cat:
                self.status_var.set("已停止识别")

    def toggle_cat_detection(self):
        if not self.is_detecting_cat:
            # 检查是否加载了分类器
            if self.cat_cascade.empty():
                messagebox.showerror("错误", "无法加载猫脸识别模型，请确保haarcascade_frontalcatface.xml文件存在")
                return

            self.is_detecting_cat = True
            self.detect_cat_btn.config(text="停止识别猫脸")
            self.status_var.set("正在识别猫脸...")
        else:
            self.is_detecting_cat = False
            self.detect_cat_btn.config(text="开始识别猫脸")
            if not self.is_detecting_dog:
                self.status_var.set("已停止识别")

    def capture_image(self):
        if self.is_camera_on and self.current_frame is not None:
            # 创建保存目录（如果不存在）
            save_dir = "captures"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/capture_{timestamp}.png"

            # 保存图像
            img_cv = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img_cv)

            self.status_var.set(f"图像已保存至: {filename}")

    def toggle_recording(self):
        if not self.is_recording:
            # 开始录制
            self.is_recording = True
            self.record_btn.config(text="停止录制")
            self.status_var.set("正在录制...")

            # 创建保存目录（如果不存在）
            save_dir = "videos"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/video_{timestamp}.mp4"

            # 获取视频的宽度和高度
            frame_width = int(self.cap.get(3))
            frame_height = int(self.cap.get(4))

            # 创建VideoWriter对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

            # 开始录制循环
            self.record_frame()
        else:
            # 停止录制
            self.is_recording = False
            self.record_btn.config(text="开始录制")
            self.status_var.set("录制已停止")

            if self.out:
                self.out.release()
                self.out = None

    def record_frame(self):
        if self.is_recording and self.is_camera_on:
            # 写入当前帧
            if self.current_frame is not None:
                frame_cv = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
                self.out.write(frame_cv)

            # 继续录制
            self.root.after(30, self.record_frame)

    def on_closing(self):
        # 释放资源并关闭窗口
        if self.is_camera_on:
            self.toggle_camera()

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
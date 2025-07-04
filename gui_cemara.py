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

        # 创建界面组件
        self.create_widgets()

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建视频显示区域
        self.video_frame = tk.Label(main_frame, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

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

        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪")
        self.status_label = tk.Label(main_frame, textvariable=self.status_var, font=self.font, fg="blue")
        self.status_label.pack(side=tk.BOTTOM, pady=5)

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
            self.status_var.set("摄像头已开启")

            # 开始更新视频帧
            self.update_frame()
        else:
            # 关闭摄像头
            self.is_camera_on = False
            self.camera_btn.config(text="打开摄像头")
            self.capture_btn.config(state=tk.DISABLED)

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

                # 添加录制指示
                if self.is_recording:
                    cv2.putText(self.current_frame, "录制中...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # 调整图像大小以适应窗口
                img = Image.fromarray(self.current_frame)
                img = img.resize((self.video_frame.winfo_width(), self.video_frame.winfo_height()),
                                 Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=img)

                # 更新显示
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk

            # 继续更新
            self.root.after(30, self.update_frame)

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
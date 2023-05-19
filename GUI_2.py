import tkinter as tk
import easygui
import cv2
import numpy as np
from Object_Detection import SSDMobileNet
from Tkinter.define import *
from tkinter import font
from PIL import Image, ImageTk


class VideoPlayer:
    def __init__(self, master):
        self.model = SSDMobileNet()
        self.stand = False
        self.sit = False
        self.lie = False
        self.font_text = font.Font(family="TkDefaultFont", size=13, weight=font.BOLD)
        self.master = master
        master.title("KidTrack")

        # Tạo khung chứa video
        self.video_frame = tk.Frame(master, width=576, height=320)
        # Load ảnh từ file
        image = Image.open("backgrond.png")
        # Chuyển ảnh sang định dạng phù hợp để hiển thị trên khung
        image = image.resize((576, 320))
        photo = ImageTk.PhotoImage(image)
        # Hiển thị ảnh trên khung
        label = tk.Label(self.video_frame, image=photo)
        label.image = photo
        label.grid(row=0, column=0, sticky="nsew")
        self.video_frame.grid(row=0, column=0, columnspan=2, sticky='nsew')

        # Tạo nút để mở video
        self.open_button = tk.Button(master, text="Open Video", font=self.font_text, activebackground=COLOR_GREEN, width=10, command=self.open_video)
        self.open_button.grid(row=3, column=1, sticky="s")

        # Tạo nút để thoát chương trình
        self.quit_button = tk.Button(master, text="Quit", font=self.font_text, activebackground=COLOR_GREEN, width=10, command=self.quit)
        self.quit_button.grid(row=4, column=1)

        # label
        self.my_label = tk.Label(root, text="Tùy Chọn", font=("Arial", 12))
        self.my_label.grid(row=1, column=0)

        # Tạo nút để thoát chương trình
        self.Stand_button = tk.Button(master, text="Stand", font=self.font_text, activebackground=COLOR_GREEN, width=10, command=self.set_Detect_Stand)
        self.Stand_button.grid(row=2, column=0)

        self.Lie_button = tk.Button(master, text="Lie", font=self.font_text, activebackground=COLOR_GREEN, width=10, command=self.set_Detect_Lie)
        self.Lie_button.grid(row=3, column=0)

        self.Sit_button = tk.Button(master, text="Sit", font=self.font_text, activebackground=COLOR_GREEN, width=10, command=self.set_Detect_Sit)
        self.Sit_button.grid(row=4, column=0)

        # Tạo nút reset các nhãn dự đoán
        self.Reset_button = tk.Button(master, text="Reset", font=self.font_text, activebackground=COLOR_GREEN, width=10, command=self.Reset_Label)
        self.Reset_button.grid(row=2, column=1)

    def set_Detect_Stand(self):
        self.stand = True
        # Vô hiệu hóa nút dừng video
        self.Stand_button.config(state=tk.DISABLED)
        print("stand: ", self.stand)

    def set_Detect_Lie(self):
        self.lie = True
        # Vô hiệu hóa nút dừng video
        self.Lie_button.config(state=tk.DISABLED)
        print("lie: ", self.lie)

    def set_Detect_Sit(self):
        self.sit = True
        # Vô hiệu hóa nút dừng video
        self.Sit_button.config(state=tk.DISABLED)
        print("sit: ", self.sit)

    def Reset_Label(self):
        print("Reset")
        self.lie = False
        # Vô hiệu hóa nút dừng video
        self.Lie_button.config(state=tk.NORMAL)
        print("lie: ", self.lie)
        self.sit = False
        # Vô hiệu hóa nút dừng video
        self.Sit_button.config(state=tk.NORMAL)
        print("sit: ", self.sit)
        self.stand = False
        # Vô hiệu hóa nút dừng video
        self.Stand_button.config(state=tk.NORMAL)
        print("stand: ", self.stand)

    def handle_left_click(self, event, x, y, flags, points):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    def draw_polygon(self, frame, points):
        for point in points:
            frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
        frame = cv2.polylines(frame, [np.int32(points)], False, (255, 0, 0), thickness=2)
        return frame

    def set_Model(self):
        if self.stand and self.lie and self.sit:
            self.model = SSDMobileNet(detect_class=['sit', 'lie', 'stand'])
        elif self.stand and self.sit:
            self.model = SSDMobileNet(detect_class=['sit', 'stand'])
        elif self.stand and self.lie:
            self.model = SSDMobileNet(detect_class=['lie', 'stand'])
        elif self.lie and self.sit:
            self.model = SSDMobileNet(detect_class=['sit', 'lie'])
        elif self.stand:
            self.model = SSDMobileNet(detect_class=['stand'])
        elif self.lie:
            self.model = SSDMobileNet(detect_class=['lie'])
        elif self.sit:
            self.model = SSDMobileNet(detect_class=['sit'])
        else:
            self.model = SSDMobileNet(detect_class=[''])

    def open_video(self):
        self.set_Model()
        detect = False
        points = []
        # Lấy đường dẫn video từ người dùng
        filename = easygui.fileopenbox(default='*.mp4', filetypes=['*.mp4'])
        # Kiểm tra xem người dùng đã chọn một tập tin hay chưa
        if filename:
            cap = cv2.VideoCapture(filename)
            # Mở video từ DroidCam
            # url = 'http://192.168.43.233:4747/video'
            # video = cv2.VideoCapture(url)
            while True:
                # Đọc từng khung hình từ video
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                frame = self.draw_polygon(frame, points)
                if detect:
                    frame = self.model.detect(frame=frame, points=points)
                # Thoát khỏi vòng lặp khi người dùng nhấn phím Esc
                key = cv2.waitKey(1)
                if key == 27:
                    break
                elif key == ord('d'):
                    points.append(points[0])
                    detect = True
                # Hiển thị khung hình trong khung chứa video
                cv2.imshow('KidTrack', frame)
                cv2.setMouseCallback("KidTrack", self.handle_left_click, points)

            # Giải phóng tài nguyên
            cap.release()
            cv2.destroyAllWindows()

    def quit(self):
        self.master.quit()


root = tk.Tk()
# Tính toán kích thước và vị trí của cửa sổ
window_width = 580
window_height = 450
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width - window_width) / 2)
y = int((screen_height - window_height) / 2)

# Thiết lập kích thước và vị trí cho cửa sổ
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

app = VideoPlayer(root)
root.mainloop()

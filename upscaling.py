import cv2
import time
import pandas as pd
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, Canvas, Scrollbar
from PIL import Image, ImageTk
from deep_sort_realtime.deepsort_tracker import DeepSort
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
import os
import threading


# --------------------- Load YOLOv8 Model ---------------------
model = YOLO("yolov8l-pose.pt")

# --------------------- Load Real-ESRGAN Upscaler ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "RealESRGAN_x4plus.pth"

upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=upscale_model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=torch.cuda.is_available(),
    device=device
)

# --------------------- Video Sources ---------------------
room_cameras = {
    "Room 1": r"room1_video.mp4",
    "Room 2": r"room2_video.mp4",
}

caps = {room: cv2.VideoCapture(path) for room, path in room_cameras.items()}
trackers = {room: DeepSort(max_age=50, n_init=5, max_iou_distance=0.3) for room in room_cameras}
data = pd.DataFrame(columns=["Room", "Timestamp", "People Count", "Status"])
monitoring_active = False

# --------------------- GUI SETUP ---------------------
window = tk.Tk()
window.title("AI-Based Multi-Room Monitoring System")
window.geometry("1600x1000")
window.configure(bg="black")

header = tk.Label(window, text="AI-Based Multi-Room Monitoring System", font=("Arial", 18, "bold"), bg="black", fg="white")
header.pack(pady=10)

button_frame = tk.Frame(window, bg="black")
button_frame.pack(pady=5)

def start_monitoring():
    global monitoring_active
    if not monitoring_active:
        monitoring_active = True
        header.config(text="Monitoring Started", fg="#00C853")
        threading.Thread(target=process_frames, daemon=True).start()


def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    for cap in caps.values():
        cap.release()
    data.to_excel("People_Count_Report.xlsx", index=False)
    header.config(text="Monitoring Stopped", fg="red")

def save_to_csv():
    data.to_csv("People_Count_Report.csv", index=False)
    header.config(text="📁 Report Saved as CSV!", fg="#00C853")

def exit_app():
    stop_monitoring()
    window.destroy()

buttons = [
    ("START", start_monitoring, "#00C853"),
    ("STOP", stop_monitoring, "#D50000"),
    ("SAVE TO CSV", save_to_csv, "#FF9100"),
    ("EXIT", exit_app, "#9C27B0")
]

for idx, (text, cmd, color) in enumerate(buttons):
    tk.Button(button_frame, text=text, command=cmd, bg=color, fg="white" if idx != 2 else "black",
              font=("Arial", 12, "bold"), width=15).grid(row=0, column=idx, padx=10, pady=5)

video_container = tk.Frame(window, bg="black")
video_container.pack(side="left", padx=10, pady=10, fill="both", expand=True)

canvas_frame = Canvas(video_container, bg="black")
scroll_y = Scrollbar(video_container, orient="vertical", command=canvas_frame.yview)
scroll_y.pack(side="right", fill="y")

canvas_frame.configure(yscrollcommand=scroll_y.set)
canvas_frame.pack(side="left", fill="both", expand=True)

video_frame = tk.Frame(canvas_frame, bg="black")
canvas_frame.create_window((0, 0), window=video_frame, anchor="nw")

canvas_list = {}
image_refs = {}  # <- Needed to prevent garbage collection

for room in room_cameras:
    tk.Label(video_frame, text=room, font=("Arial", 14, "bold"), fg="#FFD700", bg="black").pack()
    canvas = tk.Canvas(video_frame, width=800, height=600, bg="black", bd=2, relief="solid")
    canvas.pack(pady=5)
    canvas_list[room] = canvas
    image_refs[room] = None

def update_scroll_region():
    video_frame.update_idletasks()
    canvas_frame.config(scrollregion=canvas_frame.bbox("all"))

video_frame.bind("<Configure>", lambda e: update_scroll_region())

report_frame = tk.Frame(window, bg="black")
report_frame.pack(side="bottom", fill="x", padx=10, pady=10)

tk.Label(report_frame, text="📊 Occupancy Report", font=("Arial", 14, "bold"), fg="white", bg="black").pack(pady=5)

columns = ("Room", "Timestamp", "People Count", "Status")
report_tree = ttk.Treeview(report_frame, columns=columns, show="headings", height=10)
report_tree.pack(fill="both", expand=True)

for col in columns:
    report_tree.heading(col, text=col)
    report_tree.column(col, width=150)

def generate_report():
    report_tree.delete(*report_tree.get_children())
    for _, row in data.iterrows():
        report_tree.insert("", "end", values=(row["Room"], row["Timestamp"], row["People Count"], row["Status"]))

tk.Button(report_frame, text="Generate Report", command=generate_report, bg="#2962FF", fg="white",
          font=("Arial", 12, "bold")).pack(fill="x", padx=5, pady=5)

tk.Button(report_frame, text="Save Report", command=save_to_csv, bg="#FF9100", fg="black",
          font=("Arial", 12, "bold")).pack(fill="x", padx=5, pady=5)

# --------------------- Frame Processing ---------------------
def process_frames():
    global data
    if not monitoring_active:
        return

    confidence_threshold = 0.6

    for room_name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result, _ = upsampler.enhance(rgb_frame)
            frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Upscale Error in {room_name}]: {e}")

        results = model(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf.cpu().numpy())
                cls_id = int(box.cls.cpu().numpy())
                if model.names[cls_id] == "person" and conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, None))

        tracked_objects = trackers[room_name].update_tracks(detections, frame=frame)
        person_count = len([track for track in tracked_objects if track.is_confirmed()])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "Active" if person_count > 0 else "Inactive"

        cv2.putText(frame, f"People: {person_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Timestamp: {timestamp}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)

        frame = cv2.resize(frame, (800, 600))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        canvas_list[room_name].create_image(0, 0, anchor=tk.NW, image=img_tk)
        image_refs[room_name] = img_tk  # <-- Prevent garbage collection

        new_entry = pd.DataFrame([[room_name, timestamp, person_count, status]], columns=data.columns)
        data = pd.concat([data, new_entry], ignore_index=True)

    generate_report()
    window.after(30, process_frames)

# --------------------- Start GUI Loop ---------------------
window.mainloop()

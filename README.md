# 🧠 CCTV Room Occupancy Monitoring System

This is an AI-powered surveillance and analytics system designed to monitor multiple rooms using CCTV footage. It tracks room-wise occupancy, generates timestamps, and classifies room statuses (Working / Not Working). It also exports real-time people count and related logs into CSV files for analytics or reporting.

---

## 📌 Key Features

- 🎥 Multi-room occupancy detection using CCTV video
- 🧍 People detection and tracking using YOLOv8 + DeepSORT
- 🕒 Real-time timestamping of people count
- 📊 CSV log generation per room
- ✅ Room state classification (Working / Not Working)
- 📈 GUI for viewing stats, logs, and controlling rooms
- 🔍 Upscaling of CCTV footage using Real-ESRGAN for better visual quality
- 📁 Demo video: `Final CCTV Room Occupency.mkv` (tracked via Git LFS)

---

## 🖥️ System Architecture

```
          +--------------------+
          |   CCTV Footage     |
          +--------+-----------+
                   |
          +--------v-----------+
          | YOLOv8 + DeepSORT  |  ---> Person Detection & Tracking
          +--------+-----------+
                   |
          +--------v-----------+
          | Room Classifier     |  ---> Working / Not Working
          +--------+-----------+
                   |
          +--------v-----------+
          | CSV Logger         |  ---> Timestamp, Count, Status
          +--------+-----------+
                   |
          +--------v-----------+
          | GUI Dashboard      |
          +--------------------+
```

---

## 📂 Project Structure

```
CCTV-Room-Occupancy/
├── people_counter_gui/        # GUI interface using Tkinter
├── people_counter_final/      # Final detection + tracking script
├── People_Count_Report/       # Generated CSVs
├── Real-ESRGAN/               # For CCTV video upscaling
├── sort/                      # DeepSORT tracking
├── yolov8l.pt                 # YOLOv8 detection model
├── yolov8l-pose.pt            # YOLOv8 pose model
├── RealESRGAN_x4plus.pth      # Super-resolution model
├── room1_video.mp4            # Sample CCTV video
├── Final CCTV Room Occupency.mkv # Demo output video (Git LFS)
└── README.md
```

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Common libraries used:

- `ultralytics`
- `opencv-python`
- `torch`
- `numpy`
- `pandas`
- `deep_sort_realtime`
- `tkinter`
- `real-esrgan`

---

## 🧪 How to Run

### 1. Detect and Log Occupancy
```bash
python people_counter_final/main.py
```

### 2. Run GUI Interface
```bash
python people_counter_gui/app.py
```

### 3. Upscale CCTV Footage (Optional)
```bash
cd Real-ESRGAN/
python upscale_video.py --input ../room1_video.mp4 --output enhanced.mp4
```

---

## 📈 Output

- CSV logs with room ID, person count, and timestamps
- Annotated video with tracking boxes and IDs
- Summary stats via GUI
- Excel export supported

---

## 📽️ Demo

> Demo video included:

```
📁 Final CCTV Room Occupency.mkv

```
## 📽️ Demo Video

▶️ [Click here to watch the demo video](https://github.com/Solomon-Alexander1/CCTV-Room-Occupancy/releases/latest/download/Final%20CCTV%20Room%20Occupency.mkv)

> Since it is tracked via Git LFS, make sure Git LFS is installed:

```bash
git lfs install
git lfs pull
```

---

## 🧠 Models Used

| Model File              | Purpose              | Size       |
|-------------------------|----------------------|------------|
| yolov8l.pt              | Person detection     | ~83 MB     |
| yolov8l-pose.pt         | Pose detection       | ~85 MB     |
| RealESRGAN_x4plus.pth   | Video upscaling      | ~63 MB     |

---

## 🧹 Notes

- Large files (>50MB) are tracked using [Git Large File Storage](https://git-lfs.github.com).
- `yolov8x-face-lindevs.pt` was removed since it exceeded GitHub’s 100MB file limit.
- Do **not** commit heavy model files directly without LFS to avoid push errors.

---

## 🤝 Contribution

Contributions are welcome! If you'd like to:

- Report bugs
- Suggest new features
- Improve the code or documentation

Just fork the repo and open a pull request! 🙌



## 👤 Author

**Solomon Goodwin Alexander**  
🎓 B.Tech in Computer Science & Engineering (Data Science)  
🏫 St. Vincent Pallotti College of Engineering and Technology, Nagpur  
🔗 [LinkedIn](https://www.linkedin.com/in/solomon-alexander-184733170/)  
💻 [GitHub](https://github.com/Solomon-Alexander1)

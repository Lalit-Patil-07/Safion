````markdown
# Safion - PPE Detection System

Safion is a real-time Personal Protective Equipment (PPE) detection system designed to enhance workplace safety. It uses a deep learning model to monitor video streams from various sources, detect PPE compliance (hardhats, masks, safety vests), and log violations for review.

---

## ✨ Features

* **Live Multi-stream Detection**: Monitor multiple video feeds simultaneously from webcams, RTSP streams, or pre-recorded video files.
* **Real-time Violation Alerts**: Get instant visual feedback when a safety violation (e.g., "NO-Hardhat", "NO-Mask") is detected.
* **Comprehensive Violation Log**: Automatically records every violation with a timestamp, violation type, and a cropped image of the individual for evidence.
* **Identity Recognition**: Anonymously groups images of unknown violators. You can then assign a name to a group of images to track repeat offenders.
* **Zoom & Theater Mode**: Focus on a single stream for detailed monitoring or view a thumbnail grid of all active streams.
* **Easy Configuration**: A simple settings page to add, name, and manage RTSP video streams.

---

## 💻 Technology Stack

* **Backend**: Python, Flask, PyTorch.
* **AI Model**: YOLOv11 for object detection, `face_recognition` for identity clustering, and OpenVINO for performance optimization.
* **Real-time Video Processing**: OpenCV.
* **Frontend**: React, Tailwind CSS, Lucide React.
* **Containerization**: Docker.

---

## 🚀 Getting Started

The easiest way to get Safion running is by using the pre-built Docker image from the GitHub Container Registry.

### Option 1: Run with Docker (Recommended)

1.  **Pull the Docker image:**
    ```bash
    docker pull ghcr.io/lalit-patil-07/safion:latest
    ```

2.  **Run the container:**
    This command will start the application and make it accessible on port 5000.
    ```bash
    docker run -p 5000:5000 ghcr.io/lalit-patil-07/safion:latest
    ```

3.  **Access the application:**
    Open your web browser and go to `http://localhost:5000`.

### Option 2: Build Docker Image from Source

If you prefer to build the image yourself:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Lalit-Patil-07/Safion.git](https://github.com/Lalit-Patil-07/Safion.git)
    cd Safion
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t safion-ppe-system .
    ```

3.  **Run the container:**
    ```bash
    docker run -p 5000:5000 safion-ppe-system
    ```

---

## 🛠️ Local Development Setup

If you want to run the frontend and backend separately for development:

### Backend

1.  Navigate to the `backend` directory and install dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
2.  Run the Flask server:
    ```bash
    python app_server.py
    ```
    The backend will start on `http://localhost:5000`.

### Frontend

1.  In a new terminal, navigate to the `frontend` directory and install dependencies:
    ```bash
    cd frontend
    npm install
    ```
2.  Start the React development server:
    ```bash
    npm start
    ```
    The frontend will open at `http://localhost:3000`.

---

## 📂 Project Structure

````

ppe-detection-system/
├── backend/
│   ├── app\_server.py       \# Flask backend server
│   ├── known\_faces/        \# Stores images for recognized individuals
│   ├── requirements.txt    \# Python dependencies
│   └── violations\_images/  \# Stores images of violations
├── frontend/
│   ├── src/
│   │   └── App.js          \# Main React application component
│   └── package.json        \# Node.js dependencies
├── best.pt                 \# Trained YOLOv11 model weights
├── Dockerfile              \# Docker configuration for deployment
└── training\_file.ipynb     \# Jupyter notebook for model training

```

---

## 🧠 The AI Model

The detection model is a **YOLOv11n** model trained on the "Construction Site Safety" dataset. The model has been fine-tuned to detect the following classes:

* Hardhat
* Mask
* NO-Hardhat
* NO-Mask
* NO-Safety Vest
* Person
* Safety Cone
* Safety Vest
* Machinery
* Vehicle

After 30 epochs of training, the model achieved a **mAP50-95 of 56.1%** and a **mAP50 of 79.8%**.

---

## 📋 Future Improvements

* **Email/SMS Notifications**: Send real-time alerts for critical violations.
* **Enhanced Dashboard**: Add more analytics and visualizations for violation trends.
* **User Authentication**: Implement user roles and permissions.
* **More Robust Identity Management**: A more advanced system for managing known and unknown individuals.
```
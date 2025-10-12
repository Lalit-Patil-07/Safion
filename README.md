# Safion - PPE Detection System

Safion is a real-time Personal Protective Equipment (PPE) detection system designed to enhance workplace safety. It uses a deep learning model to monitor video streams from various sources, detect PPE compliance (hardhats, masks, safety vests), and log violations for review.

## ✨ Features

- **Live Multi-stream Detection**: Monitor multiple video feeds simultaneously from webcams, RTSP streams, or pre-recorded video files.
- **Real-time Violation Alerts**: Get instant visual feedback when a safety violation (e.g., "NO-Hardhat", "NO-Mask") is detected.
- **Comprehensive Violation Log**: Automatically records every violation with a timestamp, violation type, and a cropped image of the individual for evidence.
- **Identity Recognition**: Anonymously groups images of unknown violators. You can then assign a name to a group of images to track repeat offenders.
- **Zoom & Theater Mode**: Focus on a single stream for detailed monitoring or view a thumbnail grid of all active streams.
- **Easy Configuration**: A simple settings page to add, name, and manage RTSP video streams.

## 💻 Technology Stack

- **Backend**: Python, Flask, PyTorch
- **AI Model**: YOLOv8 for object detection and face_recognition for identity clustering
- **GPU Acceleration**: NVIDIA CUDA
- **Real-time Video Processing**: OpenCV
- **Frontend**: React, Tailwind CSS, Lucide React
- **Containerization**: Docker

## 🚀 Getting Started

There are two ways to run this application: using Docker for a quick and easy setup, or setting up a Local Development Environment for contributing to the code.

### Option 1: Docker Deployment (Recommended)

This is the simplest way to get the application running.

**Prerequisites:**
- Docker installed
- NVIDIA Container Toolkit installed for GPU support

**Build and Run the Container:**

```bash
docker build -t ppe-detection-system .
docker run -p 5000:5000 --gpus all ppe-detection-system
```

Open your browser and navigate to `http://localhost:5000`.

### Option 2: Local Development Setup

Follow these steps to run the application on your local machine with full GPU acceleration.

#### 1. Prerequisites

- An NVIDIA GPU
- Python 3.10+
- Node.js and npm
- Git

#### 2. Install NVIDIA CUDA Toolkit

- **Install NVIDIA Drivers**: Ensure you have the latest drivers for your GPU from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).
- **Install CUDA Toolkit**: Download and install the CUDA Toolkit version 12.1 or newer from the [NVIDIA Developer website](https://developer.nvidia.com/cuda-downloads). This is essential for GPU acceleration.

#### 3. Clone the Repository

```bash
git clone <repository_url>
cd ppe-detection-system
```

#### 4. Setup the Backend

Create and activate a Python virtual environment:

```bash
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**Install PyTorch**: Install the GPU-enabled version of PyTorch that matches your CUDA installation. Get the correct command from the [PyTorch official website](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Install Dlib**: Compiling dlib can be difficult. It's easier to install a pre-compiled version.

1. Go to the [Dlib_Windows_Python3.x repository](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
2. Find the `.whl` file that matches your Python version (e.g., `dlib-19.22.99-cp310-cp310-win_amd64.whl` for Python 3.10)
3. Download the file and install it using pip:

```bash
pip install "path/to/your/downloaded/dlib-file.whl"
```

Install the rest of the dependencies:

```bash
pip install -r backend/requirements.txt
```

#### 5. Setup the Frontend

Navigate to the frontend directory and install the Node.js dependencies:

```bash
cd frontend
npm install
```

#### 6. Run the Application

**Start the Backend Server**: In your first terminal, from the root directory, run:

```bash
python backend/app_server.py
```

**Start the Frontend Server**: Open a second terminal and from the frontend directory, run:

```bash
npm start
```

The application will open automatically in your browser at `http://localhost:3000`.

## 📂 Project Structure

```
ppe-detection-system/
├── backend/
│   ├── app_server.py       # Flask backend server
│   ├── known_faces/        # Stores images for recognized individuals
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   └── App.js          # Main React application component
│   └── package.json        # Node.js dependencies
├── best.pt                 # Trained YOLOv8 model weights
├── Dockerfile              # Docker configuration for deployment
└── training_file.ipynb     # Jupyter notebook for model training
```

## 🧠 The AI Model

The detection model is a YOLOv8n model trained on a custom dataset for PPE detection. The model is fine-tuned to detect the following classes:

- Hardhat
- Mask
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest
- Machinery
- Vehicle

## 📋 Future Improvements

- **Email/SMS Notifications**: Send real-time alerts for critical violations
- **Advanced Analytics**: A dashboard with charts and statistics on violation trends
- **Model Optimization**: Further training and optimization of the YOLO model for higher accuracy
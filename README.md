# Safion - PPE Detection System

Safion is a real-time Personal Protective Equipment (PPE) detection system designed to enhance workplace safety. It uses a deep learning model to monitor video streams from various sources, detect PPE compliance (hardhats, masks, safety vests), and log violations for review.

-----

## ✨ Features

  * **Live Multi-stream Detection**: Monitor multiple video feeds simultaneously from webcams, RTSP streams, or pre-recorded video files.
  * **Real-time Violation Alerts**: Get instant visual feedback when a safety violation (e.g., "NO-Hardhat", "NO-Mask") is detected.
  * **Comprehensive Violation Log**: Automatically records every violation with a timestamp, violation type, and a cropped image of the individual for evidence.
  * **Identity Recognition**: Anonymously groups images of unknown violators. You can then assign a name to a group of images to track repeat offenders.
  * **Zoom & Theater Mode**: Focus on a single stream for detailed monitoring or view a thumbnail grid of all active streams.
  * **Easy Configuration**: A simple settings page to add, name, and manage RTSP video streams.

-----

## 💻 Technology Stack

  * **Backend**: Python, Flask, PyTorch
  * **AI Model**: YOLOv11 for object detection, `face_recognition` for identity clustering, and OpenVINO for potential performance optimization.
  * **Real-time Video Processing**: OpenCV
  * **Frontend**: React, Tailwind CSS, Lucide React
  * **Deployment**: Docker

-----

## 🚀 Getting Started

The easiest way to get Safion running is by using Docker.

### Prerequisites

  * Docker installed on your system.
  * A webcam or RTSP stream URL for testing.

### Installation & Running with Docker

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd ppe-detection-system
    ```

2.  **Build the Docker image:**
    The `Dockerfile` handles everything from building the React frontend to installing the Python dependencies.

    ```bash
    docker build -t safion-ppe-system .
    ```

3.  **Run the Docker container:**
    This command maps port 5000 from the container to your local machine.

    ```bash
    docker run -p 5000:5000 safion-ppe-system
    ```

4.  **Access the Application:**
    Open your web browser and navigate to `http://localhost:5000`.

-----

## 🛠️ Local Development Setup

If you prefer to run the frontend and backend separately for development.

### Backend

1.  **Navigate to the backend directory:**

    ```bash
    cd backend
    ```

2.  **Install Python dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask server:**

    ```bash
    python app_server.py
    ```

    The backend server will start on `http://localhost:5000`.

### Frontend

1.  **Navigate to the frontend directory:**

    ```bash
    cd frontend
    ```

2.  **Install Node.js dependencies:**

    ```bash
    npm install
    ```

3.  **Start the React development server:**

    ```bash
    npm start
    ```

    The frontend will open automatically in your browser at `http://localhost:3000` and will connect to the backend server running on port 5000.

-----

## 📂 Project Structure

```
ppe-detection-system/
├── backend/
│   ├── app_server.py       # Flask backend server
│   ├── known_faces/        # Stores images for recognized individuals
│   ├── requirements.txt    # Python dependencies
│   └── violations_images/  # Stores images of violations
├── frontend/
│   ├── public/             # Public assets
│   ├── src/
│   │   └── App.js          # Main React application component
│   └── package.json        # Node.js dependencies
├── best.pt                 # Trained YOLOv11 model weights
├── Dockerfile              # Docker configuration for deployment
└── training_file.ipynb     # Jupyter notebook for model training
```

-----

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

-----

## 📋 Future Improvements

  * **Email/SMS Notifications**: Send real-time alerts for critical violations.
  * **Enhanced Dashboard**: Add more analytics and visualizations for violation trends.
  * **User Authentication**: Implement user roles and permissions.
  * **More Robust Identity Management**: A more advanced system for managing known and unknown individuals.
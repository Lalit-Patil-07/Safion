"""
Entry point for development.
For production use gunicorn:
    gunicorn "app:create_app()" -w 1 -b 0.0.0.0:5000 --timeout 120
Note: -w 1 (single worker) is intentional — the YOLO model and face pipeline
are in-process singletons.  Scale horizontally with multiple containers,
not multiple gunicorn workers in the same process.
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

All components interact via API calls and are managed centrally by the web app/dashboard.
## Getting Started
### Prerequisites
- Python 3.11+
- OpenCV, InsightFace, FAISS, Flask, SQLite, and other requirements in `requirements.txt`
- Webcam/Camera
- For geolocation: enable browser location permissions

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/DemigodKushal/CHAI.git
    cd CHAI
    ```
2. Set up a Python virtual environment and activate it:
    ```
    python -m venv venv
    venv\Scripts\activate
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Download pre-trained InsightFace weights as instructed in `README` or documentation.

### Running the App

1. Start the server:
    ```
    python app.py
    ```
2. Open your browser and navigate to `http://localhost:5000`
3. Enroll users using the web dashboard, then mark attendance.

### Folder Structure

- `main.py`: Application entry point
- `app.py`: Web server and dashboard
- `attendance_service.py`, `database_service.py`, etc.: Core modules
- `templates/`: HTML frontend
- `static/`: CSS, JS, image assets

---

## How Liveness Detection Works

- A flash pattern is displayed on the screen.
- Parameters like brightness increase, color variance, edge density, and nonuniformity are analyzed.
- The system checks thresholds to differentiate between live faces and spoof attempts.

---

## Contributors

- Kushal Shrivastava (24124025)
- Mayank Rathore (24124028)
- Panshull Choudhary (24124034)
- Parv Gandhi (24124035)

**IIT BHU - Department Mathematical Scieences**

## License

This project is released for educational purposes. Please review code before any commercial use.

## Reference Links

- [InsightFace](https://github.com/deepinsight/insightface)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenCV](https://opencv.org)
- [SQLite](https://www.sqlite.org/docs.html)


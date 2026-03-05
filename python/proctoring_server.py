"""
Exam Flow Proctoring Server
============================
FastAPI server that accepts webcam frames (base64 JPEG) and detects
number of faces using MediaPipe FaceDetector (new Task API, mediapipe >= 0.10).

Run with:
    uvicorn proctoring_server:app --port 8000 --reload

Install deps:
    pip install fastapi uvicorn opencv-python-headless mediapipe numpy python-multipart
"""

import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request
import os
import ast

app = FastAPI(title="Exam Proctoring Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Download the MediaPipe face detection model if not cached ─────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading face detection model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

# ─── Build the FaceDetector (IMAGE mode — stateless, per-request) ─────────────
_options = FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.35,
)
_detector = FaceDetector.create_from_options(_options)

_face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)


# ─── Pydantic models ──────────────────────────────────────────────────────────

class FramePayload(BaseModel):
    # Base64-encoded JPEG or PNG frame (data URL or raw base64)
    frame: str

class AnalysisResult(BaseModel):
    face_count: int
    status: str   # "ok" | "no_face" | "multiple_faces" | "error"
    message: str

    model_config = {"populate_by_name": True}

class CodePayload(BaseModel):
    code: str

class CodeAnalysisResult(BaseModel):
    status: str
    message: str
    has_for_loop: bool = False
    has_while_loop: bool = False
    has_recursion: bool = False
    has_restricted_imports: bool = False
    function_count: int = 0
    class_count: int = 0

    model_config = {"populate_by_name": True}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "server": "Exam Proctoring Server"}


@app.post("/analyze", response_model=AnalysisResult)
def analyze_frame(payload: FramePayload):
    """
    Analyze a webcam frame for face presence.
    Accepts a base64-encoded image (with or without the data-URL prefix).
    """
    try:
        if not payload.frame:
            return AnalysisResult(face_count=0, status="error", message="Empty frame payload")

        # Strip data-URL prefix if present
        raw_b64 = payload.frame
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        # Decode base64 → numpy array → OpenCV BGR image
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return AnalysisResult(face_count=0, status="error", message="Could not decode image frame")

        # Convert BGR → RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Wrap in MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Run detection
        result = _detector.detect(mp_image)
        face_count = len(result.detections) if result.detections else 0

        if face_count == 0 and not _face_cascade.empty():
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = _face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(40, 40),
            )
            face_count = len(faces)

        if face_count == 0:
            return AnalysisResult(face_count=0, status="no_face", message="No face detected in the frame")
        elif face_count == 1:
            return AnalysisResult(face_count=1, status="ok", message="Face detected — student present")
        else:
            return AnalysisResult(face_count=face_count, status="multiple_faces",
                                  message=f"{face_count} faces detected in the frame")

    except Exception as e:
        return AnalysisResult(face_count=-1, status="error", message=f"Analysis error: {str(e)}")

@app.post("/analyze_code", response_model=CodeAnalysisResult)
async def analyze_code(payload: CodePayload):
    """
    Parses submitted Python code using the built-in `ast` module.
    Detects loops, recursion, function/class counts, and restricted imports.
    """
    code = payload.code

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CodeAnalysisResult(
            status="error",
            message=f"Syntax error in code: {str(e)}"
        )

    has_for_loop = False
    has_while_loop = False
    has_recursion = False
    has_restricted_imports = False
    function_count = 0
    class_count = 0

    restricted_modules = {'os', 'sys', 'subprocess'}

    class ASTVisitor(ast.NodeVisitor):
        def __init__(self):
            self.has_for_loop = False
            self.has_while_loop = False
            self.has_recursion = False
            self.has_restricted_imports = False
            self.function_count = 0
            self.class_count = 0

        def visit_For(self, node):
            self.has_for_loop = True
            self.generic_visit(node)

        def visit_While(self, node):
            self.has_while_loop = True
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            self.function_count += 1
            # Check for recursion
            func_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and hasattr(child.func, 'id'):
                    if getattr(child.func, 'id') == func_name:
                        self.has_recursion = True
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            self.class_count += 1
            self.generic_visit(node)

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name.split('.')[0] in restricted_modules:
                    self.has_restricted_imports = True
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module and node.module.split('.')[0] in restricted_modules:
                self.has_restricted_imports = True
            self.generic_visit(node)

    visitor = ASTVisitor()
    visitor.visit(tree)

    return CodeAnalysisResult(
        status="ok",
        message="Code analyzed successfully",
        has_for_loop=visitor.has_for_loop,
        has_while_loop=visitor.has_while_loop,
        has_recursion=visitor.has_recursion,
        has_restricted_imports=visitor.has_restricted_imports,
        function_count=visitor.function_count,
        class_count=visitor.class_count
    )

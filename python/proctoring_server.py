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

from mediapipe.tasks.python.vision import ImageEmbedder, ImageEmbedderOptions, FaceLandmarker, FaceLandmarkerOptions

# ─── Build the ImageEmbedder (Used for face matching) ──────────────────────
EMBEDDER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobilenet_v3_small.tflite")
EMBEDDER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"

if not os.path.exists(EMBEDDER_MODEL_PATH):
    print(f"Downloading image embedder model to {EMBEDDER_MODEL_PATH} ...")
    urllib.request.urlretrieve(EMBEDDER_MODEL_URL, EMBEDDER_MODEL_PATH)
    print("Model downloaded successfully.")

_embedder_options = ImageEmbedderOptions(
    base_options=mp_python.BaseOptions(model_asset_path=EMBEDDER_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
)
_embedder = ImageEmbedder.create_from_options(_embedder_options)

# ─── Build the FaceLandmarker (Used for geometric proportions) ─────────────
LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(LANDMARKER_MODEL_PATH):
    print(f"Downloading face landmarker model to {LANDMARKER_MODEL_PATH} ...")
    urllib.request.urlretrieve(LANDMARKER_MODEL_URL, LANDMARKER_MODEL_PATH)
    print("Model downloaded successfully.")

_landmarker_options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
)
_landmarker = FaceLandmarker.create_from_options(_landmarker_options)

_face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)


# ─── Pydantic models ──────────────────────────────────────────────────────────

class FramePayload(BaseModel):
    # Base64-encoded JPEG or PNG frame (data URL or raw base64)
    frame: str
    reference_image: str = None  # Optional signup photo for identity verification

class AnalysisResult(BaseModel):
    face_count: int
    status: str   # "ok" | "no_face" | "multiple_faces" | "face_mismatch" | "error"
    message: str
    similarity: float = 0.0

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
    print("DEBUG: Health check received from frontend")
    return {"status": "ok", "server": "Exam Proctoring Server"}


@app.post("/analyze", response_model=AnalysisResult)
def analyze_frame(payload: FramePayload):
    print(f"DEBUG: Frame received for analysis. Ref image present: {bool(payload.reference_image)}")
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
            print(f"RESULT: no_face")
            return AnalysisResult(face_count=0, status="no_face", message="No face detected in the frame")
        elif face_count > 1:
            print(f"RESULT: multiple_faces ({face_count})")
            return AnalysisResult(face_count=face_count, status="multiple_faces",
                                  message=f"{face_count} faces detected in the frame")

        # Exactly 1 face detected, now check identity if reference_image is provided
        if payload.reference_image:
            try:
                # Decode reference image
                ref_b64 = payload.reference_image
                if "," in ref_b64:
                    ref_b64 = ref_b64.split(",", 1)[1]
                ref_bytes = base64.b64decode(ref_b64)
                ref_np = np.frombuffer(ref_bytes, dtype=np.uint8)
                ref_bgr = cv2.imdecode(ref_np, cv2.IMREAD_COLOR)
                
                if ref_bgr is not None:
                    print(f"DEBUG: Reference image received and decoded. Shape: {ref_bgr.shape}")
                    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                    mp_ref_image_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_rgb)
                    
                    # Detect face in reference image to get a crop (much better for matching)
                    ref_detect_result = _detector.detect(mp_ref_image_full)
                    if ref_detect_result.detections:
                        ref_bbox = ref_detect_result.detections[0].bounding_box
                        rh, rw, _ = ref_bgr.shape
                        rx1 = max(0, int(ref_bbox.origin_x))
                        ry1 = max(0, int(ref_bbox.origin_y))
                        rx2 = min(rw, rx1 + int(ref_bbox.width))
                        ry2 = min(rh, ry1 + int(ref_bbox.height))
                        ref_face_crop_bgr = ref_bgr[ry1:ry2, rx1:rx2]
                        ref_face_crop_rgb = cv2.cvtColor(ref_face_crop_bgr, cv2.COLOR_BGR2RGB)
                        mp_ref_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_face_crop_rgb)
                        print(f"DEBUG: Reference face cropped. Shape: {ref_face_crop_bgr.shape}")
                    else:
                        # Fallback to full image if no face detected in ref (rare)
                        mp_ref_image = mp_ref_image_full
                        print("DEBUG: No face detected in reference image, using full image.")
                    
                    # For the current frame, we have exactly 1 face. Crop it.
                    if not result.detections:
                         print("DEBUG: Face detected by fallback but not by MediaPipe detector. Skipping identity match.")
                         return AnalysisResult(face_count=1, status="ok", message="Identity match skipped (detection fallback)")
                    
                    bbox = result.detections[0].bounding_box
                    h, w, _ = img_bgr.shape
                    x1 = max(0, int(bbox.origin_x))
                    y1 = max(0, int(bbox.origin_y))
                    x2 = min(w, x1 + int(bbox.width))
                    y2 = min(h, y1 + int(bbox.height))
                    
                    face_crop_bgr = img_bgr[y1:y2, x1:x2]
                    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                    mp_face_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop_rgb)
                    print(f"DEBUG: Current frame face cropped. Shape: {face_crop_bgr.shape}")
                    
                    # ─── Geometric Check (Face Mesh) ───
                    def get_face_ratio(mp_img):
                        lm_result = _landmarker.detect(mp_img)
                        if not lm_result.face_landmarks:
                            return None
                        lms = lm_result.face_landmarks[0]
                        # Eye distance (centers)
                        lex = (lms[159].x + lms[145].x) / 2
                        ley = (lms[159].y + lms[145].y) / 2
                        rex = (lms[386].x + lms[374].x) / 2
                        rey = (lms[386].y + lms[374].y) / 2
                        eye_dist = ((lex-rex)**2 + (ley-rey)**2)**0.5
                        # Face width (edges)
                        fw = ((lms[234].x - lms[454].x)**2 + (lms[234].y - lms[454].y)**2)**0.5
                        return eye_dist / fw if fw > 0 else None

                    ref_ratio = get_face_ratio(mp_ref_image)
                    frame_ratio = get_face_ratio(mp_face_crop)
                    
                    geo_sim = 1.0
                    if ref_ratio and frame_ratio:
                        # Ratios are typically ~0.4. A 0.04 diff is a 10% change.
                        diff = abs(ref_ratio - frame_ratio)
                        geo_sim = max(0.0, 1.0 - (diff * 5.0)) # 0.1 diff -> 0.5 sim
                        print(f"DEBUG: Geometric Ratio Diff: {diff:.4f}, Geo Sim: {geo_sim:.4f}")

                    # Extract embeddings
                    ref_result = _embedder.embed(mp_ref_image)
                    frame_result = _embedder.embed(mp_face_crop)
                    
                    if ref_result.embeddings and frame_result.embeddings:
                        embed_sim = ImageEmbedder.cosine_similarity(
                            ref_result.embeddings[0], frame_result.embeddings[0]
                        )
                        
                        # Hybrid Similarity
                        # 70% Embedder, 30% Geometric
                        similarity = (0.7 * embed_sim) + (0.3 * geo_sim)
                        print(f"DEBUG: Embed Sim: {embed_sim:.4f}, Final Hybrid Similarity: {similarity:.4f}")
                        
                        # Tolerant Threshold (0.75) for better real-world usage
                        MATCH_THRESHOLD = 0.75 
                        
                        if similarity < MATCH_THRESHOLD:
                            return AnalysisResult(
                                face_count=1, 
                                status="face_mismatch", 
                                message=f"Identity mismatch! (Confidence: {similarity:.2f})",
                                similarity=similarity
                            )
                        
                        return AnalysisResult(
                            face_count=1, 
                            status="ok", 
                            message="Face matched successfully", 
                            similarity=similarity
                        )
                    else:
                        print("DEBUG: Could not generate embeddings for one of the images.")
                else:
                    print("DEBUG: Decoding reference image failed (None result).")
            except Exception as embed_err:
                print(f"DEBUG: Analysis error: {embed_err}")
                import traceback
                traceback.print_exc()
        
        return AnalysisResult(face_count=1, status="ok", message="Face detected — student present")

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

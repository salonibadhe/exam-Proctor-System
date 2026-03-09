import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { examAPI, questionAPI, resultAPI } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { useToast } from "@/hooks/use-toast";
import { Clock, Send, Play, Code2, Camera, CameraOff, AlertTriangle, CheckCircle2, XCircle, BookOpen } from "lucide-react";
import Editor from '@monaco-editor/react';

// ─── Types ────────────────────────────────────────────────────────────────────

type Exam = {
  _id: string;
  title: string;
  description?: string;
  examCode: string;
  examDate: string;
  durationMinutes: number;
  teacherId: string;
  createdAt: string;
};

type TestCase = {
  _id?: string;
  input: string;
  expectedOutput?: string;
  isHidden?: boolean;
  explanation?: string;
};

type Question = {
  _id: string;
  examId: string;
  questionType: 'mcq' | 'coding';
  questionText: string;
  points?: number;
  optionA?: string;
  optionB?: string;
  optionC?: string;
  optionD?: string;
  problemStatement?: string;
  inputFormat?: string;
  outputFormat?: string;
  constraints?: string;
  testCases?: TestCase[];
  timeLimit?: number;
  starterCode?: {
    javascript?: string;
    python?: string;
    cpp?: string;
    java?: string;
  };
  createdAt: string;
};

// Violation types we track
export type ViolationType =
  | 'tab_switch'
  | 'copy_paste'
  | 'keyboard_shortcut'
  | 'no_face'
  | 'multiple_faces'
  | 'face_mismatch';

type ViolationLog = Record<ViolationType, number>;

const PYTHON_SERVER = 'http://127.0.0.1:8000';
const FRAME_INTERVAL_MS = 5000; // Analyze a frame every 5 seconds
const VIDEO_READY_STATE_CURRENT_DATA = 2;

// ─── Component ────────────────────────────────────────────────────────────────

const TakeExam = () => {
  const { examId } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [user, setUser] = useState<any>(null);
  const [exam, setExam] = useState<Exam | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [codeSubmissions, setCodeSubmissions] = useState<Record<string, string>>({});
  const [selectedLanguages, setSelectedLanguages] = useState<Record<string, string>>({});
  const [runningCode, setRunningCode] = useState<string | null>(null);
  const [codeOutput, setCodeOutput] = useState<Record<string, any>>({});
  const [customInputs, setCustomInputs] = useState<Record<string, string>>({});
  const [useCustomInput, setUseCustomInput] = useState<Record<string, boolean>>({});
  const [submitting, setSubmitting] = useState(false);
  const [timeLeft, setTimeLeft] = useState<number>(0);
  const [isFaceLocked, setIsFaceLocked] = useState(false);
  const [lastMatchScore, setLastMatchScore] = useState<number | null>(null);

  // Violation tracking — per type
  const [violations, setViolations] = useState<ViolationLog>({
    tab_switch: 0,
    copy_paste: 0,
    keyboard_shortcut: 0,
    no_face: 0,
    multiple_faces: 0,
    face_mismatch: 0,
  });

  // Camera proctoring
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [pythonServerUp, setPythonServerUp] = useState<boolean | null>(null); // null = checking
  const pythonServerUpRef = useRef<boolean | null>(null);
  const userRef = useRef<any>(null);
  const isFaceLockedRef = useRef(false);
  const consecutiveMismatchesRef = useRef(0);

  // Use a ref for violations so interval callbacks always read the latest value
  const violationsRef = useRef<ViolationLog>(violations);
  const lastViolationTimeRef = useRef<Record<ViolationType, number>>({
    tab_switch: 0,
    copy_paste: 0,
    keyboard_shortcut: 0,
    no_face: 0,
    multiple_faces: 0,
    face_mismatch: 0,
  });

  useEffect(() => {
    violationsRef.current = violations;
  }, [violations]);

  // ─── Init ────────────────────────────────────────────────────────────────

  useEffect(() => {
    checkUser();
    fetchExam();
    fetchQuestions();
    checkPythonServer();
    startCamera();

    return () => {
      // Cleanup on unmount
      stopCamera();
    };
  }, [examId]);

  // ─── Python Server Check ─────────────────────────────────────────────────

  const checkPythonServer = async () => {
    try {
      const res = await fetch(`${PYTHON_SERVER}/health`, { signal: AbortSignal.timeout(3000) });
      const data = await res.json();
      const isUp = data.status === 'ok';
      setPythonServerUp(isUp);
      pythonServerUpRef.current = isUp;
    } catch {
      setPythonServerUp(false);
      pythonServerUpRef.current = false;
    }
  };

  // ─── Camera Setup ────────────────────────────────────────────────────────

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      streamRef.current = stream;
      setCameraActive(true);
      setCameraError(null);
    } catch (err: any) {
      setCameraError('Camera access denied. Proctoring may be limited.');
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  };

  // ─── Frame Capture & Analysis ────────────────────────────────────────────

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;
    if (!pythonServerUpRef.current) return; // Don't analyze if server is down

    const video = videoRef.current;
    if (video.readyState < VIDEO_READY_STATE_CURRENT_DATA) return;
    if (!video.videoWidth || !video.videoHeight) return;

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64Frame = canvas.toDataURL('image/jpeg', 0.85);

    try {
      const response = await fetch(`${PYTHON_SERVER}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame: base64Frame,
          reference_image: userRef.current?.profilePhoto
        }),
        signal: AbortSignal.timeout(4000),
      });

      if (!response.ok) {
        console.error('Python server error:', await response.text());
        return;
      }
      const result = await response.json();
      console.log('Frame analysis result:', result);

      if (result.status === 'no_face') {
        handleViolation('no_face', 'No face detected! Please stay in front of the camera.');
      } else if (result.status === 'multiple_faces') {
        const count = result.faceCount ?? result.face_count ?? 2;
        handleViolation('multiple_faces', `Multiple faces detected (${count})! Only you should be visible.`);
      } else if (result.status === 'face_mismatch') {
        consecutiveMismatchesRef.current += 1;
        console.log(`DEBUG: Consecutive Mismatches: ${consecutiveMismatchesRef.current}`);

        // Only trigger lock after 3 consecutive failures (approx 15 seconds)
        if (consecutiveMismatchesRef.current >= 3) {
          handleViolation('face_mismatch', 'Identity mismatch! Face does not match signup photo.');
          setIsFaceLocked(true);
          isFaceLockedRef.current = true;
        }
        setLastMatchScore(result.similarity);
      } else if (result.status === 'ok' && result.similarity !== undefined) {
        // Reset counter on success
        consecutiveMismatchesRef.current = 0;

        // If it was locked and we get an 'ok' with good similarity, unlock
        if (isFaceLockedRef.current && result.similarity >= 0.75) {
          setIsFaceLocked(false);
          isFaceLockedRef.current = false;
          toast({
            title: 'Identity Verified',
            description: 'Success! You may continue the exam.',
            className: 'bg-green-500 text-white',
          });
        }
        setLastMatchScore(result.similarity);
      }
      else if (result.status === 'error') {
        console.error('Python Analysis Error:', result.message);
      }
    } catch (err) {
      console.error('Frame capture/network error:', err);
      // Silently ignore network timeout errors for frame analysis — exam should not break
    }
  }, []);

  // Update the interval callback whenever camera Active changes
  useEffect(() => {
    if (cameraActive) {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = setInterval(captureAndAnalyze, FRAME_INTERVAL_MS);
    }
  }, [cameraActive]);

  // ─── Proctoring: Tab Switching ───────────────────────────────────────────

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        handleViolation('tab_switch', 'Tab switching is not allowed!');
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  // ─── Proctoring: Copy/Paste ──────────────────────────────────────────────

  useEffect(() => {
    const prevent = (e: Event) => {
      e.preventDefault();
      handleViolation('copy_paste', 'Copy/Paste is not allowed during the exam!');
    };
    window.addEventListener('copy', prevent);
    window.addEventListener('cut', prevent);
    window.addEventListener('paste', prevent);
    window.addEventListener('contextmenu', prevent);
    return () => {
      window.removeEventListener('copy', prevent);
      window.removeEventListener('cut', prevent);
      window.removeEventListener('paste', prevent);
      window.removeEventListener('contextmenu', prevent);
    };
  }, []);

  // ─── Proctoring: Keyboard Shortcuts ──────────────────────────────────────

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && ['c', 'v', 'x', 'a'].includes(e.key)) {
        e.preventDefault();
        handleViolation('keyboard_shortcut', `Keyboard shortcut Ctrl+${e.key.toUpperCase()} is disabled!`);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // ─── Violation Handler (per type) ────────────────────────────────────────

  const handleViolation = (type: ViolationType, message: string) => {
    const now = Date.now();
    // 15-second cooldown for the exact same violation type to prevent spam/annoyance
    if (now - lastViolationTimeRef.current[type] < 15000) {
      return;
    }
    lastViolationTimeRef.current[type] = now;

    // Read from the ref so we always have the latest counts, 
    // even if called from a stale effect closure like captureAndAnalyze
    const updatedViolations = { ...violationsRef.current, [type]: violationsRef.current[type] + 1 };

    setViolations(updatedViolations);
    // Also explicitly update the ref immediately so subsequent calls in the same render cycle see it
    violationsRef.current = updatedViolations;

    toast({
      variant: 'destructive',
      title: '⚠️ Exam Violation',
      description: message,
      duration: 5000,
    });
  };

  // ─── Timer ───────────────────────────────────────────────────────────────

  useEffect(() => {
    if (timeLeft <= 0 || isFaceLocked) return;
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        const newVal = prev <= 1 ? 0 : prev - 1;
        // Also persist to localStorage
        if (examId) localStorage.setItem(`exam_timer_${examId}`, newVal.toString());
        if (newVal === 0) handleSubmit();
        return newVal;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [timeLeft, isFaceLocked, examId]);

  // ─── Data Fetching ───────────────────────────────────────────────────────

  const checkUser = async () => {
    const userData = localStorage.getItem('user');
    if (!userData) { navigate('/auth'); return; }
    const parsedUser = JSON.parse(userData);
    console.log('DEBUG: Current user has profile photo:', !!parsedUser?.profilePhoto);
    setUser(parsedUser);
    userRef.current = parsedUser;
  };

  const fetchExam = async () => {
    try {
      const data = await examAPI.getById(examId!);
      setExam(data);
      // Try to load from localStorage first for persistence
      const savedTime = localStorage.getItem(`exam_timer_${examId}`);
      if (savedTime) {
        setTimeLeft(parseInt(savedTime, 10));
      } else {
        setTimeLeft(data.durationMinutes * 60);
      }
    } catch {
      navigate('/student/dashboard');
    }
  };

  const fetchQuestions = async () => {
    try {
      const data = await questionAPI.getByExamId(examId!);
      setQuestions(data);
      const initialCode: Record<string, string> = {};
      const initialLanguages: Record<string, string> = {};
      data.forEach((q: Question) => {
        if (q.questionType === 'coding') {
          initialLanguages[q._id] = 'javascript';
          initialCode[q._id] = q.starterCode?.javascript || '// Write your JavaScript code here';
        }
      });
      setCodeSubmissions(initialCode);
      setSelectedLanguages(initialLanguages);
    } catch (error) {
      console.error('Error fetching questions:', error);
    }
  };

  // ─── Code Execution ──────────────────────────────────────────────────────

  const handleLanguageChange = (questionId: string, language: string) => {
    const question = questions.find(q => q._id === questionId);
    if (!question) return;
    const starters: Record<string, string> = {
      javascript: question.starterCode?.javascript || '// Write your JavaScript code here',
      python: question.starterCode?.python || '# Write your Python code here\n',
      cpp: question.starterCode?.cpp || '// Write your C++ code here\n#include <iostream>\nusing namespace std;\n\nint main() {\n    return 0;\n}',
      java: question.starterCode?.java || '// Write your Java code here\npublic class Main {\n    public static void main(String[] args) {\n        \n    }\n}',
    };
    setSelectedLanguages({ ...selectedLanguages, [questionId]: language });
    setCodeSubmissions({ ...codeSubmissions, [questionId]: starters[language] || '' });
  };

  const handleRunCode = async (questionId: string) => {
    setRunningCode(questionId);
    const question = questions.find(q => q._id === questionId);
    const code = codeSubmissions[questionId];
    const language = selectedLanguages[questionId] || 'javascript';
    const isCustom = useCustomInput[questionId];
    const customInput = customInputs[questionId] || '';

    if (!question || !code) {
      toast({ variant: 'destructive', title: 'Error', description: 'No code to run' });
      setRunningCode(null);
      return;
    }

    try {
      if (isCustom) {
        const result = await questionAPI.runCode({ code, language, input: customInput });
        setCodeOutput({ ...codeOutput, [questionId]: { type: 'single', ...result } });
      } else {
        const results = await (questionAPI as any).runTestCases({ code, language, questionId });
        setCodeOutput({ ...codeOutput, [questionId]: { type: 'batch', results } });
      }
      toast({ title: 'Execution complete', description: 'Check results below' });
    } catch (error: any) {
      toast({ variant: 'destructive', title: 'Error', description: error.response?.data?.message || error.message });
      setCodeOutput({ ...codeOutput, [questionId]: { type: 'error', error: error.message } });
    } finally {
      setRunningCode(null);
    }
  };

  // ─── Submit Exam ─────────────────────────────────────────────────────────

  const handleSubmit = async () => {
    if (submitting) return;
    setSubmitting(true);
    stopCamera();

    // Build violations array (only non-zero counts)
    const violationsList = (Object.entries(violationsRef.current) as [ViolationType, number][])
      .map(([type, count]) => ({ type, count }));

    try {
      const responses = questions.map(q => {
        if (q.questionType === 'mcq') {
          return { questionId: q._id, selectedAnswer: answers[q._id] || '' };
        } else {
          return {
            questionId: q._id,
            code: codeSubmissions[q._id] || '',
            language: selectedLanguages[q._id] || 'javascript',
          };
        }
      });

      const result = await resultAPI.submit({
        examId: examId!,
        responses,
        violations: violationsList,
      });

      // Clear the timer from localStorage on successful submission
      localStorage.removeItem(`exam_timer_${examId}`);

      toast({
        title: 'Exam submitted!',
        description: `You scored ${result.result.score}/${result.result.totalQuestions} (${result.percentage}%)`,
      });
      navigate('/student/dashboard');
    } catch (error: any) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: error.response?.data?.message || error.message,
      });
      setSubmitting(false);
    }
  };

  // ─── Helpers ─────────────────────────────────────────────────────────────

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const totalViolations = Object.values(violations).reduce((a, b) => a + b, 0);

  // ─── Loading State ───────────────────────────────────────────────────────

  if (!exam || questions.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Loading exam...</div>
      </div>
    );
  }

  const answeredCount = questions.filter(q => {
    if (q.questionType === 'mcq') return answers[q._id];
    return codeSubmissions[q._id] && codeSubmissions[q._id].trim().length > 0;
  }).length;
  const allAnswered = answeredCount === questions.length;

  // ─── Face Lock Overlay ───────────────────────────────────────────────────

  const FaceLockOverlay = () => {
    if (!isFaceLocked) return null;

    return (
      <div className="fixed inset-0 z-[100] bg-slate-900/95 backdrop-blur-md flex items-center justify-center p-6">
        <Card className="max-w-4xl w-full border-2 border-red-500/50 shadow-2xl shadow-red-500/10">
          <CardHeader className="text-center pb-2">
            <div className="mx-auto w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-4">
              <AlertTriangle className="w-8 h-8 text-red-600" />
            </div>
            <CardTitle className="text-3xl font-bold text-red-600">Identity Verification Required</CardTitle>
            <CardDescription className="text-lg font-medium text-slate-400">
              Exam is paused. Please ensure you are looking at the camera.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
              {/* Reference Image */}
              <div className="space-y-3">
                <p className="text-xs font-bold text-slate-500 uppercase text-center">Signup Photo</p>
                <div className="relative aspect-square rounded-2xl overflow-hidden border-4 border-slate-700 bg-slate-800">
                  {user?.profilePhoto ? (
                    <img src={user.profilePhoto} alt="Reference" className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-500">
                      No signup photo found
                    </div>
                  )}
                  <div className="absolute top-3 right-3 bg-indigo-600 p-2 rounded-lg shadow-lg">
                    <CheckCircle2 className="w-5 h-5 text-white" />
                  </div>
                </div>
              </div>

              {/* Live Feedback */}
              <div className="space-y-3">
                <p className="text-xs font-bold text-slate-500 uppercase text-center">Live Webcam Feed</p>
                <div className="relative aspect-square rounded-2xl overflow-hidden border-4 border-red-500 ring-4 ring-red-500/20 bg-black">
                  <video
                    autoPlay
                    muted
                    playsInline
                    className="w-full h-full object-cover grayscale-[0.5]"
                    ref={(el) => {
                      if (el && streamRef.current && el.srcObject !== streamRef.current) {
                        el.srcObject = streamRef.current;
                        el.play().catch(e => console.error("Overlay video play failed:", e));
                      }
                    }}
                    style={{ transform: 'scaleX(-1)' }}
                  />
                  <div className="absolute inset-0 border-[16px] border-red-500/10 pointer-events-none animate-pulse" />
                  <div className="absolute bottom-4 left-0 right-0 flex justify-center">
                    <div className="bg-red-600 text-white px-4 py-2 rounded-full text-sm font-bold shadow-xl flex items-center gap-2">
                      <Camera className="w-4 h-4" />
                      Live Verifying...
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 flex flex-col items-center gap-4">
              <div className="flex items-center gap-4 w-full max-w-md">
                <div className="flex-1 space-y-2">
                  <div className="flex justify-between text-sm font-bold uppercase tracking-wider text-slate-500">
                    <span>Similarity Score</span>
                    <span className={lastMatchScore && lastMatchScore >= 0.75 ? 'text-green-600' : 'text-red-500'}>
                      {lastMatchScore ? Math.round(lastMatchScore * 100) : 0}%
                    </span>
                  </div>
                  <div className="h-4 bg-slate-200 rounded-full overflow-hidden border border-slate-300">
                    <div
                      className={`h-full transition-all duration-500 ${lastMatchScore && lastMatchScore >= 0.75 ? 'bg-green-500' : 'bg-red-500'}`}
                      style={{ width: `${(lastMatchScore || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              <p className="text-sm text-slate-500 text-center max-w-lg italic">
                The exam will automatically resume once the similarity score reaches 75% or higher.
                Ensure you are in a well-lit area and looking directly at the camera.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  // ─── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background">
      <FaceLockOverlay />
      {/* Header */}
      <header className="border-b bg-card shadow-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold">{exam.title}</h1>
              <p className="text-sm text-muted-foreground">
                {answeredCount}/{questions.length} questions answered
              </p>
            </div>
            <div className="flex items-center gap-4">
              {/* Violation badge */}
              {totalViolations > 0 && (
                <div className="flex items-center gap-1.5 px-3 py-1.5 bg-red-100 border border-red-300 rounded-md text-red-700 text-sm font-semibold">
                  <AlertTriangle className="w-4 h-4" />
                  {totalViolations} violation{totalViolations > 1 ? 's' : ''}
                </div>
              )}

              {/* Timer */}
              <div className={`flex items-center gap-2 px-4 py-2 rounded-md ${timeLeft < 300 ? 'bg-red-100 text-red-700' : 'bg-primary/10'}`}>
                <Clock className="w-5 h-5 text-primary" />
                <span className="font-mono font-semibold text-lg">{formatTime(timeLeft)}</span>
              </div>

              <Button onClick={handleSubmit} disabled={!allAnswered || submitting} size="lg">
                <Send className="w-4 h-4 mr-2" />
                {submitting ? 'Submitting...' : 'Submit Exam'}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Camera PiP — fixed bottom-right corner */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-2">
        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} className="hidden" />

        <div className="relative w-44 rounded-xl overflow-hidden border-2 border-primary/40 shadow-xl bg-black">
          <video
            ref={(el) => {
              // Maintain the videoRef for captureAndAnalyze
              (videoRef as any).current = el;
              if (el && streamRef.current && el.srcObject !== streamRef.current) {
                el.srcObject = streamRef.current;
                el.play().catch(e => console.error("PiP video play failed:", e));
              }
            }}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
            style={{ transform: 'scaleX(-1)' }} /* Mirror for natural feel */
          />

          {/* Status overlay */}
          <div className="absolute top-1.5 left-1.5 flex items-center gap-1 bg-black/60 text-white text-[10px] px-2 py-0.5 rounded-full">
            {cameraActive ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Monitored
              </>
            ) : (
              <>
                <CameraOff className="w-3 h-3 text-red-400" />
                {cameraError ? 'No Camera' : 'Starting…'}
              </>
            )}
          </div>

          {/* Python server badge */}
          {pythonServerUp === false && (
            <div className="absolute bottom-1 left-1 right-1 bg-yellow-500/90 text-[9px] text-white text-center px-1 py-0.5 rounded">
              AI offline — limited proctoring
            </div>
          )}
        </div>

        {cameraError && (
          <div className="max-w-[180px] text-[10px] text-red-600 text-center bg-red-50 border border-red-200 rounded-lg px-2 py-1">
            <Camera className="w-3 h-3 inline mr-1" />
            {cameraError}
          </div>
        )}
      </div>

      {/* Main content */}
      <main className="container mx-auto px-4 py-8">
        <div className={`mx-auto ${questions.some(q => q.questionType === 'coding') ? 'max-w-6xl' : 'max-w-3xl'} space-y-8`}>
          {questions.map((question, index) => (
            <Card key={question._id}>
              <CardHeader>
                <div className="flex items-center gap-2 mb-2">
                  <CardTitle>Question {index + 1}</CardTitle>
                  <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">
                    {question.questionType === 'mcq' ? 'MCQ' : 'Coding'}
                  </span>
                  <span className="text-xs bg-accent/10 text-accent px-2 py-1 rounded-full">
                    {question.points || 1} pts
                  </span>
                </div>
                <CardDescription className="text-base text-foreground">
                  {question.questionText}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {question.questionType === 'mcq' ? (
                  <div className="max-w-2xl mx-auto py-4">
                    <RadioGroup
                      value={answers[question._id] || ''}
                      onValueChange={(value) => setAnswers({ ...answers, [question._id]: value })}
                    >
                      <div className="space-y-3">
                        {(['A', 'B', 'C', 'D'] as const).map(opt => (
                          <div key={opt} className="flex items-center space-x-3 p-4 rounded-md border hover:bg-accent/10 cursor-pointer">
                            <RadioGroupItem value={opt} id={`${question._id}-${opt}`} />
                            <Label htmlFor={`${question._id}-${opt}`} className="flex-1 cursor-pointer">
                              <span className="font-semibold mr-2">{opt}.</span>
                              {question[`option${opt}` as keyof Question] as string}
                            </Label>
                          </div>
                        ))}
                      </div>
                    </RadioGroup>
                  </div>
                ) : (
                  <ResizablePanelGroup direction="horizontal" className="min-h-[700px] border rounded-xl overflow-hidden shadow-sm bg-card">
                    {/* Left Pane: Question Details */}
                    <ResizablePanel defaultSize={40} minSize={30}>
                      <div className="h-full overflow-y-auto p-6 space-y-6">
                        <div className="space-y-4">
                          <div className="flex items-center gap-2 text-indigo-600 font-semibold mb-2">
                            <BookOpen className="w-4 h-4" />
                            <span>Problem Description</span>
                          </div>
                          <div className="bg-muted/30 p-4 rounded-lg border border-indigo-50">
                            <p className="text-sm whitespace-pre-wrap leading-relaxed">{question.problemStatement}</p>
                          </div>
                        </div>

                        {(question.inputFormat || question.outputFormat) && (
                          <div className="space-y-3">
                            <h4 className="text-sm font-semibold text-gray-700">Format</h4>
                            <div className="grid grid-cols-1 gap-3">
                              {question.inputFormat && (
                                <div className="bg-indigo-50/50 p-3 rounded-lg border border-indigo-100/50">
                                  <p className="text-[10px] font-bold text-indigo-700 uppercase mb-1">Input Format</p>
                                  <p className="text-xs text-gray-600">{question.inputFormat}</p>
                                </div>
                              )}
                              {question.outputFormat && (
                                <div className="bg-purple-50/50 p-3 rounded-lg border border-purple-100/50">
                                  <p className="text-[10px] font-bold text-purple-700 uppercase mb-1">Output Format</p>
                                  <p className="text-xs text-gray-600">{question.outputFormat}</p>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {question.constraints && (
                          <div className="space-y-2">
                            <h4 className="text-sm font-semibold text-gray-700">Constraints</h4>
                            <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                              <p className="text-xs text-slate-600 font-mono">{question.constraints}</p>
                            </div>
                          </div>
                        )}

                        {/* Sample Test Cases */}
                        {question.testCases && question.testCases.length > 0 && (
                          <div className="space-y-3">
                            <h4 className="text-sm font-semibold text-gray-700">Sample Test Cases</h4>
                            <div className="space-y-3">
                              {question.testCases.filter(tc => !tc.isHidden).map((tc, idx) => (
                                <div key={idx} className="bg-white border rounded-lg overflow-hidden shadow-sm">
                                  <div className="bg-slate-50 px-3 py-1.5 border-b flex justify-between items-center text-[10px] font-bold text-gray-500 uppercase">
                                    <span>Example {idx + 1}</span>
                                  </div>
                                  <div className="p-3 space-y-2">
                                    <div className="grid grid-cols-1 gap-2">
                                      <div>
                                        <p className="text-[9px] font-bold text-slate-400 uppercase mb-1">Input</p>
                                        <pre className="font-mono bg-slate-50 p-2 rounded text-xs overflow-x-auto">{tc.input}</pre>
                                      </div>
                                      {tc.expectedOutput && (
                                        <div>
                                          <p className="text-[9px] font-bold text-slate-400 uppercase mb-1">Output</p>
                                          <pre className="font-mono bg-green-50/50 p-2 rounded text-xs overflow-x-auto text-green-700">{tc.expectedOutput}</pre>
                                        </div>
                                      )}
                                    </div>
                                    {tc.explanation && (
                                      <p className="text-[10px] text-muted-foreground mt-1 border-l-2 border-indigo-200 pl-2 italic">{tc.explanation}</p>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </ResizablePanel>

                    <ResizableHandle withHandle />

                    {/* Right Pane: Compiler & Output */}
                    <ResizablePanel defaultSize={60} minSize={40}>
                      <div className="h-full flex flex-col bg-slate-900 overflow-hidden">
                        {/* Control Bar */}
                        <div className="bg-slate-800 p-3 flex justify-between items-center border-b border-slate-700">
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2">
                              <Select
                                value={selectedLanguages[question._id] || 'javascript'}
                                onValueChange={(value) => handleLanguageChange(question._id, value)}
                              >
                                <SelectTrigger className="w-[130px] h-9 bg-slate-700 border-slate-600 text-slate-100 hover:bg-slate-600">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent className="bg-slate-800 border-slate-700 text-slate-100">
                                  <SelectItem value="javascript">JavaScript</SelectItem>
                                  <SelectItem value="python">Python</SelectItem>
                                  <SelectItem value="cpp">C++</SelectItem>
                                  <SelectItem value="java">Java</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 rounded-md border border-slate-600">
                              <Switch
                                id={`custom-input-${question._id}`}
                                checked={useCustomInput[question._id] || false}
                                onCheckedChange={(checked) => setUseCustomInput({ ...useCustomInput, [question._id]: checked })}
                                className="data-[state=checked]:bg-indigo-500 scale-75"
                              />
                              <Label htmlFor={`custom-input-${question._id}`} className="text-[11px] font-semibold text-slate-300 cursor-pointer">
                                Custom Input
                              </Label>
                            </div>
                          </div>

                          <Button
                            type="button"
                            size="sm"
                            className="bg-green-600 hover:bg-green-700 text-white shadow-lg shadow-green-900/20 font-bold"
                            onClick={() => handleRunCode(question._id)}
                            disabled={runningCode === question._id}
                          >
                            <Play className="w-4 h-4 mr-2" />
                            {runningCode === question._id ? 'Executing...' : 'Run Code'}
                          </Button>
                        </div>

                        {/* Editor Area */}
                        <div className="flex-1 relative border-b border-slate-700">
                          {useCustomInput[question._id] && (
                            <div className="absolute top-0 left-0 right-0 z-10 p-3 bg-slate-800/90 backdrop-blur-sm border-b border-slate-700">
                              <Textarea
                                placeholder="Enter manual test input here..."
                                className="font-mono text-xs min-h-[60px] max-h-[120px] bg-slate-900 border-slate-600 text-indigo-400 placeholder:text-slate-500"
                                value={customInputs[question._id] || ''}
                                onChange={(e) => setCustomInputs({ ...customInputs, [question._id]: e.target.value })}
                              />
                            </div>
                          )}

                          <Editor
                            height="100%"
                            language={selectedLanguages[question._id] || 'javascript'}
                            value={codeSubmissions[question._id] || ''}
                            onChange={(value) => setCodeSubmissions({ ...codeSubmissions, [question._id]: value || '' })}
                            theme="vs-dark"
                            options={{
                              minimap: { enabled: false },
                              fontSize: 14,
                              lineNumbers: 'on',
                              scrollBeyondLastLine: false,
                              automaticLayout: true,
                              padding: { top: useCustomInput[question._id] ? 100 : 20 },
                            }}
                          />
                        </div>

                        {/* Output Console area (Fixed bottom section) */}
                        <div className={`h-1/3 bg-slate-950 border-t border-slate-700 flex flex-col transition-all duration-300 ${codeOutput[question._id] ? 'opacity-100' : 'opacity-0 h-0 pointer-events-none'}`}>
                          <div className="bg-slate-900 px-4 py-2 border-b border-slate-800 flex justify-between items-center">
                            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Console Output</span>
                            {codeOutput[question._id]?.type === 'batch' && (
                              <span className="text-[9px] bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 px-2 py-0.5 rounded-full font-bold">
                                {codeOutput[question._id].results.filter((r: any) => r.passed).length}/{codeOutput[question._id].results.length} Passed
                              </span>
                            )}
                          </div>

                          <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                            {codeOutput[question._id]?.type === 'error' ? (
                              <div className="bg-red-950/30 border border-red-900/50 rounded p-3">
                                <pre className="text-xs font-mono text-red-400 whitespace-pre-wrap">
                                  {codeOutput[question._id].error}
                                </pre>
                              </div>
                            ) : codeOutput[question._id]?.type === 'single' ? (
                              <div className="space-y-3">
                                <div className={`rounded p-3 border ${codeOutput[question._id].success ? 'bg-indigo-950/30 border-indigo-900/50' : 'bg-red-950/30 border-red-900/50'}`}>
                                  <pre className={`text-xs font-mono whitespace-pre-wrap ${codeOutput[question._id].success ? 'text-indigo-400' : 'text-red-400'}`}>
                                    {codeOutput[question._id].success ? codeOutput[question._id].output : codeOutput[question._id].error}
                                  </pre>
                                </div>
                              </div>
                            ) : codeOutput[question._id]?.type === 'batch' && (
                              <div className="grid grid-cols-1 gap-3">
                                {codeOutput[question._id].results.map((res: any, idx: number) => (
                                  <div key={idx} className={`rounded-lg overflow-hidden border shadow-sm ${res.passed ? 'bg-green-950/20 border-green-900/30' : 'bg-red-950/20 border-red-900/30'}`}>
                                    <div className="px-3 py-1.5 border-b border-white/5 flex justify-between items-center">
                                      <div className="flex items-center gap-2">
                                        {res.passed ? <CheckCircle2 className="w-3.5 h-3.5 text-green-500" /> : <XCircle className="w-3.5 h-3.5 text-red-500" />}
                                        <span className="text-[11px] font-bold text-slate-300">Test Case {idx + 1}</span>
                                      </div>
                                      <span className="text-[9px] text-slate-500 font-mono">{res.executionTime}ms</span>
                                    </div>
                                    <div className="p-3 grid grid-cols-2 gap-3 text-[10px] font-mono">
                                      <div className="space-y-1">
                                        <p className="text-slate-500 uppercase tracking-tighter">Input</p>
                                        <div className="bg-black/40 p-1.5 rounded text-slate-400 truncate">{res.input}</div>
                                      </div>
                                      <div className="space-y-1">
                                        <p className="text-slate-500 uppercase tracking-tighter">Output</p>
                                        <div className={`p-1.5 rounded truncate bg-black/40 ${res.passed ? 'text-green-400' : 'text-red-400'}`}>
                                          {res.error || res.actualOutput}
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </ResizablePanel>
                  </ResizablePanelGroup>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
    </div>
  );
};

export default TakeExam;

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { authAPI, examAPI, resultAPI } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { Plus, LogOut, BookOpen, Calendar, Clock, Code, Users, BarChart3, Edit, GraduationCap, TrendingUp, Award, ShieldAlert, ShieldCheck } from "lucide-react";

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

type ExamResult = {
  _id: string;
  studentId: {
    _id: string;
    fullName: string;
    email: string;
  };
  examId: string;
  score: number;
  totalQuestions: number;
  totalPoints?: number;
  submittedAt: string;
  violations?: { type: string; count: number }[];
};

const TeacherDashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [user, setUser] = useState<any>(null);
  const [exams, setExams] = useState<Exam[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [resultsDialogOpen, setResultsDialogOpen] = useState(false);
  const [selectedExam, setSelectedExam] = useState<Exam | null>(null);
  const [examResults, setExamResults] = useState<ExamResult[]>([]);
  const [loadingResults, setLoadingResults] = useState(false);

  const [examData, setExamData] = useState({
    title: "",
    description: "",
    examDate: "",
    durationMinutes: 60,
  });

  useEffect(() => {
    checkUser();
    fetchExams();
  }, []);

  const checkUser = async () => {
    const userData = localStorage.getItem('user');
    if (!userData) {
      navigate("/auth");
      return;
    }

    const user = JSON.parse(userData);

    if (user.role !== "teacher") {
      navigate("/student/dashboard");
      return;
    }

    setUser(user);
    setLoading(false);
  };

  const fetchExams = async () => {
    try {
      const data = await examAPI.getAll();
      setExams(data);
    } catch (error) {
      console.error('Error fetching exams:', error);
    }
  };

  const fetchExamResults = async (examId: string) => {
    setLoadingResults(true);
    try {
      const data = await resultAPI.getExamResults(examId);
      setExamResults(data);
    } catch (error) {
      console.error('Error fetching results:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch exam results",
      });
    } finally {
      setLoadingResults(false);
    }
  };

  const handleViewResults = (exam: Exam, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedExam(exam);
    setResultsDialogOpen(true);
    fetchExamResults(exam._id);
  };

  const handleCreateExam = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const data = await examAPI.create(examData);

      toast({
        title: "Exam created!",
        description: `Exam code: ${data.examCode}`,
      });

      setDialogOpen(false);
      setExamData({
        title: "",
        description: "",
        examDate: "",
        durationMinutes: 60,
      });
      fetchExams();
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Error",
        description: error.response?.data?.message || error.message,
      });
    }
  };

  const handleLogout = async () => {
    authAPI.logout();
    navigate("/auth");
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 via-white to-purple-50">
        <div className="text-xl font-semibold text-indigo-600">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-indigo-100 sticky top-0 z-50 shadow-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-2 rounded-lg">
              <GraduationCap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Teacher Dashboard
              </h1>
              <p className="text-sm text-gray-600">Welcome back, {user?.fullName}!</p>
            </div>
          </div>
          <Button
            variant="outline"
            onClick={handleLogout}
            className="border-indigo-200 hover:bg-indigo-50"
          >
            <LogOut className="w-4 h-4 mr-2" />
            Logout
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="border-indigo-100 bg-gradient-to-br from-indigo-50 to-white hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Exams</p>
                  <p className="text-3xl font-bold text-indigo-600">{exams.length}</p>
                </div>
                <div className="bg-indigo-100 p-3 rounded-full">
                  <BookOpen className="w-6 h-6 text-indigo-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-purple-100 bg-gradient-to-br from-purple-50 to-white hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Active Exams</p>
                  <p className="text-3xl font-bold text-purple-600">
                    {exams.filter(e => new Date(e.examDate) >= new Date()).length}
                  </p>
                </div>
                <div className="bg-purple-100 p-3 rounded-full">
                  <TrendingUp className="w-6 h-6 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-green-100 bg-gradient-to-br from-green-50 to-white hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">This Month</p>
                  <p className="text-3xl font-bold text-green-600">
                    {exams.filter(e => {
                      const examDate = new Date(e.examDate);
                      const now = new Date();
                      return examDate.getMonth() === now.getMonth() && examDate.getFullYear() === now.getFullYear();
                    }).length}
                  </p>
                </div>
                <div className="bg-green-100 p-3 rounded-full">
                  <Award className="w-6 h-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Header Section */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2">
              My Exams
            </h2>
            <p className="text-gray-600">Create and manage your exams</p>
          </div>
          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button
                size="lg"
                className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-lg"
              >
                <Plus className="w-5 h-5 mr-2" />
                Create Exam
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  Create New Exam
                </DialogTitle>
                <DialogDescription>
                  Fill in the exam details. An exam code will be auto-generated.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleCreateExam} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="title">Exam Title</Label>
                  <Input
                    id="title"
                    value={examData.title}
                    onChange={(e) => setExamData({ ...examData, title: e.target.value })}
                    placeholder="e.g., Mathematics Final Exam"
                    className="border-indigo-200 focus:border-indigo-500"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={examData.description || ""}
                    onChange={(e) => setExamData({ ...examData, description: e.target.value })}
                    placeholder="Brief description of the exam..."
                    className="border-indigo-200 focus:border-indigo-500"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="date">Exam Date</Label>
                  <Input
                    id="date"
                    type="date"
                    value={examData.examDate}
                    onChange={(e) => setExamData({ ...examData, examDate: e.target.value })}
                    className="border-indigo-200 focus:border-indigo-500"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="duration">Duration (minutes)</Label>
                  <Input
                    id="duration"
                    type="number"
                    min="1"
                    value={examData.durationMinutes}
                    onChange={(e) => setExamData({ ...examData, durationMinutes: parseInt(e.target.value) })}
                    className="border-indigo-200 focus:border-indigo-500"
                    required
                  />
                </div>
                <Button
                  type="submit"
                  className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                >
                  Create Exam
                </Button>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        {/* Exams Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {exams.map((exam) => (
            <Card
              key={exam._id}
              className="group border-2 border-indigo-100 hover:border-indigo-300 hover:shadow-xl transition-all duration-300 cursor-pointer bg-white"
              onClick={() => navigate(`/teacher/exam/${exam._id}`)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-xl mb-2 group-hover:text-indigo-600 transition-colors">
                      {exam.title}
                    </CardTitle>
                    <CardDescription className="line-clamp-2">{exam.description}</CardDescription>
                  </div>
                  <div className="bg-gradient-to-r from-indigo-100 to-purple-100 p-2 rounded-lg">
                    <BookOpen className="w-5 h-5 text-indigo-600" />
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 p-2 bg-indigo-50 rounded-lg">
                    <Code className="w-4 h-4 text-indigo-600 flex-shrink-0" />
                    <span className="font-mono font-bold text-indigo-700">{exam.examCode}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Calendar className="w-4 h-4 text-purple-600" />
                    <span>{new Date(exam.examDate).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    })}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Clock className="w-4 h-4 text-purple-600" />
                    <span>{exam.durationMinutes} minutes</span>
                  </div>
                </div>

                <div className="flex gap-2 pt-3 border-t border-indigo-100">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(`/teacher/exam/${exam._id}`);
                    }}
                  >
                    <Edit className="w-4 h-4 mr-1" />
                    Manage
                  </Button>
                  <Button
                    size="sm"
                    className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                    onClick={(e) => handleViewResults(exam, e)}
                  >
                    <BarChart3 className="w-4 h-4 mr-1" />
                    Results
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Empty State */}
        {exams.length === 0 && (
          <div className="text-center py-16">
            <div className="bg-gradient-to-r from-indigo-100 to-purple-100 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6">
              <BookOpen className="w-12 h-12 text-indigo-600" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">No exams yet</h3>
            <p className="text-gray-600 mb-6">Create your first exam to get started</p>
            <Button
              onClick={() => setDialogOpen(true)}
              className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
            >
              <Plus className="w-4 h-4 mr-2" />
              Create Your First Exam
            </Button>
          </div>
        )}
      </main>

      {/* Results Dialog */}
      <Dialog open={resultsDialogOpen} onOpenChange={setResultsDialogOpen}>
        <DialogContent className="max-w-5xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              <BarChart3 className="w-6 h-6 text-indigo-600" />
              <span className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Exam Results: {selectedExam?.title}
              </span>
            </DialogTitle>
            <DialogDescription>
              View all student submissions and scores for this exam
            </DialogDescription>
          </DialogHeader>

          {loadingResults ? (
            <div className="text-center py-12">
              <div className="text-gray-600">Loading results...</div>
            </div>
          ) : examResults.length === 0 ? (
            <div className="text-center py-16">
              <div className="bg-gradient-to-r from-indigo-100 to-purple-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                <Users className="w-10 h-10 text-indigo-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">No submissions yet</h3>
              <p className="text-gray-600">Students haven't taken this exam yet</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Summary Stats */}
              <div className="grid grid-cols-3 gap-4">
                <Card className="border-indigo-100 bg-gradient-to-br from-indigo-50 to-white">
                  <CardContent className="pt-6 text-center">
                    <div className="text-3xl font-bold text-indigo-600 mb-1">{examResults.length}</div>
                    <div className="text-sm text-gray-600">Total Submissions</div>
                  </CardContent>
                </Card>
                <Card className="border-green-100 bg-gradient-to-br from-green-50 to-white">
                  <CardContent className="pt-6 text-center">
                    <div className="text-3xl font-bold text-green-600 mb-1">
                      {Math.round(
                        (examResults.reduce((sum, r) => sum + r.score, 0) /
                          examResults.reduce((sum, r) => sum + (r.totalPoints || r.totalQuestions), 0)) * 100
                      )}%
                    </div>
                    <div className="text-sm text-gray-600">Average Score</div>
                  </CardContent>
                </Card>
                <Card className="border-blue-100 bg-gradient-to-br from-blue-50 to-white">
                  <CardContent className="pt-6 text-center">
                    <div className="text-3xl font-bold text-blue-600 mb-1">
                      {examResults.filter(r => {
                        const total = r.totalPoints || r.totalQuestions;
                        return (r.score / total) >= 0.6;
                      }).length}
                    </div>
                    <div className="text-sm text-gray-600">Passed (≥60%)</div>
                  </CardContent>
                </Card>
              </div>

              {/* Results Table */}
              <div className="space-y-3">
                <h4 className="font-semibold text-lg">Student Results</h4>
                <div className="border-2 border-indigo-100 rounded-xl overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-gradient-to-r from-indigo-50 to-purple-50">
                      <tr>
                        <th className="text-left p-4 text-sm font-semibold text-gray-700">#</th>
                        <th className="text-left p-4 text-sm font-semibold text-gray-700">Student Name</th>
                        <th className="text-left p-4 text-sm font-semibold text-gray-700">Email</th>
                        <th className="text-center p-4 text-sm font-semibold text-gray-700">Score</th>
                        <th className="text-center p-4 text-sm font-semibold text-gray-700">Percentage</th>
                        <th className="text-center p-4 text-sm font-semibold text-gray-700">Violations</th>
                        <th className="text-left p-4 text-sm font-semibold text-gray-700">Submitted</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white">
                      {examResults
                        .sort((a, b) => {
                          const aTotal = a.totalPoints || a.totalQuestions;
                          const bTotal = b.totalPoints || b.totalQuestions;
                          return (b.score / bTotal) - (a.score / aTotal);
                        })
                        .map((result, index) => {
                          const totalPoints = result.totalPoints || result.totalQuestions;
                          const percentage = Math.round((result.score / totalPoints) * 100);
                          const passed = percentage >= 60;

                          return (
                            <tr key={result._id} className="border-t border-indigo-50 hover:bg-indigo-50/50 transition-colors">
                              <td className="p-4 text-sm font-medium text-gray-700">{index + 1}</td>
                              <td className="p-4 text-sm font-semibold text-gray-900">{result.studentId.fullName}</td>
                              <td className="p-4 text-sm text-gray-600">{result.studentId.email}</td>
                              <td className="p-4 text-center">
                                <span className="font-mono font-bold text-indigo-600">
                                  {result.score}/{totalPoints}
                                </span>
                              </td>
                              <td className="p-4 text-center">
                                <span
                                  className={`inline-flex items-center justify-center px-3 py-1 text-sm font-bold rounded-full ${passed
                                      ? 'bg-green-100 text-green-700'
                                      : 'bg-red-100 text-red-700'
                                    }`}
                                >
                                  {percentage}%
                                </span>
                              </td>
                              {/* Violations cell */}
                              <td className="p-4">
                                {(() => {
                                  const vList = result.violations || [];
                                  const totalV = vList.reduce((s, v) => s + v.count, 0);
                                  const labelMap: Record<string, string> = {
                                    tab_switch: 'Tab Switch',
                                    copy_paste: 'Copy/Paste',
                                    keyboard_shortcut: 'Shortcut',
                                    no_face: 'No Face',
                                    multiple_faces: 'Multi-Face',
                                  };
                                  if (totalV === 0) {
                                    return (
                                      <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">
                                        <ShieldCheck className="w-3 h-3" /> Clean
                                      </span>
                                    );
                                  }
                                  return (
                                    <div className="flex flex-col gap-1">
                                      <span className="inline-flex items-center gap-1 px-2 py-1 bg-red-100 text-red-700 text-xs font-bold rounded-full w-fit">
                                        <ShieldAlert className="w-3 h-3" /> {totalV} total
                                      </span>
                                      <div className="flex flex-wrap gap-1 mt-0.5">
                                        {vList.map(v => (
                                          <span key={v.type} className="px-1.5 py-0.5 bg-orange-100 text-orange-700 text-[10px] font-medium rounded">
                                            {labelMap[v.type] || v.type}: {v.count}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  );
                                })()}
                              </td>
                              <td className="p-4 text-sm text-gray-600">
                                {new Date(result.submittedAt).toLocaleString('en-US', {
                                  month: 'short',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Export Button */}
              <div className="flex justify-end pt-4 border-t border-indigo-100">
                <Button
                  variant="outline"
                  className="border-indigo-200 hover:bg-indigo-50"
                  onClick={() => {
                    toast({
                      title: "Export feature",
                      description: "CSV export coming soon!",
                    });
                  }}
                >
                  Export Results
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default TeacherDashboard;

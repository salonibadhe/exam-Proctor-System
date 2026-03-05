import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { authAPI, examAPI, resultAPI } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { LogOut, BookOpen, Search, Trophy, Calendar, TrendingUp, Award, Target, ArrowRight } from "lucide-react";

type ExamResult = {
  _id: string;
  studentId: string;
  examId: {
    _id: string;
    title: string;
    examCode: string;
  };
  score: number;
  totalQuestions: number;
  totalPoints?: number;
  submittedAt: string;
};

const StudentDashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [examCode, setExamCode] = useState("");
  const [results, setResults] = useState<ExamResult[]>([]);

  useEffect(() => {
    checkUser();
    fetchResults();
  }, []);

  const checkUser = async () => {
    const userData = localStorage.getItem('user');
    if (!userData) {
      navigate("/auth");
      return;
    }
    
    const user = JSON.parse(userData);
    
    if (user.role !== "student") {
      navigate("/teacher/dashboard");
      return;
    }
    
    setUser(user);
    setLoading(false);
  };

  const fetchResults = async () => {
    try {
      const data = await resultAPI.getStudentResults();
      setResults(data);
    } catch (error) {
      console.error('Error fetching results:', error);
    }
  };

  const handleStartExam = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const exam = await examAPI.getByCode(examCode);
      navigate(`/student/exam/${exam._id}`);
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Error",
        description: error.response?.data?.message || "Invalid exam code or you have already taken this exam.",
      });
    }
  };

  const handleLogout = async () => {
    authAPI.logout();
    navigate("/auth");
  };

  // Calculate stats
  const totalExams = results.length;
  const averageScore = totalExams > 0 
    ? Math.round(results.reduce((sum, r) => {
        const total = r.totalPoints || r.totalQuestions;
        return sum + (r.score / total) * 100;
      }, 0) / totalExams)
    : 0;
  const passedExams = results.filter(r => {
    const total = r.totalPoints || r.totalQuestions;
    return (r.score / total) >= 0.6;
  }).length;

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 via-white to-indigo-50">
        <div className="text-xl font-semibold text-purple-600">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-purple-100 sticky top-0 z-50 shadow-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-purple-600 to-indigo-600 p-2 rounded-lg">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
                Student Dashboard
              </h1>
              <p className="text-sm text-gray-600">Welcome back, {user?.fullName}!</p>
            </div>
          </div>
          <Button 
            variant="outline" 
            onClick={handleLogout}
            className="border-purple-200 hover:bg-purple-50"
          >
            <LogOut className="w-4 h-4 mr-2" />
            Logout
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Stats Cards */}
        {/* <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="border-purple-100 bg-gradient-to-br from-purple-50 to-white hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Exams Taken</p>
                  <p className="text-3xl font-bold text-purple-600">{totalExams}</p>
                </div>
                <div className="bg-purple-100 p-3 rounded-full">
                  <BookOpen className="w-6 h-6 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-indigo-100 bg-gradient-to-br from-indigo-50 to-white hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Average Score</p>
                  <p className="text-3xl font-bold text-indigo-600">{averageScore}%</p>
                </div>
                <div className="bg-indigo-100 p-3 rounded-full">
                  <TrendingUp className="w-6 h-6 text-indigo-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-green-100 bg-gradient-to-br from-green-50 to-white hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Passed</p>
                  <p className="text-3xl font-bold text-green-600">{passedExams}</p>
                </div>
                <div className="bg-green-100 p-3 rounded-full">
                  <Trophy className="w-6 h-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div> */}

        {/* Start Exam Section */}
        <Card className="mb-8 border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-white shadow-lg">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center gap-2">
              <Target className="w-6 h-6 text-purple-600" />
              <span className="bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
                Start New Exam
              </span>
            </CardTitle>
            <CardDescription>Enter the exam code provided by your teacher</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleStartExam} className="flex gap-3">
              <div className="flex-1">
                <Input
                  type="text"
                  placeholder="Enter exam code (e.g., ABC123)"
                  value={examCode}
                  onChange={(e) => setExamCode(e.target.value.toUpperCase())}
                  className="text-lg font-mono border-purple-200 focus:border-purple-500 h-12"
                  required
                />
              </div>
              <Button 
                type="submit" 
                size="lg"
                className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 shadow-lg px-8"
              >
                <Search className="w-5 h-5 mr-2" />
                Start Exam
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Results Section */}
        {/* <div className="space-y-4"> */}
          {/* <div className="flex items-center justify-between">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
              My Results
            </h2>
            {results.length > 0 && (
              <span className="text-sm text-gray-600">{results.length} exam{results.length !== 1 ? 's' : ''} completed</span>
            )}
          </div> */}

          {/* {results.length === 0 ? (
            <Card className="border-2 border-purple-100">
              <CardContent className="text-center py-16">
                <div className="bg-gradient-to-r from-purple-100 to-indigo-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Trophy className="w-10 h-10 text-purple-600" />
                </div>
                <h3 className="text-xl font-semibold mb-2">No exams taken yet</h3>
                <p className="text-gray-600 mb-6">Enter an exam code above to start your first exam</p>
              </CardContent>
            </Card>
          ) : ( */}
            {/* <div className="grid gap-4">
              {results
                .sort((a, b) => new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime())
                .map((result) => {
                  const totalPoints = result.totalPoints || result.totalQuestions;
                  const percentage = Math.round((result.score / totalPoints) * 100);
                  const passed = percentage >= 60;
                  
                  return (
                    <Card 
                      key={result._id} 
                      className="group border-2 border-purple-100 hover:border-purple-300 hover:shadow-lg transition-all duration-300"
                    >
                      <CardContent className="p-6">
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <h3 className="text-xl font-bold text-gray-900 group-hover:text-purple-600 transition-colors">
                                {result.examId.title}
                              </h3>
                              <span className="px-3 py-1 bg-purple-100 text-purple-700 text-xs font-mono font-semibold rounded-full">
                                {result.examId.examCode}
                              </span>
                            </div>
                            <div className="flex items-center gap-4 text-sm text-gray-600">
                              <div className="flex items-center gap-1">
                                <Calendar className="w-4 h-4" />
                                <span>{new Date(result.submittedAt).toLocaleDateString('en-US', {
                                  year: 'numeric',
                                  month: 'long',
                                  day: 'numeric'
                                })}</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <Award className="w-4 h-4" />
                                <span className="font-mono font-semibold">{result.score}/{totalPoints} points</span>
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-4">
                            <div className="text-right">
                              <div className={`text-4xl font-bold ${passed ? 'text-green-600' : 'text-red-600'}`}>
                                {percentage}%
                              </div>
                              <div className={`text-sm font-semibold ${passed ? 'text-green-600' : 'text-red-600'}`}>
                                {passed ? 'Passed' : 'Failed'}
                              </div>
                            </div>
                            <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                              passed ? 'bg-green-100' : 'bg-red-100'
                            }`}>
                              {passed ? (
                                <Trophy className={`w-8 h-8 text-green-600`} />
                              ) : (
                                <Target className={`w-8 h-8 text-red-600`} />
                              )}
                            </div>
                          </div>
                        </div>
                        
                        <div className="mt-4 pt-4 border-t border-purple-100">
                          <div className="flex items-center justify-between text-xs text-gray-600 mb-2">
                            <span>Score Progress</span>
                            <span>{result.score} / {totalPoints}</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                            <div 
                              className={`h-full rounded-full transition-all duration-500 ${
                                passed 
                                  ? 'bg-gradient-to-r from-green-500 to-green-600' 
                                  : 'bg-gradient-to-r from-red-500 to-red-600'
                              }`}
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
            </div> */}
          {/* )} */}
        {/* </div> */}
      </main>
    </div>
  );
};

export default StudentDashboard;

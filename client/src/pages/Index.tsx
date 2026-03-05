import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { GraduationCap, BookOpen, Users, CheckCircle, Sparkles, Award, Clock } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    checkUser();
  }, []);

  const checkUser = () => {
    const userData = localStorage.getItem('user');
    if (userData) {
      const user = JSON.parse(userData);
      navigate(user.role === "teacher" ? "/teacher/dashboard" : "/student/dashboard");
    } else {
      // If no user is logged in, redirect to auth page
      navigate("/auth");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <header className="container mx-auto px-4 py-6 backdrop-blur-sm bg-white/50 sticky top-0 z-50 shadow-sm">
        <nav className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-2 rounded-lg">
              <GraduationCap className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
              Exam Management
            </h1>
          </div>
          <Button 
            onClick={() => navigate("/auth")} 
            size="lg"
            className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-lg"
          >
            Get Started
          </Button>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-6xl mx-auto">
          {/* Main Hero */}
          <div className="text-center space-y-8 mb-20">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-100 text-indigo-700 text-sm font-medium mb-4">
              <Sparkles className="w-4 h-4" />
              Modern Exam Platform
            </div>
            
            <h2 className="text-6xl font-bold tracking-tight bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Transform Your
              <br />
              Exam Experience
            </h2>
            
            <p className="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
              A complete solution for teachers to create exams and students to take them with real-time scoring and instant results.
            </p>

            <div className="flex justify-center gap-4 pt-4">
              <Button 
                onClick={() => navigate("/auth")} 
                size="lg" 
                className="text-lg px-10 py-7 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-xl hover:shadow-2xl transition-all"
              >
                <BookOpen className="w-5 h-5 mr-2" />
                Start Now
              </Button>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-2 gap-8 mb-20">
            {/* Teacher Card */}
            <div className="group p-8 rounded-2xl border-2 border-indigo-100 bg-gradient-to-br from-indigo-50 to-white hover:shadow-2xl hover:border-indigo-300 transition-all duration-300">
              <div className="w-16 h-16 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <GraduationCap className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold mb-4 text-indigo-900">For Teachers</h3>
              <ul className="space-y-3">
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-0.5 flex-shrink-0" />
                  <span>Create and manage unlimited exams</span>
                </li>
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-0.5 flex-shrink-0" />
                  <span>Add multiple-choice questions easily</span>
                </li>
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-0.5 flex-shrink-0" />
                  <span>Auto-generate unique exam codes</span>
                </li>
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-0.5 flex-shrink-0" />
                  <span>View detailed student results</span>
                </li>
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-indigo-600 mt-0.5 flex-shrink-0" />
                  <span>OTP-based secure login</span>
                </li>
              </ul>
            </div>

            {/* Student Card */}
            <div className="group p-8 rounded-2xl border-2 border-purple-100 bg-gradient-to-br from-purple-50 to-white hover:shadow-2xl hover:border-purple-300 transition-all duration-300">
              <div className="w-16 h-16 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <BookOpen className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold mb-4 text-purple-900">For Students</h3>
              <ul className="space-y-3">
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
                  <span>Join exams with a simple code</span>
                </li>
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
                  <span>Take timed exams seamlessly</span>
                </li>
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
                  <span>Get instant results and feedback</span>
                </li>
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
                  <span>Track your performance over time</span>
                </li>
                <li className="flex items-start gap-3 text-gray-700">
                  <CheckCircle className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
                  <span>User-friendly interface</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Stats Section */}
          <div className="grid grid-cols-3 gap-8 p-8 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
            <div className="text-center">
              <Award className="w-12 h-12 mx-auto mb-3 opacity-90" />
              <div className="text-4xl font-bold mb-2">100%</div>
              <div className="text-indigo-100">Automated</div>
            </div>
            <div className="text-center">
              <Clock className="w-12 h-12 mx-auto mb-3 opacity-90" />
              <div className="text-4xl font-bold mb-2">Instant</div>
              <div className="text-indigo-100">Results</div>
            </div>
            <div className="text-center">
              <Users className="w-12 h-12 mx-auto mb-3 opacity-90" />
              <div className="text-4xl font-bold mb-2">Secure</div>
              <div className="text-indigo-100">Platform</div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 text-center text-gray-600 border-t">
        <p>© 2024 Exam Management System. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default Index;

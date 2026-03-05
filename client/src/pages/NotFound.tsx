import { useLocation, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Home, Search, ArrowLeft, BookOpen } from "lucide-react";

const NotFound = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-4">
      <div className="text-center max-w-2xl">
        {/* 404 Illustration */}
        <div className="mb-8 relative">
          <div className="text-[200px] font-black bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent leading-none">
            404
          </div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="bg-white/80 backdrop-blur-sm p-6 rounded-full shadow-xl">
              <Search className="w-16 h-16 text-purple-600" />
            </div>
          </div>
        </div>

        {/* Message */}
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Oops! Page Not Found
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            onClick={() => navigate(-1)}
            variant="outline"
            size="lg"
            className="border-indigo-200 hover:bg-indigo-50"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Go Back
          </Button>
          <Button
            onClick={() => navigate("/")}
            size="lg"
            className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 shadow-lg"
          >
            <Home className="w-5 h-5 mr-2" />
            Back to Home
          </Button>
        </div>

        {/* Additional Info */}
        <div className="mt-12 p-6 bg-white/50 backdrop-blur-sm rounded-2xl border-2 border-indigo-100">
          <div className="flex items-center justify-center gap-2 text-gray-600">
            <BookOpen className="w-5 h-5 text-indigo-600" />
            <p className="text-sm">
              Looking for exams? Head to the{" "}
              <button
                onClick={() => navigate("/auth")}
                className="text-indigo-600 font-semibold hover:underline"
              >
                login page
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotFound;

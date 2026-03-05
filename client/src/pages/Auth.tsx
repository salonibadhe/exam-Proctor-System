import { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { authAPI } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useToast } from "@/hooks/use-toast";
import { GraduationCap, BookOpen, AlertCircle, Smartphone, Mail, Lock, User, Phone, Camera, RotateCcw } from "lucide-react";
import Webcam from "react-webcam";

type UserRole = "teacher" | "student";
type LoginMode = "student" | "teacher";

const Auth = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [role, setRole] = useState<UserRole>("student");
  const [loginMode, setLoginMode] = useState<LoginMode>("student");
  const [errors, setErrors] = useState<Record<string, string>>({});

  // OTP states
  const [otpSent, setOtpSent] = useState(false);
  const [otpLoading, setOtpLoading] = useState(false);

  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    password: "",
    confirmPassword: "",
    mobile: "",
    otp: "",
  });

  // Camera states
  const [profilePhoto, setProfilePhoto] = useState<string | null>(null);
  const [showCamera, setShowCamera] = useState(false);
  const webcamRef = useRef<Webcam>(null);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setProfilePhoto(imageSrc);
      setShowCamera(false);
    }
  }, [webcamRef]);

  const retake = () => {
    setProfilePhoto(null);
    setShowCamera(true);
  };

  // Validation functions
  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validateMobile = (mobile: string): boolean => {
    return /^[0-9]{10}$/.test(mobile);
  };

  const validatePassword = (password: string): boolean => {
    return password.length >= 6;
  };

  const validateFullName = (name: string): boolean => {
    return name.trim().length >= 3;
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (isLogin) {
      if (loginMode === "student") {
        // Student login validation
        if (!formData.email) {
          newErrors.email = "Email is required";
        } else if (!validateEmail(formData.email)) {
          newErrors.email = "Please enter a valid email address";
        }

        if (!formData.password) {
          newErrors.password = "Password is required";
        }
      } else {
        // Teacher OTP login validation
        if (!formData.mobile) {
          newErrors.mobile = "Mobile number is required";
        } else if (!validateMobile(formData.mobile)) {
          newErrors.mobile = "Please enter a valid 10-digit mobile number";
        }

        if (otpSent && !formData.otp) {
          newErrors.otp = "OTP is required";
        }
      }
    } else {
      // Signup validation
      if (!formData.fullName) {
        newErrors.fullName = "Full name is required";
      } else if (!validateFullName(formData.fullName)) {
        newErrors.fullName = "Name must be at least 3 characters long";
      }

      if (role === "student") {
        if (!formData.email) {
          newErrors.email = "Email is required";
        } else if (!validateEmail(formData.email)) {
          newErrors.email = "Please enter a valid email address";
        }

        if (!formData.password) {
          newErrors.password = "Password is required";
        } else if (!validatePassword(formData.password)) {
          newErrors.password = "Password must be at least 6 characters long";
        }

        if (!formData.confirmPassword) {
          newErrors.confirmPassword = "Please confirm your password";
        } else if (formData.password !== formData.confirmPassword) {
          newErrors.confirmPassword = "Passwords do not match";
        }
      } else {
        // Teacher signup - now requires email, password, and mobile
        if (!formData.email) {
          newErrors.email = "Email is required";
        } else if (!validateEmail(formData.email)) {
          newErrors.email = "Please enter a valid email address";
        }

        if (!formData.password) {
          newErrors.password = "Password is required";
        } else if (!validatePassword(formData.password)) {
          newErrors.password = "Password must be at least 6 characters long";
        }

        if (!formData.confirmPassword) {
          newErrors.confirmPassword = "Please confirm your password";
        } else if (formData.password !== formData.confirmPassword) {
          newErrors.confirmPassword = "Passwords do not match";
        }

        if (!formData.mobile) {
          newErrors.mobile = "Mobile number is required";
        } else if (!validateMobile(formData.mobile)) {
          newErrors.mobile = "Please enter a valid 10-digit mobile number";
        }
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSendOTP = async () => {
    if (!formData.mobile) {
      setErrors({ mobile: "Mobile number is required" });
      return;
    }

    if (!validateMobile(formData.mobile)) {
      setErrors({ mobile: "Please enter a valid 10-digit mobile number" });
      return;
    }

    setOtpLoading(true);
    try {
      const response = await authAPI.sendOTP({ mobile: formData.mobile });

      toast({
        title: "OTP Sent!",
        description: response.message || "OTP has been sent to your mobile number",
      });

      // Show OTP in dev mode
      if (response.otp) {
        toast({
          title: "Development Mode",
          description: `Your OTP is: ${response.otp}`,
          duration: 10000,
        });
      }

      setOtpSent(true);
      setErrors({});
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Error",
        description: error.response?.data?.message || "Failed to send OTP",
      });
    } finally {
      setOtpLoading(false);
    }
  };

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate form
    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      if (isLogin) {
        if (loginMode === "student") {
          // Student login
          const data = await authAPI.login({
            email: formData.email,
            password: formData.password,
          });

          localStorage.setItem('token', data.token);
          localStorage.setItem('user', JSON.stringify(data));

          navigate("/student/dashboard");
        } else {
          // Teacher OTP login
          const data = await authAPI.verifyOTP({
            mobile: formData.mobile,
            otp: formData.otp,
          });

          localStorage.setItem('token', data.token);
          localStorage.setItem('user', JSON.stringify(data));

          navigate("/teacher/dashboard");
        }
      } else {
        // Signup
        const signupData: any = {
          fullName: formData.fullName,
          email: formData.email,
          password: formData.password,
          role: role,
          profilePhoto: profilePhoto,
        };

        // Add mobile for teachers
        if (role === "teacher") {
          signupData.mobile = formData.mobile;
        }

        await authAPI.signup(signupData);

        toast({
          title: "Account created!",
          description: role === "student"
            ? "You can now log in with your credentials."
            : "You can now log in with OTP.",
        });

        // Reset form and switch to login
        setFormData({
          fullName: "",
          email: role === "student" ? formData.email : "",
          password: "",
          confirmPassword: "",
          mobile: role === "teacher" ? formData.mobile : "",
          otp: "",
        });
        setErrors({});
        setIsLogin(true);
        setLoginMode(role);
        setProfilePhoto(null);
        setShowCamera(false);
      }
    } catch (error: any) {
      console.error("Auth error:", error);

      // Handle validation errors
      if (error.response?.data?.errors) {
        const validationErrors = error.response.data.errors;
        const errorMessages = validationErrors.map((err: any) => err.msg).join(", ");
        toast({
          variant: "destructive",
          title: "Validation Error",
          description: errorMessages,
        });
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: error.response?.data?.message || error.message || "An error occurred",
        });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-4">
      <Card className="w-full max-w-md shadow-2xl border-0">
        <CardHeader className="text-center space-y-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-t-lg pb-8">
          <div className="flex justify-center mb-2">
            <div className="bg-white/20 p-3 rounded-full backdrop-blur-sm">
              <GraduationCap className="w-10 h-10" />
            </div>
          </div>
          <CardTitle className="text-3xl font-bold">Exam Management</CardTitle>
          <CardDescription className="text-indigo-100">
            {isLogin ? "Sign in to your account" : "Create a new account"}
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <Tabs value={isLogin ? "login" : "signup"} onValueChange={(v) => {
            setIsLogin(v === "login");
            setOtpSent(false);
            setErrors({});
          }}>
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="login" className="data-[state=active]:bg-indigo-600 data-[state=active]:text-white">
                Login
              </TabsTrigger>
              <TabsTrigger value="signup" className="data-[state=active]:bg-indigo-600 data-[state=active]:text-white">
                Sign Up
              </TabsTrigger>
            </TabsList>

            <TabsContent value="login">
              {/* Login Mode Selector */}
              <div className="mb-6">
                <Label className="text-sm font-semibold mb-3 block">Login as:</Label>
                <RadioGroup value={loginMode} onValueChange={(v) => {
                  setLoginMode(v as LoginMode);
                  setOtpSent(false);
                  setFormData({ ...formData, otp: "", email: "", password: "", mobile: "" });
                  setErrors({});
                }}>
                  <div className="grid grid-cols-2 gap-3">
                    <div className={`relative flex items-center space-x-2 p-4 rounded-lg border-2 cursor-pointer transition-all ${loginMode === "student" ? "border-indigo-600 bg-indigo-50" : "border-gray-200 hover:border-indigo-300"
                      }`}>
                      <RadioGroupItem value="student" id="login-student" />
                      <Label htmlFor="login-student" className="flex items-center gap-2 cursor-pointer flex-1">
                        <BookOpen className="w-5 h-5 text-indigo-600" />
                        <span className="font-medium">Student</span>
                      </Label>
                    </div>
                    <div className={`relative flex items-center space-x-2 p-4 rounded-lg border-2 cursor-pointer transition-all ${loginMode === "teacher" ? "border-purple-600 bg-purple-50" : "border-gray-200 hover:border-purple-300"
                      }`}>
                      <RadioGroupItem value="teacher" id="login-teacher" />
                      <Label htmlFor="login-teacher" className="flex items-center gap-2 cursor-pointer flex-1">
                        <GraduationCap className="w-5 h-5 text-purple-600" />
                        <span className="font-medium">Teacher</span>
                      </Label>
                    </div>
                  </div>
                </RadioGroup>
              </div>

              <form onSubmit={handleAuth} className="space-y-4">
                {loginMode === "student" ? (
                  <>
                    {/* Student Login */}
                    <div className="space-y-2">
                      <Label htmlFor="email" className="flex items-center gap-2">
                        <Mail className="w-4 h-4 text-indigo-600" />
                        Email
                      </Label>
                      <Input
                        id="email"
                        type="email"
                        placeholder="student@example.com"
                        value={formData.email}
                        onChange={(e) => {
                          setFormData({ ...formData, email: e.target.value });
                          if (errors.email) setErrors({ ...errors, email: "" });
                        }}
                        className={`${errors.email ? "border-red-500" : "border-gray-300"} focus:border-indigo-600`}
                      />
                      {errors.email && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.email}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="password" className="flex items-center gap-2">
                        <Lock className="w-4 h-4 text-indigo-600" />
                        Password
                      </Label>
                      <Input
                        id="password"
                        type="password"
                        placeholder="Enter your password"
                        value={formData.password}
                        onChange={(e) => {
                          setFormData({ ...formData, password: e.target.value });
                          if (errors.password) setErrors({ ...errors, password: "" });
                        }}
                        className={`${errors.password ? "border-red-500" : "border-gray-300"} focus:border-indigo-600`}
                      />
                      {errors.password && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.password}
                        </p>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    {/* Teacher OTP Login */}
                    <div className="space-y-2">
                      <Label htmlFor="mobile" className="flex items-center gap-2">
                        <Phone className="w-4 h-4 text-purple-600" />
                        Mobile Number
                      </Label>
                      <div className="flex gap-2">
                        <Input
                          id="mobile"
                          type="tel"
                          placeholder="10-digit mobile number"
                          value={formData.mobile}
                          onChange={(e) => {
                            setFormData({ ...formData, mobile: e.target.value });
                            if (errors.mobile) setErrors({ ...errors, mobile: "" });
                          }}
                          disabled={otpSent}
                          className={`${errors.mobile ? "border-red-500" : "border-gray-300"} focus:border-purple-600`}
                          maxLength={10}
                        />
                        {!otpSent && (
                          <Button
                            type="button"
                            onClick={handleSendOTP}
                            disabled={otpLoading}
                            className="bg-purple-600 hover:bg-purple-700 whitespace-nowrap"
                          >
                            {otpLoading ? "Sending..." : "Send OTP"}
                          </Button>
                        )}
                      </div>
                      {errors.mobile && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.mobile}
                        </p>
                      )}
                    </div>

                    {otpSent && (
                      <div className="space-y-2">
                        <Label htmlFor="otp" className="flex items-center gap-2">
                          <Smartphone className="w-4 h-4 text-purple-600" />
                          Enter OTP
                        </Label>
                        <Input
                          id="otp"
                          type="text"
                          placeholder="6-digit OTP"
                          value={formData.otp}
                          onChange={(e) => {
                            setFormData({ ...formData, otp: e.target.value });
                            if (errors.otp) setErrors({ ...errors, otp: "" });
                          }}
                          className={`${errors.otp ? "border-red-500" : "border-gray-300"} focus:border-purple-600`}
                          maxLength={6}
                        />
                        {errors.otp && (
                          <p className="text-sm text-red-500 flex items-center gap-1">
                            <AlertCircle className="w-3 h-3" />
                            {errors.otp}
                          </p>
                        )}
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setOtpSent(false);
                            setFormData({ ...formData, otp: "" });
                          }}
                          className="text-purple-600 hover:text-purple-700"
                        >
                          Change Mobile Number
                        </Button>
                      </div>
                    )}
                  </>
                )}

                <Button
                  type="submit"
                  className={`w-full ${loginMode === "student" ? "bg-indigo-600 hover:bg-indigo-700" : "bg-purple-600 hover:bg-purple-700"}`}
                  disabled={loading || (loginMode === "teacher" && !otpSent)}
                >
                  {loading ? "Signing in..." : "Sign In"}
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="signup">
              <form onSubmit={handleAuth} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="fullName" className="flex items-center gap-2">
                    <User className="w-4 h-4 text-indigo-600" />
                    Full Name
                  </Label>
                  <Input
                    id="fullName"
                    type="text"
                    placeholder="John Doe"
                    value={formData.fullName}
                    onChange={(e) => {
                      setFormData({ ...formData, fullName: e.target.value });
                      if (errors.fullName) setErrors({ ...errors, fullName: "" });
                    }}
                    className={`${errors.fullName ? "border-red-500" : "border-gray-300"} focus:border-indigo-600`}
                  />
                  {errors.fullName && (
                    <p className="text-sm text-red-500 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" />
                      {errors.fullName}
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label>I am a:</Label>
                  <RadioGroup value={role} onValueChange={(v) => {
                    setRole(v as UserRole);
                    setFormData({ ...formData, email: "", password: "", confirmPassword: "", mobile: "" });
                    setErrors({});
                  }}>
                    <div className="grid grid-cols-2 gap-3">
                      <div className={`flex items-center space-x-2 p-4 rounded-lg border-2 cursor-pointer transition-all ${role === "student" ? "border-indigo-600 bg-indigo-50" : "border-gray-200 hover:border-indigo-300"
                        }`}>
                        <RadioGroupItem value="student" id="student" />
                        <Label htmlFor="student" className="flex items-center gap-2 cursor-pointer flex-1">
                          <BookOpen className="w-5 h-5 text-indigo-600" />
                          <span className="font-medium">Student</span>
                        </Label>
                      </div>
                      <div className={`flex items-center space-x-2 p-4 rounded-lg border-2 cursor-pointer transition-all ${role === "teacher" ? "border-purple-600 bg-purple-50" : "border-gray-200 hover:border-purple-300"
                        }`}>
                        <RadioGroupItem value="teacher" id="teacher" />
                        <Label htmlFor="teacher" className="flex items-center gap-2 cursor-pointer flex-1">
                          <GraduationCap className="w-5 h-5 text-purple-600" />
                          <span className="font-medium">Teacher</span>
                        </Label>
                      </div>
                    </div>
                  </RadioGroup>
                </div>

                {role === "student" ? (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="signup-email" className="flex items-center gap-2">
                        <Mail className="w-4 h-4 text-indigo-600" />
                        Email
                      </Label>
                      <Input
                        id="signup-email"
                        type="email"
                        placeholder="student@example.com"
                        value={formData.email}
                        onChange={(e) => {
                          setFormData({ ...formData, email: e.target.value });
                          if (errors.email) setErrors({ ...errors, email: "" });
                        }}
                        className={`${errors.email ? "border-red-500" : "border-gray-300"} focus:border-indigo-600`}
                      />
                      {errors.email && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.email}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="signup-password" className="flex items-center gap-2">
                        <Lock className="w-4 h-4 text-indigo-600" />
                        Password
                      </Label>
                      <Input
                        id="signup-password"
                        type="password"
                        placeholder="At least 6 characters"
                        value={formData.password}
                        onChange={(e) => {
                          setFormData({ ...formData, password: e.target.value });
                          if (errors.password) setErrors({ ...errors, password: "" });
                        }}
                        className={`${errors.password ? "border-red-500" : "border-gray-300"} focus:border-indigo-600`}
                      />
                      {errors.password && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.password}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="confirm-password" className="flex items-center gap-2">
                        <Lock className="w-4 h-4 text-indigo-600" />
                        Confirm Password
                      </Label>
                      <Input
                        id="confirm-password"
                        type="password"
                        placeholder="Re-enter your password"
                        value={formData.confirmPassword}
                        onChange={(e) => {
                          setFormData({ ...formData, confirmPassword: e.target.value });
                          if (errors.confirmPassword) setErrors({ ...errors, confirmPassword: "" });
                        }}
                        className={`${errors.confirmPassword ? "border-red-500" : "border-gray-300"} focus:border-indigo-600`}
                      />
                      {errors.confirmPassword && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.confirmPassword}
                        </p>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="teacher-email" className="flex items-center gap-2">
                        <Mail className="w-4 h-4 text-purple-600" />
                        Email
                      </Label>
                      <Input
                        id="teacher-email"
                        type="email"
                        placeholder="teacher@example.com"
                        value={formData.email}
                        onChange={(e) => {
                          setFormData({ ...formData, email: e.target.value });
                          if (errors.email) setErrors({ ...errors, email: "" });
                        }}
                        className={`${errors.email ? "border-red-500" : "border-gray-300"} focus:border-purple-600`}
                      />
                      {errors.email && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.email}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="teacher-password" className="flex items-center gap-2">
                        <Lock className="w-4 h-4 text-purple-600" />
                        Password
                      </Label>
                      <Input
                        id="teacher-password"
                        type="password"
                        placeholder="At least 6 characters"
                        value={formData.password}
                        onChange={(e) => {
                          setFormData({ ...formData, password: e.target.value });
                          if (errors.password) setErrors({ ...errors, password: "" });
                        }}
                        className={`${errors.password ? "border-red-500" : "border-gray-300"} focus:border-purple-600`}
                      />
                      {errors.password && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.password}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="teacher-confirm-password" className="flex items-center gap-2">
                        <Lock className="w-4 h-4 text-purple-600" />
                        Confirm Password
                      </Label>
                      <Input
                        id="teacher-confirm-password"
                        type="password"
                        placeholder="Re-enter your password"
                        value={formData.confirmPassword}
                        onChange={(e) => {
                          setFormData({ ...formData, confirmPassword: e.target.value });
                          if (errors.confirmPassword) setErrors({ ...errors, confirmPassword: "" });
                        }}
                        className={`${errors.confirmPassword ? "border-red-500" : "border-gray-300"} focus:border-purple-600`}
                      />
                      {errors.confirmPassword && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.confirmPassword}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="signup-mobile" className="flex items-center gap-2">
                        <Phone className="w-4 h-4 text-purple-600" />
                        Mobile Number
                      </Label>
                      <Input
                        id="signup-mobile"
                        type="tel"
                        placeholder="10-digit mobile number"
                        value={formData.mobile}
                        onChange={(e) => {
                          setFormData({ ...formData, mobile: e.target.value });
                          if (errors.mobile) setErrors({ ...errors, mobile: "" });
                        }}
                        className={`${errors.mobile ? "border-red-500" : "border-gray-300"} focus:border-purple-600`}
                        maxLength={10}
                      />
                      {errors.mobile && (
                        <p className="text-sm text-red-500 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.mobile}
                        </p>
                      )}
                      <p className="text-xs text-gray-500">
                        You'll use OTP to login as a teacher
                      </p>
                    </div>
                  </>
                )}

                {/* Photo Capture Section */}
                <div className="space-y-4">
                  <Label className="flex items-center gap-2">
                    <Camera className="w-4 h-4 text-indigo-600" />
                    Profile Photo
                  </Label>

                  <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-4 bg-gray-50">
                    {profilePhoto ? (
                      <div className="relative">
                        <img src={profilePhoto} alt="Profile" className="w-48 h-48 object-cover rounded-full border-4 border-white shadow-lg" />
                        <Button
                          type="button"
                          variant="secondary"
                          size="sm"
                          onClick={retake}
                          className="absolute bottom-0 right-0 rounded-full p-2"
                        >
                          <RotateCcw className="w-4 h-4" />
                        </Button>
                      </div>
                    ) : showCamera ? (
                      <div className="flex flex-col items-center gap-4 w-full">
                        <Webcam
                          audio={false}
                          ref={webcamRef}
                          screenshotFormat="image/jpeg"
                          className="w-full rounded-lg"
                        />
                        <Button type="button" onClick={capture} variant="default" size="sm">
                          Capture Photo
                        </Button>
                      </div>
                    ) : (
                      <div className="text-center">
                        <div className="bg-white p-4 rounded-full mb-2 inline-block shadow-sm">
                          <User className="w-8 h-8 text-gray-400" />
                        </div>
                        <p className="text-sm text-gray-500 mb-4">Capture a photo for your profile</p>
                        <Button type="button" onClick={() => setShowCamera(true)} variant="outline" size="sm">
                          Open Camera
                        </Button>
                      </div>
                    )}
                  </div>
                </div>

                <Button
                  type="submit"
                  className={`w-full ${role === "student" ? "bg-indigo-600 hover:bg-indigo-700" : "bg-purple-600 hover:bg-purple-700"}`}
                  disabled={loading}
                >
                  {loading ? "Creating account..." : "Create Account"}
                </Button>
              </form>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default Auth;

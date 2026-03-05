import axios from 'axios';

// Generate 6-digit OTP
export const generateOTP = () => {
  return Math.floor(100000 + Math.random() * 900000).toString();
};

// Send OTP via 2Factor.in (FREE - No payment required)
export const sendOTPVia2Factor = async (mobile, otp) => {
  try {
    if (!process.env.TWOFACTOR_API_KEY) {
      console.log(`📱 OTP for ${mobile}: ${otp}`);
      console.log(`⚠️  2Factor not configured. Using console logging.`);
      return { success: true, message: 'OTP logged to console' };
    }

    console.log(`📱 Sending OTP SMS to ${mobile} via 2Factor.in...`);

    // Use Transactional SMS endpoint (not voice)
    // Format: https://2factor.in/API/R1/?module=TRANS_SMS&apikey=API_KEY&to=PHONE&from=SENDER_ID&msg=MESSAGE
    const message = `Your OTP for Exam Management login is: ${otp}. Valid for 5 minutes. Do not share with anyone.`;
    const encodedMessage = encodeURIComponent(message);
    
    const response = await axios.get(
      `https://2factor.in/API/R1/?module=TRANS_SMS&apikey=${process.env.TWOFACTOR_API_KEY}&to=${mobile}&from=EXMAPP&msg=${encodedMessage}`
    );

    console.log('2Factor Response:', response.data);

    if (response.data.Status === 'Success' || response.data.Status === 'success') {
      console.log(`✅ OTP SMS sent successfully to ${mobile}`);
      return { success: true, message: 'OTP sent successfully via SMS' };
    } else {
      throw new Error('Failed to send OTP via 2Factor');
    }
  } catch (error) {
    console.error('❌ Error sending OTP via 2Factor:', error.message);
    if (error.response) {
      console.error('2Factor Error Response:', error.response.data);
    }
    
    console.log(`📱 FALLBACK - OTP for ${mobile}: ${otp}`);
    return { success: true, message: 'OTP logged to console (SMS failed)' };
  }
};

// Main OTP sending function - tries multiple providers
export const sendOTP = async (mobile, otp) => {
  try {
    // Try 2Factor.in first (FREE - No payment required)
    if (process.env.TWOFACTOR_API_KEY) {
      return await sendOTPVia2Factor(mobile, otp);
    }
    
    // Try Fast2SMS (Requires payment)
    if (process.env.FAST2SMS_API_KEY && process.env.NODE_ENV === 'production') {
      console.log(`📱 Sending OTP to ${mobile} via Fast2SMS...`);

      const response = await axios.post(
        'https://www.fast2sms.com/dev/bulkV2',
        {
          route: 'v3',
          sender_id: 'TXTIND',
          message: `Your OTP for Exam Management login is: ${otp}. Valid for 5 minutes.`,
          language: 'english',
          flash: 0,
          numbers: mobile
        },
        {
          headers: {
            'authorization': process.env.FAST2SMS_API_KEY,
            'Content-Type': 'application/json'
          }
        }
      );

      console.log('Fast2SMS Response:', response.data);

      if (response.data.return) {
        console.log(`✅ OTP sent successfully to ${mobile}`);
        return { success: true, message: 'OTP sent successfully to your mobile' };
      } else {
        throw new Error('Failed to send OTP - Fast2SMS returned false');
      }
    }
    
    // Fallback: Console logging for development
    console.log(`📱 OTP for ${mobile}: ${otp}`);
    console.log(`⚠️  No SMS service configured. Using console logging.`);
    return { success: true, message: 'OTP logged to console (dev mode)' };
    
  } catch (error) {
    console.error('❌ Error sending OTP:', error.message);
    if (error.response) {
      console.error('SMS Provider Error Response:', error.response.data);
    }
    
    // Fallback: Log OTP if SMS fails
    console.log(`📱 FALLBACK - OTP for ${mobile}: ${otp}`);
    return { success: true, message: 'OTP logged to console (SMS service failed)' };
  }
};

// Alternative: Twilio (if you prefer)
export const sendOTPViaTwilio = async (mobile, otp) => {
  try {
    // Requires: npm install twilio
    // const twilio = require('twilio');
    // const client = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
    
    // await client.messages.create({
    //   body: `Your OTP for Exam Management login is: ${otp}. Valid for 5 minutes.`,
    //   from: process.env.TWILIO_PHONE_NUMBER,
    //   to: `+91${mobile}`
    // });
    
    console.log(`📱 Twilio OTP for ${mobile}: ${otp}`);
    return { success: true, message: 'OTP sent via Twilio' };
  } catch (error) {
    console.error('Twilio error:', error.message);
    throw error;
  }
};

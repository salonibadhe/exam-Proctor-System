import mongoose from 'mongoose';
import dns from 'node:dns';

// Force the Node process to use Google DNS to resolve MongoDB SRV records
// this bypasses local network/ISP DNS issues without changing system settings.
dns.setServers(['8.8.8.8', '8.8.4.4']);
dns.setDefaultResultOrder('ipv4first');

const connectDB = async () => {
  try {
    console.log('Attempting to connect to MongoDB...');
    const redactedUri = process.env.MONGODB_URI ? process.env.MONGODB_URI.replace(/:([^@]+)@/, ':****@') : 'NOT FOUND';
    console.log('MongoDB URI:', redactedUri);

    const conn = await mongoose.connect(process.env.MONGODB_URI, {
      serverSelectionTimeoutMS: 5000,
    });

    console.log(`✅ MongoDB Connected: ${conn.connection.host}`);
    console.log(`Database: ${conn.connection.name}`);
  } catch (error) {
    console.error(`❌ MongoDB Connection Error: ${error.message}`);
    console.error('Full error:', error);
    process.exit(1);
  }
};

export default connectDB;

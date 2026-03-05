import mongoose from 'mongoose';
import dotenv from 'dotenv';

dotenv.config();

const MONGODB_URI = process.env.MONGO_URI;

async function fixEmailIndex() {
  try {
    // Connect to MongoDB
    await mongoose.connect(MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    console.log('Connected to MongoDB');

    // Get the users collection
    const usersCollection = mongoose.connection.db.collection('users');

    // Drop the existing email index
    try {
      await usersCollection.dropIndex('email_1');
      console.log('Dropped existing email_1 index');
    } catch (error) {
      console.log('No existing email_1 index to drop, creating new one...');
    }

    // Create a new index that allows multiple null values
    await usersCollection.createIndex(
      { email: 1 },
      {
        unique: true,
        partialFilterExpression: { email: { $type: 'string' } }, // Only index non-null strings
        name: 'email_1'
      }
    );

    console.log('Created new email index that allows multiple null values');
    
    process.exit(0);
  } catch (error) {
    console.error('Error fixing email index:', error);
    process.exit(1);
  }
}

fixEmailIndex();

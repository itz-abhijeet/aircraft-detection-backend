import { initializeApp } from 'firebase/app';
import { getDatabase } from 'firebase/database';

const firebaseConfig = {
  apiKey: "Google API Key",
  authDomain: "plane-tracker-d8524.firebaseapp.com",
  databaseURL: "https://plane-tracker-d8524-default-rtdb.firebaseio.com",
  projectId: "plane-tracker-d8524",
  storageBucket: "plane-tracker-d8524.firebasestorage.app",
  messagingSenderId: "528187697558",
  appId: "1:528187697558:web:c364229578cd07b30e15b7",
};

const app = initializeApp(firebaseConfig);
export const db = getDatabase(app);

// MongoDB initialization script
// This will run when the MongoDB container is first created
db = db.getSiblingDB('puresearch');

// Create collections
db.createCollection('content');
db.createCollection('search_history');

// Create indexes
db.content.createIndex({ url: 1 }, { unique: true });
db.content.createIndex({ "title": "text", "description": "text", "content": "text" });
db.content.createIndex({ confidence: 1 });
db.content.createIndex({ created_at: 1 });

db.search_history.createIndex({ query: 1 });
db.search_history.createIndex({ timestamp: 1 });

// Insert some initial data for testing
db.content.insertMany([
  {
    url: "https://example.com/classical-music-history",
    title: "The History of Classical Music - Authentic Analysis",
    description: "An in-depth exploration of classical music through the ages, with authentic analysis from leading music historians.",
    content: "Classical music has evolved over centuries, from the Baroque period with composers like Bach and Handel, through the Classical era with Mozart and Haydn, to the Romantic period with Beethoven, Chopin, and Tchaikovsky. Each period brought new innovations in composition, instrumentation, and expression...",
    confidence: 95,
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    url: "https://example.com/traditional-cooking-methods",
    title: "Traditional Cooking Methods from Around the World",
    description: "Explore authentic cooking techniques passed down through generations across different cultures and regions.",
    content: "Traditional cooking methods reflect the cultural heritage, available ingredients, and environmental conditions of regions worldwide. From clay pot cooking in Asia to tandoor ovens in India and South Asia, earth ovens in the Pacific, and smoking techniques in North America, these methods have been refined over centuries...",
    confidence: 88,
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    url: "https://example.com/travel-asia-villages",
    title: "Personal Travel Journal: Exploring Remote Villages in Asia",
    description: "A personal account of travels through remote villages in Southeast Asia, with first-hand observations and cultural insights.",
    content: "My journey through the remote villages of Southeast Asia began in northern Thailand, where I stayed with a Karen hill tribe family. Their hospitality was remarkable, offering me a place in their home despite having very little themselves. Each morning, I would wake to the sounds of the village coming to life...",
    confidence: 92,
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Create admin user
db.createUser({
  user: "puresearch_admin",
  pwd: "secure_password_here",  // Replace with actual secure password in production
  roles: [
    { role: "readWrite", db: "puresearch" },
    { role: "dbAdmin", db: "puresearch" }
  ]
}); 
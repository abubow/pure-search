from flask import Flask, request, jsonify
import logging
import os
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Mock classification function
# In a real implementation, this would use a trained ML model
def classify_content(text, url=None):
    """
    Classify content as human-written or AI-generated
    
    Args:
        text: The text content to classify
        url: Optional URL of the content source
        
    Returns:
        dict: Classification results with confidence score
    """
    # Simulate processing time
    time.sleep(0.5)
    
    # For demo purposes, use a simple heuristic:
    # Longer content gets higher human confidence scores
    text_length = len(text)
    
    if text_length < 100:
        # Very short content - suspicious
        confidence = max(50, random.randint(40, 60))
    elif text_length < 500:
        # Medium length content
        confidence = random.randint(60, 80)
    else:
        # Long content - likely human
        confidence = random.randint(75, 98)
        
    # Some content patterns that might indicate AI generation
    ai_patterns = [
        "language model",
        "as an AI",
        "I don't have personal",
        "I don't have feelings",
        "I was trained",
        "I cannot provide",
    ]
    
    # Check for AI patterns and reduce confidence if found
    for pattern in ai_patterns:
        if pattern.lower() in text.lower():
            confidence = max(30, confidence - 20)
            break
            
    return {
        "is_human": confidence >= 50,
        "confidence": confidence,
        "analysis": {
            "length": text_length,
            "complexity": random.randint(1, 10),
            "patterns_detected": any(pattern.lower() in text.lower() for pattern in ai_patterns),
        }
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "OK",
        "service": "content-classifier"
    })

@app.route('/classify', methods=['POST'])
def classify():
    """Classify text content as human or AI-generated"""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    text = data.get('text')
    url = data.get('url')
    
    if not text:
        return jsonify({"error": "No text content provided"}), 400
        
    logger.info(f"Received classification request for content (length: {len(text)})")
    
    result = classify_content(text, url)
    
    return jsonify({
        "result": result,
        "text_sample": text[:100] + "..." if len(text) > 100 else text,
        "url": url
    })

@app.route('/batch-classify', methods=['POST'])
def batch_classify():
    """Classify multiple text contents in one request"""
    data = request.json
    
    if not data or not isinstance(data, list):
        return jsonify({"error": "Invalid data format. Expected a list of items."}), 400
        
    results = []
    
    for item in data:
        text = item.get('text')
        url = item.get('url')
        
        if not text:
            results.append({"error": "No text content provided"})
            continue
            
        result = classify_content(text, url)
        
        results.append({
            "result": result,
            "text_sample": text[:100] + "..." if len(text) > 100 else text,
            "url": url
        })
    
    return jsonify(results)

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082))
    debug_mode = os.environ.get("FLASK_ENV", "development") == "development"
    
    logger.info(f"Starting Content Classifier service on port {port}")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode
    ) 
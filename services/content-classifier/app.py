from flask import Flask, request, jsonify
import logging
import os
import time
import random
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:3001"]}}, supports_credentials=True)

# Configure Swagger UI
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "PureSearch Content Classifier API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Create swagger spec directory if it doesn't exist
os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)

# Write swagger.json specification
with open(os.path.join(app.root_path, 'static/swagger.json'), 'w') as f:
    f.write('''{
  "swagger": "2.0",
  "info": {
    "title": "PureSearch Content Classifier API",
    "description": "API for classifying content as human-written or AI-generated",
    "version": "1.0.0",
    "contact": {
      "name": "API Support",
      "url": "http://puresearch.example.com/support",
      "email": "support@puresearch.example.com"
    },
    "license": {
      "name": "MIT",
      "url": "https://opensource.org/licenses/MIT"
    }
  },
  "host": "localhost:8082",
  "basePath": "/",
  "schemes": ["http"],
  "consumes": ["application/json"],
  "produces": ["application/json"],
  "paths": {
    "/health": {
      "get": {
        "summary": "Health check endpoint",
        "description": "Returns the health status of the service",
        "produces": ["application/json"],
        "responses": {
          "200": {
            "description": "Service is healthy",
            "schema": {
              "type": "object",
              "properties": {
                "status": {"type": "string"},
                "service": {"type": "string"}
              }
            }
          }
        }
      }
    },
    "/classify": {
      "post": {
        "summary": "Classify content",
        "description": "Analyze text to determine if it was written by a human or AI",
        "produces": ["application/json"],
        "consumes": ["application/json"],
        "parameters": [
          {
            "name": "body",
            "in": "body",
            "description": "Content to classify",
            "required": true,
            "schema": {
              "type": "object",
              "required": ["text"],
              "properties": {
                "text": {"type": "string", "description": "Text content to analyze"},
                "url": {"type": "string", "description": "Optional source URL"}
              }
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Classification results",
            "schema": {
              "type": "object",
              "properties": {
                "human_written": {"type": "boolean"},
                "confidence": {"type": "number", "format": "float"},
                "analysis": {
                  "type": "object",
                  "properties": {
                    "length": {"type": "integer"},
                    "complexity": {"type": "number", "format": "float"},
                    "patternsDetected": {"type": "boolean"}
                  }
                }
              }
            }
          },
          "400": {
            "description": "Bad request",
            "schema": {
              "type": "object",
              "properties": {
                "error": {"type": "string"}
              }
            }
          }
        }
      }
    }
  }
}''')

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
        "I'm an AI",
        "as a language",
        "I cannot provide",
        "I cannot browse",
        "I don't have the ability to"
    ]
    
    # Check if any patterns are in the text
    patterns_detected = any(pattern.lower() in text.lower() for pattern in ai_patterns)
    
    # Adjust confidence if patterns are detected
    if patterns_detected:
        confidence = max(10, confidence - 30)
    
    # Calculate a complexity score based on unique words ratio
    words = text.lower().split()
    unique_words = set(words)
    complexity = len(unique_words) / max(1, len(words))
    
    # Create the result
    human_written = confidence >= 70
    
    return {
        "human_written": human_written,
        "confidence": confidence,
        "analysis": {
            "length": text_length,
            "complexity": round(complexity, 2),
            "patternsDetected": patterns_detected,
            "language": "english"  # In a real implementation, would detect language
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
    """Classify a single piece of content"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing required parameter: text"}), 400
    
    text = data.get('text', '')
    url = data.get('url', None)
    
    logger.info(f"Classifying content, length: {len(text)}")
    
    # Perform classification
    result = classify_content(text, url)
    
    # Log result
    logger.info(f"Classification result: human_written={result['human_written']}, " 
                f"confidence={result['confidence']}")
    
    return jsonify(result)

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
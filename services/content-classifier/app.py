from flask import Flask, request, jsonify
import logging
import os
import time
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.corpus import wordnet
import re
import string
from typing import Dict, List, Tuple
import joblib
from langdetect import detect, LangDetectException
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, RobertaModel, RobertaTokenizer
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:3001"]}}, supports_credentials=True)

# Initialize ML components
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3),
    stop_words='english'
)

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

scaler = StandardScaler()

# Initialize models for text analysis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def preprocess_text(text: str) -> str:
    """Preprocess text for feature extraction."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def calculate_perplexity(text: str) -> float:
    """Calculate the perplexity of the text using GPT-2."""
    try:
        inputs = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = gpt2_model(**inputs)
            logits = outputs.logits
            
        loss = torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            inputs["input_ids"].view(-1)
        )
        
        perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        logger.error(f"Error in perplexity calculation: {str(e)}")
        return 0.0

def calculate_burstiness(text: str) -> float:
    """Calculate the burstiness of the text using RoBERTa embeddings."""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0
            
        embeddings = []
        for sentence in sentences:
            inputs = roberta_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = roberta_model(**inputs)
                embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                embeddings.append(embedding)
        
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(similarity)
        
        burstiness = np.std(similarities) if similarities else 0.0
        return burstiness
    except Exception as e:
        logger.error(f"Error in burstiness calculation: {str(e)}")
        return 0.0

def calculate_curvature(text: str, num_samples: int = 10) -> float:
    """Calculate the probability curvature of the text using DetectGPT approach."""
    try:
        original_prob = get_gpt2_probability(text)
        perturbed_probs = []
        words = text.split()
        
        for _ in range(num_samples):
            perturbed_words = words.copy()
            for i in range(len(perturbed_words)):
                if np.random.random() < 0.1:
                    synsets = wordnet.synsets(perturbed_words[i])
                    if synsets:
                        synonyms = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
                        if synonyms:
                            perturbed_words[i] = np.random.choice(synonyms)
            
            perturbed_text = " ".join(perturbed_words)
            perturbed_prob = get_gpt2_probability(perturbed_text)
            perturbed_probs.append(perturbed_prob)
        
        avg_perturbed_prob = np.mean(perturbed_probs)
        curvature = abs(original_prob - avg_perturbed_prob)
        
        return curvature
    except Exception as e:
        logger.error(f"Error in curvature calculation: {str(e)}")
        return 0.0

def get_gpt2_probability(text: str) -> float:
    """Calculate the probability of text being generated by GPT-2."""
    try:
        inputs = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = gpt2_model(**inputs)
            logits = outputs.logits
            
        loss = torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            inputs["input_ids"].view(-1)
        )
        
        probability = torch.exp(-loss).item()
        return probability
    except Exception as e:
        logger.error(f"Error in GPT-2 probability calculation: {str(e)}")
        return 0.5

def extract_features(text: str) -> Dict[str, float]:
    """Extract statistical and semantic features from text."""
    # Text preprocessing
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Statistical features
    unique_words = set(words)
    vocabulary_richness = len(unique_words) / word_count if word_count > 0 else 0
    
    # N-gram diversity
    bigrams = list(ngrams(words, 2))
    trigrams = list(ngrams(words, 3))
    bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
    trigram_diversity = len(set(trigrams)) / len(trigrams) if trigrams else 0
    
    # Sentence structure analysis
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0
    
    # Semantic coherence
    gpt2_prob = get_gpt2_probability(text)
    curvature = calculate_curvature(text)
    perplexity = calculate_perplexity(text)
    burstiness = calculate_burstiness(text)
    
    return {
        'vocabulary_richness': vocabulary_richness,
        'bigram_diversity': bigram_diversity,
        'trigram_diversity': trigram_diversity,
        'sentence_length_std': sentence_length_std,
        'gpt2_probability': gpt2_prob,
        'probability_curvature': curvature,
        'perplexity': perplexity,
        'burstiness': burstiness
    }

def classify_content(text: str, url: str = None) -> Dict:
    """
    Classify content as human-written or AI-generated using statistical and ML-based analysis
    """
    # Extract features
    features = extract_features(text)
    
    # Initialize base confidence
    confidence = 0.5
    
    # Analyze semantic coherence (DetectGPT)
    gpt2_prob = features['gpt2_probability']
    curvature = features['probability_curvature']
    
    # Stronger weight on semantic coherence
    if gpt2_prob > 0.65:  # More conservative threshold
        confidence = min(0.9, confidence + 0.35)
    elif gpt2_prob < 0.35:  # More conservative threshold
        confidence = max(0.1, confidence - 0.35)
    
    if curvature < 0.15:  # More sensitive to low curvature
        confidence = min(0.9, confidence + 0.25)
    elif curvature > 0.35:  # More sensitive to high curvature
        confidence = max(0.1, confidence - 0.25)
    
    # Analyze text structure (GPTZero)
    perplexity = features['perplexity']
    burstiness = features['burstiness']
    
    # Adjusted perplexity thresholds
    if perplexity > 45:  # Higher threshold for AI detection
        confidence = min(0.9, confidence + 0.3)
    elif perplexity < 25:  # Lower threshold for human detection
        confidence = max(0.1, confidence - 0.3)
    
    # Adjusted burstiness thresholds
    if burstiness < 0.25:  # More sensitive to low burstiness
        confidence = min(0.9, confidence + 0.25)
    elif burstiness > 0.45:  # More sensitive to high burstiness
        confidence = max(0.1, confidence - 0.25)
    
    # Analyze linguistic diversity with more weight
    vocab_richness = features['vocabulary_richness']
    if vocab_richness < 0.35:  # More conservative threshold
        confidence = min(0.9, confidence + 0.25)
    elif vocab_richness > 0.75:  # More conservative threshold
        confidence = max(0.1, confidence - 0.25)
    
    # Analyze n-gram diversity
    if features['trigram_diversity'] < 0.4:  # Low diversity indicates AI
        confidence = min(0.9, confidence + 0.2)
    elif features['trigram_diversity'] > 0.8:  # High diversity indicates human
        confidence = max(0.1, confidence - 0.2)
    
    # Create result
    human_written = confidence < 0.5
    
    return {
        "human_written": human_written,
        "confidence": round(confidence * 100, 2),
        "analysis": {
            "length": len(text),
            "complexity": round(features['vocabulary_richness'] * 100, 2),
            "metrics": {
                "vocabularyRichness": round(features['vocabulary_richness'] * 100, 2),
                "bigramDiversity": round(features['bigram_diversity'] * 100, 2),
                "trigramDiversity": round(features['trigram_diversity'] * 100, 2),
                "sentenceLengthStd": round(features['sentence_length_std'], 2),
                "gpt2Probability": round(features['gpt2_probability'] * 100, 2),
                "probabilityCurvature": round(features['probability_curvature'] * 100, 2),
                "perplexity": round(features['perplexity'], 2),
                "burstiness": round(features['burstiness'] * 100, 2)
            }
        }
    }

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
    "description": "API for classifying content as human-written or AI-generated using ML techniques",
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
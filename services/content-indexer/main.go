package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/gzip"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// @title          PureSearch Content Indexer API
// @version        1.0
// @description    API for indexing human-written content into Elasticsearch
// @termsOfService http://puresearch.example.com/terms/

// @contact.name  API Support
// @contact.url   http://puresearch.example.com/support
// @contact.email support@puresearch.example.com

// @license.name MIT
// @license.url  https://opensource.org/licenses/MIT

// @host      localhost:8083
// @BasePath  /

// IndexRequest represents a request to index content
type IndexRequest struct {
	URL         string `json:"url" binding:"required"`
	Title       string `json:"title" binding:"required"`
	Description string `json:"description" binding:"required"`
	Content     string `json:"content" binding:"required"`
}

// IndexResponse represents the response from an index request
type IndexResponse struct {
	ID          string    `json:"id"`
	URL         string    `json:"url"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Content     string    `json:"content"`
	Confidence  float64   `json:"confidence"`
	Timestamp   time.Time `json:"timestamp"`
}

var elasticsearchURL = "http://elasticsearch:9200"
var classifierURL = "http://content-classifier:8082"

func main() {
	// Set Gin to release mode
	gin.SetMode(gin.ReleaseMode)

	// Create a new Gin router
	router := gin.New()

	// Add gzip middleware
	router.Use(gzip.Gzip(gzip.DefaultCompression))

	// Configure CORS
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Swagger documentation
	router.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "OK",
			"service": "content-indexer",
		})
	})

	// Index endpoints
	router.POST("/index", handleIndexRequest)
	router.GET("/index/:id", handleGetIndexedContent)
	router.DELETE("/index/:id", handleDeleteIndexedContent)

	// Get port from environment variable or use default
	port := os.Getenv("PORT")
	if port == "" {
		port = "8083"
	}

	// Start the server
	log.Printf("Content Indexer Service starting on port %s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %s", err)
	}

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down Content Indexer server...")
}

// generateID creates a properly formatted UUID
func generateID() string {
	return fmt.Sprintf("idx-%s", uuid.New().String())
}

// getContentConfidence sends content to the classifier service to determine its confidence score
func getContentConfidence(content string) (float64, error) {
	// Prepare request body
	reqBody := map[string]interface{}{
		"text": content,
	}

	// Convert to JSON
	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		log.Printf("Error marshaling classifier request: %s", err)
		return 0.95, err
	}

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	// Make request to classifier service
	url := fmt.Sprintf("%s/classify", classifierURL)
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(reqJSON))
	if err != nil {
		log.Printf("Error calling classifier service: %s", err)
		return 0.95, err
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		log.Printf("Classifier service returned status code %d", resp.StatusCode)
		return 0.95, fmt.Errorf("classifier service returned status code %d", resp.StatusCode)
	}

	// Parse response
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Printf("Error decoding classifier response: %s", err)
		return 0.95, err
	}

	// Extract confidence score
	confidence, ok := result["confidence"].(float64)
	if !ok {
		log.Printf("Classifier response missing confidence score or invalid format")
		return 0.95, fmt.Errorf("invalid classifier response format")
	}

	// Convert confidence from 0-100 scale to 0-1 scale if needed
	if confidence > 1.0 {
		confidence = confidence / 100.0
	}

	log.Printf("Classified content with confidence score: %.2f", confidence)
	return confidence, nil
}

// @Summary      Index new content
// @Description  Add content to the search index
// @Tags         index
// @Accept       json
// @Produce      json
// @Param        request  body      IndexRequest  true  "Index Request"
// @Success      200      {object}  IndexResponse
// @Failure      400      {object}  map[string]string
// @Failure      500      {object}  map[string]string
// @Router       /index [post]
func handleIndexRequest(c *gin.Context) {
	var request IndexRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Generate a unique ID
	id := generateID()

	// Get content confidence score from classifier (TODO: implement real integration)
	confidence, err := getContentConfidence(request.Content)
	if err != nil {
		log.Printf("Warning: Error getting confidence score, using default: %s", err)
		confidence = 0.95
	}

	// Prepare document for indexing
	doc := map[string]interface{}{
		"id":          id,
		"title":       request.Title,
		"url":         request.URL,
		"description": request.Description,
		"content":     request.Content,
		"confidence":  confidence,
		"indexed_at":  time.Now().Format(time.RFC3339),
	}

	// Convert document to JSON
	docJSON, err := json.Marshal(doc)
	if err != nil {
		log.Printf("Error marshaling document: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error preparing document for indexing"})
		return
	}

	// Index the document
	url := fmt.Sprintf("%s/content/_doc/%s?refresh=true", elasticsearchURL, id)
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(docJSON))
	if err != nil {
		log.Printf("Error indexing document: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error indexing document"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		log.Printf("Error indexing document: status code %d", resp.StatusCode)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error indexing document"})
		return
	}

	response := IndexResponse{
		ID:          id,
		URL:         request.URL,
		Title:       request.Title,
		Description: request.Description,
		Content:     request.Content,
		Confidence:  confidence,
		Timestamp:   time.Now(),
	}

	c.JSON(http.StatusOK, response)
}

// @Summary      Get indexed content
// @Description  Retrieve details of indexed content by ID
// @Tags         index
// @Accept       json
// @Produce      json
// @Param        id   path      string  true  "Content ID"
// @Success      200  {object}  IndexResponse
// @Failure      404  {object}  map[string]string
// @Failure      500  {object}  map[string]string
// @Router       /index/{id} [get]
func handleGetIndexedContent(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "ID is required"})
		return
	}

	url := fmt.Sprintf("%s/content/_doc/%s", elasticsearchURL, id)
	resp, err := http.Get(url)
	if err != nil {
		log.Printf("Error retrieving document: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error retrieving document"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		c.JSON(http.StatusNotFound, gin.H{"error": "Document not found"})
		return
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("Error retrieving document: status code %d", resp.StatusCode)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error retrieving document"})
		return
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Printf("Error decoding response: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error decoding response"})
		return
	}

	c.JSON(http.StatusOK, result)
}

// @Summary      Delete indexed content
// @Description  Remove content from the search index
// @Tags         index
// @Accept       json
// @Produce      json
// @Param        id   path      string  true  "Content ID"
// @Success      200  {object}  map[string]string
// @Failure      404  {object}  map[string]string
// @Failure      500  {object}  map[string]string
// @Router       /index/{id} [delete]
func handleDeleteIndexedContent(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "ID is required"})
		return
	}

	url := fmt.Sprintf("%s/content/_doc/%s?refresh=true", elasticsearchURL, id)
	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		log.Printf("Error creating delete request: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error creating delete request"})
		return
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error deleting document: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error deleting document"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		c.JSON(http.StatusNotFound, gin.H{"error": "Document not found"})
		return
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("Error deleting document: status code %d", resp.StatusCode)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error deleting document"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Document deleted successfully"})
}

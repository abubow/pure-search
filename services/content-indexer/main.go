package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// IndexRequest represents a request to index content
type IndexRequest struct {
	URL         string `json:"url" binding:"required"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Content     string `json:"content" binding:"required"`
}

// IndexResponse represents the response from an index request
type IndexResponse struct {
	ID          string  `json:"id"`
	URL         string  `json:"url"`
	Title       string  `json:"title"`
	Description string  `json:"description"`
	Confidence  float64 `json:"confidence"`
	Indexed     bool    `json:"indexed"`
	Message     string  `json:"message"`
	Timestamp   string  `json:"timestamp"`
}

func main() {
	// Set Gin mode based on environment
	ginMode := os.Getenv("GIN_MODE")
	if ginMode == "" {
		ginMode = "debug"
	}
	gin.SetMode(ginMode)

	// Create Gin router with default middleware
	router := gin.Default()

	// Configure CORS
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

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

	// Get server address from environment or use default
	serverAddr := os.Getenv("SERVER_ADDR")
	if serverAddr == "" {
		serverAddr = ":8083"
	}

	// Create HTTP server
	srv := &http.Server{
		Addr:    serverAddr,
		Handler: router,
	}

	// Start server in a goroutine
	go func() {
		log.Printf("Starting Content Indexer server on %s", serverAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down Content Indexer server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Content Indexer server exited")
}

// handleIndexRequest processes requests to index new content
func handleIndexRequest(c *gin.Context) {
	var request IndexRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Generate a unique ID - in a real implementation, this would come from the database
	id := generateID()

	// In a real implementation, we would:
	// 1. Extract content from the URL if no content is provided
	// 2. Clean and process the content
	// 3. Send the content to the classifier service to get a confidence score
	// 4. Index the content in Elasticsearch or similar search engine
	// 5. Store metadata in a database

	// For now, simulate processing
	time.Sleep(500 * time.Millisecond)

	// Generate a random confidence score for this example
	confidence := 75.0 + float64(time.Now().UnixNano()%20)

	// Create a response
	response := IndexResponse{
		ID:          id,
		URL:         request.URL,
		Title:       request.Title,
		Description: request.Description,
		Confidence:  confidence,
		Indexed:     true,
		Message:     "Content successfully indexed",
		Timestamp:   time.Now().Format(time.RFC3339),
	}

	c.JSON(http.StatusOK, response)
}

// handleGetIndexedContent retrieves indexed content by ID
func handleGetIndexedContent(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "ID parameter is required"})
		return
	}

	// In a real implementation, we would fetch the content from a database
	// For now, return a mock response
	c.JSON(http.StatusOK, gin.H{
		"id":          id,
		"url":         "https://example.com/sample-" + id,
		"title":       "Sample Indexed Content " + id,
		"description": "This is a sample description for content ID " + id,
		"confidence":  85.5,
		"indexed":     true,
		"timestamp":   time.Now().Add(-24 * time.Hour).Format(time.RFC3339),
	})
}

// handleDeleteIndexedContent removes indexed content by ID
func handleDeleteIndexedContent(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "ID parameter is required"})
		return
	}

	// In a real implementation, we would remove the content from the index and database
	// For now, return a success response
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"deleted": true,
		"message": "Content successfully removed from index",
	})
}

// generateID creates a simple unique ID
// In a real implementation, this would be a UUID or similar
func generateID() string {
	return "idx-" + time.Now().Format("20060102150405") + "-" + time.Now().Format("999999999")[0:6]
} 
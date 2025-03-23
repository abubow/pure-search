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
		AllowOrigins:     []string{"http://localhost:3000", "http://localhost:3002"},
		AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "OK",
			"service": "api-gateway",
		})
	})

	// API routes
	apiV1 := router.Group("/api/v1")
	{
		// Proxy search requests to the search service
		apiV1.GET("/search", proxySearchRequest)
		
		// Content classifier endpoints
		apiV1.POST("/classify", proxyClassifyRequest)
		
		// Content indexer endpoints
		apiV1.POST("/index", proxyIndexRequest)
	}

	// Get server address from environment or use default
	serverAddr := os.Getenv("SERVER_ADDR")
	if serverAddr == "" {
		serverAddr = ":8080"
	}

	// Create HTTP server
	srv := &http.Server{
		Addr:    serverAddr,
		Handler: router,
	}

	// Start server in a goroutine
	go func() {
		log.Printf("Starting API Gateway server on %s", serverAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down API Gateway server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("API Gateway server exited")
}

// Handler to proxy search requests to the search service
func proxySearchRequest(c *gin.Context) {
	// In a real implementation, this would forward the request to the Search API service
	// For this example, we'll just return a mock response
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Query parameter 'q' is required"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"query": query,
		"results": []gin.H{
			{
				"id":          "1",
				"title":       "The History of Classical Music - Authentic Analysis",
				"url":         "https://example.com/classical-music-history",
				"description": "An in-depth exploration of classical music through the ages, with authentic analysis from leading music historians.",
				"confidence":  95,
			},
			{
				"id":          "2",
				"title":       "Traditional Cooking Methods from Around the World",
				"url":         "https://example.com/traditional-cooking-methods",
				"description": "Explore authentic cooking techniques passed down through generations across different cultures and regions.",
				"confidence":  88,
			},
		},
		"total": 2,
	})
}

// Handler to proxy content classification requests to the classifier service
func proxyClassifyRequest(c *gin.Context) {
	// In a real implementation, this would forward the request to the Content Classifier service
	c.JSON(http.StatusOK, gin.H{
		"status": "Classification request would be forwarded to the classifier service",
	})
}

// Handler to proxy indexing requests to the indexer service
func proxyIndexRequest(c *gin.Context) {
	// In a real implementation, this would forward the request to the Content Indexer service
	c.JSON(http.StatusOK, gin.H{
		"status": "Indexing request would be forwarded to the indexer service",
	})
} 
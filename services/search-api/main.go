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

// SearchResult represents a single search result
type SearchResult struct {
	ID          string  `json:"id"`
	Title       string  `json:"title"`
	URL         string  `json:"url"`
	Description string  `json:"description"`
	Confidence  float64 `json:"confidence"`
}

// SearchResponse represents the response from a search query
type SearchResponse struct {
	Query   string         `json:"query"`
	Results []SearchResult `json:"results"`
	Total   int            `json:"total"`
	Page    int            `json:"page"`
	PerPage int            `json:"per_page"`
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
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "OK",
			"service": "search-api",
		})
	})

	// Search endpoints
	router.GET("/search", handleSearch)

	// Get server address from environment or use default
	serverAddr := os.Getenv("SERVER_ADDR")
	if serverAddr == "" {
		serverAddr = ":8081"
	}

	// Create HTTP server
	srv := &http.Server{
		Addr:    serverAddr,
		Handler: router,
	}

	// Start server in a goroutine
	go func() {
		log.Printf("Starting Search API server on %s", serverAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down Search API server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Search API server exited")
}

// handleSearch processes search requests
func handleSearch(c *gin.Context) {
	// Get query parameters
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Query parameter 'q' is required"})
		return
	}

	page := 1 // Default page
	perPage := 10 // Default results per page

	// In a real implementation, we would query Elasticsearch or other search backend
	// For now, return mock data
	mockResults := []SearchResult{
		{
			ID:          "1",
			Title:       "The History of Classical Music - Authentic Analysis",
			URL:         "https://example.com/classical-music-history",
			Description: "An in-depth exploration of classical music through the ages, with authentic analysis from leading music historians.",
			Confidence:  95.0,
		},
		{
			ID:          "2",
			Title:       "Traditional Cooking Methods from Around the World",
			URL:         "https://example.com/traditional-cooking-methods",
			Description: "Explore authentic cooking techniques passed down through generations across different cultures and regions.",
			Confidence:  88.0,
		},
		{
			ID:          "3",
			Title:       "Personal Travel Journal: Exploring Remote Villages in Asia",
			URL:         "https://example.com/travel-asia-villages",
			Description: "A personal account of travels through remote villages in Southeast Asia, with first-hand observations and cultural insights.",
			Confidence:  92.0,
		},
		{
			ID:          "4",
			Title:       "Handcrafted Furniture: Techniques and Materials",
			URL:         "https://example.com/handcrafted-furniture",
			Description: "Learn about traditional woodworking techniques and materials used in creating handcrafted furniture.",
			Confidence:  76.0,
		},
		{
			ID:          "5",
			Title:       "Local Wildlife Conservation Efforts in the Amazon",
			URL:         "https://example.com/amazon-conservation",
			Description: "Documenting local efforts to preserve biodiversity in the Amazon rainforest, with reports from field researchers.",
			Confidence:  85.0,
		},
		{
			ID:          "6",
			Title:       "Historical Weather Patterns and Climate Change",
			URL:         "https://example.com/historical-weather-patterns",
			Description: "Analysis of historical weather data and how it relates to current climate change patterns.",
			Confidence:  68.0,
		},
		{
			ID:          "7",
			Title:       "Family Recipes: Mediterranean Cuisine",
			URL:         "https://example.com/mediterranean-family-recipes",
			Description: "Collection of authentic family recipes from the Mediterranean region, passed down through generations.",
			Confidence:  93.0,
		},
	}

	response := SearchResponse{
		Query:   query,
		Results: mockResults,
		Total:   len(mockResults),
		Page:    page,
		PerPage: perPage,
	}

	// Add a small delay to simulate processing time
	time.Sleep(200 * time.Millisecond)

	c.JSON(http.StatusOK, response)
} 
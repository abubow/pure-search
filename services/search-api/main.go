package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// @title          PureSearch Search API
// @version        1.0
// @description    API for searching human-written content
// @termsOfService http://puresearch.example.com/terms/

// @contact.name  API Support
// @contact.url   http://puresearch.example.com/support
// @contact.email support@puresearch.example.com

// @license.name MIT
// @license.url  https://opensource.org/licenses/MIT

// @host      localhost:8081
// @BasePath  /

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

// ElasticSearchSource represents the _source field in Elasticsearch results
type ElasticSearchSource struct {
	ID          string  `json:"id"`
	Title       string  `json:"title"`
	URL         string  `json:"url"`
	Description string  `json:"description"`
	Confidence  float64 `json:"confidence"`
	Content     string  `json:"content,omitempty"`
}

// ElasticSearchHit represents a single hit in Elasticsearch response
type ElasticSearchHit struct {
	Source ElasticSearchSource `json:"_source"`
}

// ElasticSearchResponse represents the response from Elasticsearch
type ElasticSearchResponse struct {
	Hits struct {
		Total struct {
			Value int `json:"value"`
		} `json:"total"`
		Hits []ElasticSearchHit `json:"hits"`
	} `json:"hits"`
}

var (
	esClient *elasticsearch.Client
)

func main() {
	// Initialize Elasticsearch client
	initElasticsearch()

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

	// Swagger documentation
	router.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

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

// Initialize Elasticsearch client
func initElasticsearch() {
	esURL := os.Getenv("ELASTICSEARCH_URL")
	if esURL == "" {
		esURL = "http://localhost:9200"
	}

	cfg := elasticsearch.Config{
		Addresses: []string{esURL},
	}

	var err error
	esClient, err = elasticsearch.NewClient(cfg)
	if err != nil {
		log.Fatalf("Error creating Elasticsearch client: %s", err)
	}

	// Check if Elasticsearch is reachable
	res, err := esClient.Info()
	if err != nil {
		log.Fatalf("Error connecting to Elasticsearch: %s", err)
	}
	defer res.Body.Close()

	// Check response status
	if res.IsError() {
		log.Fatalf("Error response from Elasticsearch: %s", res.String())
	}

	log.Println("Successfully connected to Elasticsearch")
}

// @Summary      Search for human-written content
// @Description  Search for content based on the query
// @Tags         search
// @Accept       json
// @Produce      json
// @Param        q        query     string  true  "Search query"
// @Param        page     query     int     false "Page number (default: 1)"
// @Param        per_page query     int     false "Results per page (default: 10)"
// @Success      200      {object}  SearchResponse
// @Failure      400      {object}  map[string]string
// @Failure      500      {object}  map[string]string
// @Router       /search [get]
func handleSearch(c *gin.Context) {
	// Get query parameters
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Query parameter 'q' is required"})
		return
	}

	pageStr := c.DefaultQuery("page", "1")
	perPageStr := c.DefaultQuery("per_page", "10")

	page, err := strconv.Atoi(pageStr)
	if err != nil || page < 1 {
		page = 1
	}

	perPage, err := strconv.Atoi(perPageStr)
	if err != nil || perPage < 1 || perPage > 100 {
		perPage = 10
	}

	// Calculate pagination
	from := (page - 1) * perPage

	// Construct Elasticsearch query
	esQuery := map[string]interface{}{
		"from": from,
		"size": perPage,
		"query": map[string]interface{}{
			"multi_match": map[string]interface{}{
				"query":  query,
				"fields": []string{"title^3", "description^2", "content"},
			},
		},
	}

	// Convert query to JSON
	queryJSON, err := json.Marshal(esQuery)
	if err != nil {
		log.Printf("Error marshaling query: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	// Execute search
	res, err := esClient.Search(
		esClient.Search.WithContext(c.Request.Context()),
		esClient.Search.WithIndex("content"),
		esClient.Search.WithBody(strings.NewReader(string(queryJSON))),
	)

	if err != nil {
		log.Printf("Error searching Elasticsearch: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Search backend error"})
		return
	}
	defer res.Body.Close()

	if res.IsError() {
		log.Printf("Elasticsearch error: %s", res.String())
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Search error: %s", res.Status())})
		return
	}

	// Parse response
	var esResp ElasticSearchResponse
	if err := json.NewDecoder(res.Body).Decode(&esResp); err != nil {
		log.Printf("Error parsing Elasticsearch response: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error processing search results"})
		return
	}

	// Convert Elasticsearch results to SearchResult
	results := make([]SearchResult, 0, len(esResp.Hits.Hits))
	for _, hit := range esResp.Hits.Hits {
		results = append(results, SearchResult{
			ID:          hit.Source.ID,
			Title:       hit.Source.Title,
			URL:         hit.Source.URL,
			Description: hit.Source.Description,
			Confidence:  hit.Source.Confidence,
		})
	}

	// Build response
	response := SearchResponse{
		Query:   query,
		Results: results,
		Total:   esResp.Hits.Total.Value,
		Page:    page,
		PerPage: perPage,
	}

	c.JSON(http.StatusOK, response)
} 
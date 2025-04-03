package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
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

	_ "search-api/docs" // Import generated docs
)

// @title          PureSearch Search API
// @version        1.0
// @description    API for searching human-written content using Elasticsearch backend
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
	PublishedAt *string `json:"published_at,omitempty"`
	ContentType *string `json:"content_type,omitempty"`
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
	IndexedAt   string  `json:"indexed_at"`
	PublishedAt *string `json:"published_at,omitempty"`
	ContentType *string `json:"content_type,omitempty"`
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
	Suggest map[string][]struct { // Added for suggestions
		Text    string `json:"text"`
		Options []struct {
			Text string `json:"text"`
		} `json:"options"`
	} `json:"suggest"`
}

// SuggestionResponse defines the structure for the suggestion endpoint
type SuggestionResponse struct {
	Suggestions []string `json:"suggestions"`
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

	// Suggest endpoint
	router.GET("/suggest", handleSuggest)

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
		Addresses:     []string{esURL},
		RetryOnStatus: []int{429, 502, 503, 504},
		MaxRetries:    5,
	}

	var err error
	esClient, err = elasticsearch.NewClient(cfg)
	if err != nil {
		log.Fatalf("Error creating Elasticsearch client: %s", err)
	}

	// Check if Elasticsearch is reachable
	res, err := esClient.Info()
	if err != nil {
		log.Printf("WARNING: Elasticsearch is not available: %s", err)
		log.Printf("Service will run in limited mode with potential errors")
		return // Continue running, but searches will fail
	}
	defer res.Body.Close()

	// Check response status
	if res.IsError() {
		log.Printf("WARNING: Elasticsearch error response: %s", res.String())
		log.Printf("Service will run in limited mode with potential errors")
		return
	}

	log.Println("Successfully connected to Elasticsearch")
}

// @Summary      Search for human-written content with filters
// @Description  Search for content based on query, with optional filters for confidence and content type
// @Tags         search
// @Accept       json
// @Produce      json
// @Param        q           query     string  true  "Search query"
// @Param        page        query     int     false "Page number (default: 1)"
// @Param        per_page    query     int     false "Results per page (default: 10)"
// @Param        refresh     query     bool    false "Force refresh of search results (default: false)"
// @Param        minConfidence query   float64 false "Minimum confidence score (0-100)"
// @Param        maxConfidence query   float64 false "Maximum confidence score (0-100)"
// @Param        contentType query   string  false "Comma-separated list of content types (e.g., article,blog)"
// @Success      200      {object}  SearchResponse
// @Failure      400      {object}  gin.H{"error": string}
// @Failure      500      {object}  gin.H{"error": string}
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
	refreshStr := c.DefaultQuery("refresh", "false")
	minConfidenceStr := c.Query("minConfidence")
	maxConfidenceStr := c.Query("maxConfidence")
	contentTypeStr := c.Query("contentType")

	page, err := strconv.Atoi(pageStr)
	if err != nil || page < 1 {
		page = 1
	}

	perPage, err := strconv.Atoi(perPageStr)
	if err != nil || perPage < 1 || perPage > 100 {
		perPage = 10
	}

	refresh, _ := strconv.ParseBool(refreshStr)

	// Parse filter parameters
	var minConfidence, maxConfidence float64
	if minConfidenceStr != "" {
		minConfidence, _ = strconv.ParseFloat(minConfidenceStr, 64)
	}
	if maxConfidenceStr != "" {
		maxConfidence, _ = strconv.ParseFloat(maxConfidenceStr, 64)
	}

	var contentTypes []string
	if contentTypeStr != "" {
		contentTypes = strings.Split(contentTypeStr, ",")
	}

	// If refresh is requested, trigger crawler service to fetch new content
	// This is done asynchronously - we don't wait for results
	if refresh {
		go triggerContentRefresh(query)
	}

	// Calculate pagination
	from := (page - 1) * perPage

	// Build a more sophisticated Elasticsearch query with scoring
	// This query does multi-field search with boosting for title and description fields
	esQuery := map[string]interface{}{
		"from": from,
		"size": perPage,
		"query": map[string]interface{}{
			"bool": map[string]interface{}{
				// Must match the query
				"must": map[string]interface{}{
					"multi_match": map[string]interface{}{
						"query":     query,
						"fields":    []string{"title^3", "description^2", "content"},
						"type":      "best_fields",
						"fuzziness": "AUTO",
					},
				},
				// Boost documents with high confidence
				"should": []map[string]interface{}{
					{
						"range": map[string]interface{}{
							"confidence": map[string]interface{}{
								"gt":    0.7,
								"boost": 2.0,
							},
						},
					},
				},
			},
		},
		// Sort by relevance score and then by indexed_at date (most recent first)
		"sort": []interface{}{
			"_score",
			map[string]interface{}{
				"indexed_at": map[string]string{
					"order": "desc",
				},
			},
		},
	}

	// Add filters
	filterQueries := []map[string]interface{}{}

	confidenceRange := map[string]float64{}
	if minConfidence > 0 {
		confidenceRange["gte"] = minConfidence
	}
	if maxConfidence > 0 && maxConfidence <= 100 {
		confidenceRange["lte"] = maxConfidence
	}
	if len(confidenceRange) > 0 {
		filterQueries = append(filterQueries, map[string]interface{}{
			"range": map[string]interface{}{"confidence": confidenceRange},
		})
	}

	if len(contentTypes) > 0 {
		filterQueries = append(filterQueries, map[string]interface{}{
			"terms": map[string]interface{}{"content_type": contentTypes},
		})
	}

	// Combine queries
	esQuery["query"] = map[string]interface{}{
		"bool": map[string]interface{}{
			"must":   esQuery["query"],
			"filter": filterQueries, // Filters applied in non-scoring context
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
		// Return empty results instead of error in case of temporary ES issues
		c.JSON(http.StatusOK, SearchResponse{
			Query:   query,
			Results: []SearchResult{},
			Total:   0,
			Page:    page,
			PerPage: perPage,
		})
		return
	}
	defer res.Body.Close()

	if res.IsError() {
		log.Printf("Elasticsearch error: %s", res.String())
		// Return empty results instead of error in case of temporary ES issues
		c.JSON(http.StatusOK, SearchResponse{
			Query:   query,
			Results: []SearchResult{},
			Total:   0,
			Page:    page,
			PerPage: perPage,
		})
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
			PublishedAt: hit.Source.PublishedAt,
			ContentType: hit.Source.ContentType,
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

// @Summary      Get search suggestions
// @Description  Provide search term suggestions based on the input query
// @Tags         search
// @Accept       json
// @Produce      json
// @Param        q query string true "Partial search query"
// @Success      200 {object} SuggestionResponse
// @Failure      400 {object} gin.H{"error": string}
// @Failure      500 {object} gin.H{"error": string}
// @Router       /suggest [get]
func handleSuggest(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Query parameter 'q' is required"})
		return
	}

	// Define the suggestion query for Elasticsearch
	suggestQuery := map[string]interface{}{
		"suggest": map[string]interface{}{
			"text": query,
			"simple_phrase": map[string]interface{}{
				"phrase": map[string]interface{}{
					"field":     "content", // Or title, or combined field
					"size":      5,         // Number of suggestions
					"gram_size": 3,
					"direct_generator": []map[string]interface{}{
						{
							"field":        "content",
							"suggest_mode": "always",
						},
					},
					"highlight": map[string]interface{}{
						"pre_tag":  "<em>",
						"post_tag": "</em>",
					},
				},
			},
		},
	}

	body, err := json.Marshal(suggestQuery)
	if err != nil {
		log.Printf("Error marshaling suggest query: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	res, err := esClient.Search(
		esClient.Search.WithContext(c.Request.Context()),
		esClient.Search.WithIndex("content"),
		esClient.Search.WithBody(strings.NewReader(string(body))),
	)

	if err != nil {
		log.Printf("Error getting suggestions from Elasticsearch: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get suggestions"})
		return
	}
	defer res.Body.Close()

	if res.IsError() {
		log.Printf("Elasticsearch suggest error: %s", res.String())
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error retrieving suggestions"})
		return
	}

	var esResp ElasticSearchResponse
	if err := json.NewDecoder(res.Body).Decode(&esResp); err != nil {
		log.Printf("Error parsing Elasticsearch suggest response: %s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error processing suggestions"})
		return
	}

	suggestions := []string{}
	if suggest, ok := esResp.Suggest["simple_phrase"]; ok && len(suggest) > 0 {
		for _, option := range suggest[0].Options {
			suggestions = append(suggestions, option.Text)
		}
	}

	c.JSON(http.StatusOK, SuggestionResponse{Suggestions: suggestions})
}

// triggerContentRefresh asynchronously requests the crawler service to refresh content for a query
func triggerContentRefresh(query string) {
	// Get crawler service URL from environment variable
	crawlerURL := os.Getenv("CRAWLER_SERVICE_URL")
	if crawlerURL == "" {
		crawlerURL = "http://crawler-service:8084"
	}

	// Build the URL with query parameter and refresh flag
	url := fmt.Sprintf("%s/search?q=%s&refresh=true", crawlerURL, url.QueryEscape(query))

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	// Make a non-blocking request to trigger the crawl
	go func() {
		resp, err := client.Get(url)
		if err != nil {
			log.Printf("Warning: Failed to trigger content refresh for query '%s': %s", query, err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			log.Printf("Warning: Received status %d when triggering content refresh for query '%s'", resp.StatusCode, query)
		}
	}()

	log.Printf("Triggered content refresh for query: %s", query)
}

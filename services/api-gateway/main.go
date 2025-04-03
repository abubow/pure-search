package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// @title          PureSearch API Gateway
// @version        1.0
// @description    Central API Gateway for all PureSearch microservices, provides unified routing and CORS support
// @termsOfService http://puresearch.example.com/terms/

// @contact.name  API Support
// @contact.url   http://puresearch.example.com/support
// @contact.email support@puresearch.example.com

// @license.name MIT
// @license.url  https://opensource.org/licenses/MIT

// @host      localhost:8080
// @BasePath  /api/v1

// ServiceProxy represents a service proxy configuration
type ServiceProxy struct {
	Name     string
	URL      string
	Endpoint string
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Status  int    `json:"status"`
	Message string `json:"message"`
	Error   string `json:"error"`
}

// Service URLs
var (
	searchAPIURL         string
	contentClassifierURL string
	contentIndexerURL    string
	crawlerServiceURL    string
)

func init() {
	// Get service URLs from environment variables
	searchAPIURL = os.Getenv("SEARCH_API_URL")
	if searchAPIURL == "" {
		searchAPIURL = "http://localhost:8081"
	}

	contentClassifierURL = os.Getenv("CONTENT_CLASSIFIER_URL")
	if contentClassifierURL == "" {
		contentClassifierURL = "http://localhost:8082"
	}

	contentIndexerURL = os.Getenv("CONTENT_INDEXER_URL")
	if contentIndexerURL == "" {
		contentIndexerURL = "http://localhost:8083"
	}

	crawlerServiceURL = os.Getenv("CRAWLER_SERVICE_URL")
	if crawlerServiceURL == "" {
		crawlerServiceURL = "http://localhost:8084"
	}
}

func getServiceURL(serviceEnvVar string) string {
	url := os.Getenv(serviceEnvVar)
	if !strings.HasPrefix(url, "http://") {
		url = "http://" + url
	}
	return url
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
		AllowOrigins:     []string{"http://localhost:3000"}, // Frontend origin - should be configurable in production
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Content-Type", "Accept", "Origin"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Swagger documentation
	router.GET("/api/v1/docs/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "OK",
			"service": "api-gateway",
		})
	})

	// API routes
	apiV1 := router.Group("/api/v1")
	{
		// Search Endpoints
		// @Summary      Search for human-written content
		// @Description  Search for content based on query and filters
		// @Tags         search
		// @Accept       json
		// @Produce      json
		// @Param        q query string true "Search query"
		// @Param        page query int false "Page number (default: 1)"
		// @Param        per_page query int false "Results per page (default: 10)"
		// @Param        refresh query bool false "Force refresh of search results (default: false)"
		// @Param        minConfidence query float64 false "Minimum confidence score (0-100)"
		// @Param        maxConfidence query float64 false "Maximum confidence score (0-100)"
		// @Param        contentType query string false "Comma-separated list of content types (e.g., article,blog)"
		// @Success      200 {object} object "SearchResponse from Search API"
		// @Failure      400 {object} ErrorResponse
		// @Failure      500 {object} ErrorResponse
		// @Router       /search [get]
		apiV1.GET("/search", proxyHandler(searchAPIURL, "/search"))

		// @Summary      Get search suggestions
		// @Description  Provide search term suggestions based on the input query
		// @Tags         search
		// @Accept       json
		// @Produce      json
		// @Param        q query string true "Partial search query"
		// @Success      200 {object} object "SuggestionResponse from Search API"
		// @Failure      400 {object} ErrorResponse
		// @Failure      500 {object} ErrorResponse
		// @Router       /suggest [get]
		apiV1.GET("/suggest", proxyHandler(searchAPIURL, "/suggest"))

		// Classify Endpoint
		// @Summary      Classify content
		// @Description  Classify text as human-written or AI-generated
		// @Tags         classify
		// @Accept       json
		// @Produce      json
		// @Param        request body object true "Classification request with text to analyze"
		// @Success      200 {object} object
		// @Failure      400 {object} ErrorResponse
		// @Failure      500 {object} ErrorResponse
		// @Router       /classify [post]
		apiV1.POST("/classify", proxyHandler(contentClassifierURL, "/classify"))

		// Index Endpoints
		// @Summary      Index content
		// @Description  Add content to the search index
		// @Tags         index
		// @Accept       json
		// @Produce      json
		// @Param        request body object true "Index request with content to add"
		// @Success      200 {object} object
		// @Failure      400 {object} ErrorResponse
		// @Failure      500 {object} ErrorResponse
		// @Router       /index [post]
		apiV1.POST("/index", proxyHandler(contentIndexerURL, "/index"))
		apiV1.GET("/index/:id", proxyHandler(contentIndexerURL, "/index"))
		apiV1.DELETE("/index/:id", proxyHandler(contentIndexerURL, "/index"))

		// Crawl Endpoints
		// @Summary      Crawl URL
		// @Description  Crawl a URL to index its content
		// @Tags         crawl
		// @Accept       json
		// @Produce      json
		// @Param        request body object true "Crawl request with URL to crawl"
		// @Success      200 {object} object
		// @Failure      400 {object} ErrorResponse
		// @Failure      500 {object} ErrorResponse
		// @Router       /crawl [post]
		apiV1.POST("/crawl", proxyHandler(crawlerServiceURL, "/crawl"))
		apiV1.GET("/crawl/status/:id", proxyHandler(crawlerServiceURL, "/crawl/status"))
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

// proxyHandler creates a reverse proxy handler for the given target URL and endpoint
func proxyHandler(targetURL, endpoint string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Remove /api/v1 prefix from the path
		path := strings.TrimPrefix(c.Request.URL.Path, "/api/v1")

		// Ensure target URL has http:// prefix
		if !strings.HasPrefix(targetURL, "http://") && !strings.HasPrefix(targetURL, "https://") {
			targetURL = "http://" + targetURL
		}

		// Create the target URL
		targetFullURL := fmt.Sprintf("%s%s", targetURL, path)

		// Log request details
		log.Printf("Original request path: %s", c.Request.URL.Path)
		log.Printf("Forwarding request to: %s", targetFullURL)
		log.Printf("Query string: %s", c.Request.URL.RawQuery)

		// Create a new request
		req, err := http.NewRequestWithContext(c.Request.Context(), c.Request.Method, targetFullURL, c.Request.Body)
		if err != nil {
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Status:  http.StatusInternalServerError,
				Message: "Error creating request",
				Error:   err.Error(),
			})
			return
		}

		// Copy headers
		for key, values := range c.Request.Header {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}

		// Copy query parameters
		req.URL.RawQuery = c.Request.URL.RawQuery

		// Create HTTP client with longer timeout
		client := &http.Client{
			Timeout: 30 * time.Second, // 30-second timeout for backend requests
		}

		// Send request
		resp, err := client.Do(req)
		if err != nil {
			c.JSON(http.StatusBadGateway, ErrorResponse{
				Status:  http.StatusBadGateway,
				Message: "Error forwarding request",
				Error:   err.Error(),
			})
			return
		}
		defer resp.Body.Close()

		// Log response status
		log.Printf("Response status: %d", resp.StatusCode)

		// Copy response headers - Skip CORS headers to avoid conflicts with our own CORS middleware
		for key, values := range resp.Header {
			// Skip CORS headers from the backend services to avoid duplication and conflicts
			if strings.HasPrefix(strings.ToLower(key), "access-control-") {
				continue
			}
			for _, value := range values {
				c.Writer.Header().Add(key, value)
			}
		}

		// Set response status
		c.Writer.WriteHeader(resp.StatusCode)

		// Copy response body
		io.Copy(c.Writer, resp.Body)
	}
}

package main

import (
	"bytes"
	"context"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
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
// @description    API Gateway for PureSearch services
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
		AllowOrigins:     []string{"http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Content-Type", "Accept"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: false,
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
		// @Summary      Search for human-written content
		// @Description  Search for content based on the query
		// @Tags         search
		// @Accept       json
		// @Produce      json
		// @Param        q        query     string  true  "Search query"
		// @Param        page     query     int     false "Page number (default: 1)"
		// @Param        per_page query     int     false "Results per page (default: 10)"
		// @Success      200      {object}  object
		// @Failure      400      {object}  ErrorResponse
		// @Failure      500      {object}  ErrorResponse
		// @Router       /search [get]
		apiV1.GET("/search", proxyHandler(searchAPIURL, "/search"))

		// @Summary      Classify content
		// @Description  Classify text as human-written or AI-generated
		// @Tags         classify
		// @Accept       json
		// @Produce      json
		// @Param        request body object true "Classification request with text to analyze"
		// @Success      200     {object} object
		// @Failure      400     {object} ErrorResponse
		// @Failure      500     {object} ErrorResponse
		// @Router       /classify [post]
		apiV1.POST("/classify", proxyHandler(contentClassifierURL, "/classify"))

		// @Summary      Index content
		// @Description  Add content to the search index
		// @Tags         index
		// @Accept       json
		// @Produce      json
		// @Param        request body object true "Index request with content to add"
		// @Success      200     {object} object
		// @Failure      400     {object} ErrorResponse
		// @Failure      500     {object} ErrorResponse
		// @Router       /index [post]
		apiV1.POST("/index", proxyHandler(contentIndexerURL, "/index"))
		apiV1.GET("/index/:id", proxyHandler(contentIndexerURL, "/index"))
		apiV1.DELETE("/index/:id", proxyHandler(contentIndexerURL, "/index"))

		// @Summary      Crawl URL
		// @Description  Crawl a URL to index its content
		// @Tags         crawl
		// @Accept       json
		// @Produce      json
		// @Param        request body object true "Crawl request with URL to crawl"
		// @Success      200     {object} object
		// @Failure      400     {object} ErrorResponse
		// @Failure      500     {object} ErrorResponse
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
		// Create a reverse proxy
		proxy := &httputil.ReverseProxy{
			Director: func(req *http.Request) {
				req.URL.Scheme = "http"
				// Remove any http:// prefix from the target URL
				targetURL = strings.TrimPrefix(targetURL, "http://")
				req.URL.Host = targetURL

				// Get the path without the /api/v1 prefix
				path := strings.TrimPrefix(req.URL.Path, "/api/v1")
				req.URL.Path = endpoint + path

				log.Printf("Original request path: %s", c.Request.URL.Path)
				log.Printf("Forwarding request to: %s://%s%s", req.URL.Scheme, req.URL.Host, req.URL.Path)
				log.Printf("Query string: %s", req.URL.RawQuery)
			},
			ModifyResponse: func(resp *http.Response) error {
				// Copy the response from the target service
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					log.Printf("Error reading response body: %v", err)
					c.JSON(http.StatusInternalServerError, ErrorResponse{
						Status:  http.StatusInternalServerError,
						Message: "Error processing response",
						Error:   err.Error(),
					})
					return err
				}

				log.Printf("Response status: %d", resp.StatusCode)

				resp.Body.Close()
				resp.Body = io.NopCloser(bytes.NewBuffer(body))
				return nil
			},
			ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
				log.Printf("Proxy error: %v", err)
				c.JSON(http.StatusBadGateway, ErrorResponse{
					Status:  http.StatusBadGateway,
					Message: "Failed to proxy request",
					Error:   err.Error(),
				})
			},
		}

		// Serve the request using the reverse proxy
		proxy.ServeHTTP(c.Writer, c.Request)
	}
}

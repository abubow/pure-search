package models

// SearchQuery represents a search request
type SearchQuery struct {
	Query   string `json:"query"`
	Page    int    `json:"page"`
	PerPage int    `json:"per_page"`
	Filters struct {
		ContentType     []string `json:"content_type"`
		MinimumConfidence int     `json:"minimum_confidence"`
		DateRange       struct {
			Start string `json:"start"`
			End   string `json:"end"`
		} `json:"date_range"`
	} `json:"filters"`
	Sort string `json:"sort"`
}

// SearchResult represents a single search result
type SearchResult struct {
	ID          string  `json:"id"`
	Title       string  `json:"title"`
	URL         string  `json:"url"`
	Description string  `json:"description"`
	Confidence  float64 `json:"confidence"`
	ContentType string  `json:"content_type"`
	PublishedAt string  `json:"published_at"`
	IndexedAt   string  `json:"indexed_at"`
}

// SearchResponse represents the response from a search query
type SearchResponse struct {
	Query   string         `json:"query"`
	Results []SearchResult `json:"results"`
	Total   int            `json:"total"`
	Page    int            `json:"page"`
	PerPage int            `json:"per_page"`
	Took    int            `json:"took"` // Time in milliseconds
}

// NewSearchQuery creates a new search query with default values
func NewSearchQuery(query string) SearchQuery {
	sq := SearchQuery{
		Query:   query,
		Page:    1,
		PerPage: 10,
		Sort:    "relevance",
	}
	sq.Filters.MinimumConfidence = 50
	return sq
} 
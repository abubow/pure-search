package models

// ClassificationRequest represents a request to classify content
type ClassificationRequest struct {
	Text string `json:"text" binding:"required"`
	URL  string `json:"url"`
}

// ClassificationResult represents the result of a classification operation
type ClassificationResult struct {
	IsHuman    bool    `json:"is_human"`
	Confidence float64 `json:"confidence"`
	Analysis   struct {
		Length           int    `json:"length"`
		Complexity       int    `json:"complexity"`
		PatternsDetected bool   `json:"patterns_detected"`
		Language         string `json:"language"`
	} `json:"analysis"`
}

// ClassificationResponse represents the response to a classification request
type ClassificationResponse struct {
	Result     ClassificationResult `json:"result"`
	TextSample string               `json:"text_sample"`
	URL        string               `json:"url"`
}

// BatchClassificationRequest represents a request to classify multiple content items
type BatchClassificationRequest []struct {
	Text string `json:"text" binding:"required"`
	URL  string `json:"url"`
}

// BatchClassificationResponse represents the response to a batch classification request
type BatchClassificationResponse []ClassificationResponse
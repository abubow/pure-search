package models

import "time"

// IndexRequest represents a request to index content
type IndexRequest struct {
	URL         string `json:"url" binding:"required"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Content     string `json:"content" binding:"required"`
	ContentType string `json:"content_type"`
	Language    string `json:"language"`
	PublishedAt string `json:"published_at"`
}

// IndexResponse represents the response from an index request
type IndexResponse struct {
	ID          string    `json:"id"`
	URL         string    `json:"url"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Confidence  float64   `json:"confidence"`
	ContentType string    `json:"content_type"`
	Language    string    `json:"language"`
	Indexed     bool      `json:"indexed"`
	Message     string    `json:"message"`
	IndexedAt   time.Time `json:"indexed_at"`
}

// IndexedContent represents content that has been indexed
type IndexedContent struct {
	ID          string    `json:"id" bson:"_id,omitempty"`
	URL         string    `json:"url" bson:"url"`
	Title       string    `json:"title" bson:"title"`
	Description string    `json:"description" bson:"description"`
	Content     string    `json:"content" bson:"content"`
	ContentType string    `json:"content_type" bson:"content_type"`
	Language    string    `json:"language" bson:"language"`
	Confidence  float64   `json:"confidence" bson:"confidence"`
	PublishedAt string    `json:"published_at" bson:"published_at"`
	IndexedAt   time.Time `json:"indexed_at" bson:"indexed_at"`
	UpdatedAt   time.Time `json:"updated_at" bson:"updated_at"`
} 
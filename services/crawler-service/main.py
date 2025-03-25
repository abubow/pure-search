from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import httpx
import asyncio
import time
import random
import logging
import os
from urllib.parse import urlparse
import json
import uuid

# Import our crawler module
from crawler import WebCrawler, search_web

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PureSearch Crawler Service",
    description="Web crawler and search service for PureSearch",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class SearchResult(BaseModel):
    id: str
    url: str
    title: str
    description: str
    content_preview: str
    confidence: float = Field(default=0.0, ge=0, le=100)
    source: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    page: int
    per_page: int
    processing_time: float

class CrawlRequest(BaseModel):
    url: HttpUrl
    depth: int = Field(default=1, ge=1, le=3)

class CrawlResponse(BaseModel):
    url: str
    success: bool
    pages_crawled: int
    pages_indexed: int
    error: Optional[str] = None

# In-memory storage for search results (replace with database in production)
indexed_pages = {}

# Dependency to get crawler instance
def get_crawler():
    return WebCrawler()

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "service": "crawler-service"
    }

@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=50, description="Results per page")
):
    """Search for content based on query"""
    if not q:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    logger.info(f"Search request: query='{q}', page={page}, per_page={per_page}")
    start_time = time.time()
    
    # Check if we have results for this query already
    query_key = q.lower().strip()
    
    if query_key not in indexed_pages:
        # Perform web search and crawl
        try:
            search_results = await search_web(query_key, num_results=10)
            
            # Store the results
            indexed_pages[query_key] = []
            
            for result in search_results:
                page_id = str(uuid.uuid4())
                indexed_result = {
                    "id": page_id,
                    "url": result["url"],
                    "title": result["title"],
                    "description": result["description"],
                    "content": result.get("content", ""),
                    "content_preview": result["content_preview"],
                    "confidence": random.uniform(75, 95),  # Mock confidence score for now
                    "source": result["source"],
                    "indexed_at": time.time()
                }
                indexed_pages[query_key].append(indexed_result)
                
        except Exception as e:
            logger.error(f"Error searching for '{query_key}': {e}")
            # Fall back to example results
            indexed_pages[query_key] = []
    
    # Get the results for this query
    results = indexed_pages.get(query_key, [])
    
    # Paginate
    total = len(results)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_results = results[start_idx:end_idx]
    
    # Convert to SearchResult objects
    search_results = [
        SearchResult(
            id=result["id"],
            url=result["url"],
            title=result["title"],
            description=result["description"],
            content_preview=result["content_preview"],
            confidence=result["confidence"],
            source=result["source"]
        )
        for result in paginated_results
    ]
    
    # Create response
    response = SearchResponse(
        query=q,
        results=search_results,
        total=total,
        page=page,
        per_page=per_page,
        processing_time=time.time() - start_time
    )
    
    return response

@app.post("/crawl")
async def crawl(
    request: CrawlRequest,
    crawler: WebCrawler = Depends(get_crawler)
):
    """Crawl and index a website"""
    logger.info(f"Crawl request for URL: {request.url}, depth: {request.depth}")
    
    try:
        crawled_data = await crawler.crawl_url(
            url=str(request.url),
            max_depth=request.depth
        )
        
        # Store crawled pages
        for page in crawled_data:
            page_id = str(uuid.uuid4())
            indexed_pages[page_id] = {
                "id": page_id,
                "url": page["url"],
                "title": page["title"],
                "description": page["description"],
                "content": page.get("content", ""),
                "content_preview": page["content_preview"],
                "confidence": random.uniform(75, 95),  # Mock confidence score
                "source": page["source"],
                "indexed_at": time.time()
            }
        
        return CrawlResponse(
            url=str(request.url),
            success=True,
            pages_crawled=len(crawled_data),
            pages_indexed=len(crawled_data)
        )
        
    except Exception as e:
        logger.error(f"Error crawling {request.url}: {e}")
        return CrawlResponse(
            url=str(request.url),
            success=False,
            pages_crawled=0,
            pages_indexed=0,
            error=str(e)
        )

# Add example pages for initial testing
def add_example_pages():
    example_urls = [
        "https://example.com/article1",
        "https://example.com/article2",
        "https://techcrunch.com/example-article",
        "https://medium.com/example-blog-post",
        "https://dev.to/example-tutorial"
    ]
    
    example_content = [
        """This is an example article about pure-search technology. 
        Pure-search is a technology that helps users find human-written content on the web.
        It uses advanced algorithms and machine learning to distinguish between AI-generated and human-written content.""",
        
        """Artificial intelligence has revolutionized many industries. However, the rise of AI-generated content 
        has made it harder to find authentic human-written material. PureSearch aims to solve this problem by 
        providing a specialized search engine that prioritizes human-written content.""",
        
        """Web crawling is an essential part of search engine technology. By systematically browsing the web, 
        search engines can index content and make it searchable. Modern crawlers respect robots.txt files and 
        implement rate limiting to avoid overloading websites.""",
        
        """The future of content creation will likely involve a combination of human writers and AI assistants. 
        Finding the right balance between AI efficiency and human creativity is the key challenge for content 
        platforms moving forward.""",
        
        """Python and FastAPI make a powerful combination for building modern web services. With async support 
        and type checking, FastAPI allows developers to create high-performance APIs with minimal boilerplate code."""
    ]
    
    titles = [
        "Introduction to Pure-Search Technology",
        "Finding Human-Written Content in the Age of AI",
        "Web Crawling Techniques for Modern Search Engines",
        "The Future of Content Creation: Humans and AI",
        "Building High-Performance APIs with Python and FastAPI"
    ]
    
    descriptions = [
        "An overview of pure-search technology and its applications in finding human-written content.",
        "How search engines can distinguish between AI-generated and human-written content.",
        "A deep dive into web crawling techniques used by modern search engines.",
        "Exploring the balance between AI assistance and human creativity in content creation.",
        "Tutorial on using Python and FastAPI to build high-performance web services."
    ]
    
    # Create a query key for example data
    query_key = "pure-search technology"
    indexed_pages[query_key] = []
    
    for i, url in enumerate(example_urls):
        page_id = str(uuid.uuid4())
        domain = urlparse(url).netloc
        
        result = {
            "id": page_id,
            "url": url,
            "title": titles[i],
            "description": descriptions[i],
            "content": example_content[i],
            "content_preview": example_content[i][:150] + "...",
            "confidence": random.uniform(80, 95),
            "source": domain,
            "indexed_at": time.time()
        }
        
        indexed_pages[query_key].append(result)
    
    logger.info(f"Added {len(example_urls)} example pages for query '{query_key}'")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Crawler Service")
    add_example_pages()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8084))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    ) 
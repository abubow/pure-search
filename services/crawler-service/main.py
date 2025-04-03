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
from datetime import datetime, timedelta
import redis.asyncio as redis
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

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
    description="Web crawler and search service for discovering and indexing human-written content",
    version="0.1.0"
)

# Get service URLs from environment variables
content_classifier_url = os.getenv("CONTENT_CLASSIFIER_URL", "http://content-classifier:8082")
content_indexer_url = os.getenv("CONTENT_INDEXER_URL", "http://content-indexer:8083")
mongodb_uri = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")
redis_uri = os.getenv("REDIS_URI", "redis://redis:6379")

# Initialize MongoDB client
mongo_client = MongoClient(mongodb_uri)
db = mongo_client["puresearch"]
crawl_queue_collection = db["crawl_queue"]
websites_collection = db["websites"]
indexed_content_collection = db["indexed_content"]

# Initialize Redis client (for message queue)
async def get_redis_pool():
    return await redis.from_url(redis_uri, encoding="utf-8", decode_responses=True)

# Redis queue names
CRAWL_QUEUE = "crawl_queue"

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
    published_date: Optional[str] = None
    content_type: Optional[str] = None

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

class ClassifyRequest(BaseModel):
    text: str
    url: Optional[str] = None

# Website management models
class Website(BaseModel):
    url: HttpUrl
    description: Optional[str] = None
    last_crawled: Optional[datetime] = None
    crawl_frequency: int = Field(default=24, description="Hours between crawls")
    max_depth: int = Field(default=2, ge=1, le=3)
    is_active: bool = Field(default=True)
    added_at: datetime = Field(default_factory=datetime.now)
    added_by: Optional[str] = None

class WebsiteResponse(BaseModel):
    id: str
    url: str
    description: Optional[str] = None
    last_crawled: Optional[datetime] = None
    crawl_frequency: int
    max_depth: int
    is_active: bool
    added_at: datetime

class WebsiteUpdate(BaseModel):
    description: Optional[str] = None
    crawl_frequency: Optional[int] = Field(default=None, ge=1)
    max_depth: Optional[int] = Field(default=None, ge=1, le=3)
    is_active: Optional[bool] = None

# Crawl job models
class CrawlJob(BaseModel):
    website_id: str
    url: HttpUrl
    max_depth: int = Field(default=2, ge=1, le=3)
    priority: int = Field(default=0, description="Higher values = higher priority")
    scheduled_time: datetime = Field(default_factory=datetime.now)
    
# TODO: Replace with persistent database storage
# In-memory storage for search results - this is temporary for development
indexed_pages = {}

# Add a global variable to track popular search terms
popular_search_terms = ["artificial intelligence", "machine learning", "web development", 
                       "programming", "python", "data science", "technology", "cloud computing"]

# Initialize a counter to rotate through popular terms
current_term_index = 0

# Dependency to get crawler instance
def get_crawler():
    return WebCrawler()

# Helper function to classify content
async def classify_content(text: str, url: Optional[str] = None) -> float:
    """
    Send text to the content classifier service
    
    Args:
        text: Text content to classify
        url: Optional URL associated with the content
        
    Returns:
        float: Confidence score (0-100)
    """
    if not text:
        return 0.0  # Cannot classify empty text
        
    try:
        request_data = ClassifyRequest(text=text, url=url)
        
        # Send request to classifier service
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{content_classifier_url}/classify",
                json=request_data.dict()
            )
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                return result.get("confidence_score", 0.0) # Return 0 if confidence not found
            else:
                logger.error(f"Error from classifier service: {response.status_code} - {response.text}")
                return 0.0 # Return low confidence on error
        
    except Exception as e:
        logger.error(f"Error classifying content: {e}")
        return 0.0 # Return low confidence on error

# Helper function to index content
async def index_content(
    url: str, 
    title: str, 
    description: str, 
    content: str, 
    confidence: float,
    published_date: Optional[str] = None,
    content_type: Optional[str] = None
) -> bool:
    """
    Send content to the indexer service
    
    Args:
        url: URL of the content
        title: Title of the content
        description: Description of the content
        content: Full text content
        confidence: Confidence score
        published_date: Optional published date string (ISO format)
        content_type: Optional content type string
        
    Returns:
        bool: True if indexing was successful
    """
    try:
        # Create the index request
        request_data = {
            "url": url,
            "title": title,
            "description": description,
            "content": content,
            "confidence": confidence,
            "published_at": published_date,
            "content_type": content_type
        }
        
        # Send request to indexer service
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{content_indexer_url}/index",
                json=request_data
            )
            
            # Check for successful response
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Error from indexer service: {response.status_code} - {response.text}")
                return False
        
    except Exception as e:
        logger.error(f"Error indexing content: {e}")
        return False

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
    per_page: int = Query(10, ge=1, le=50, description="Results per page"),
    refresh: bool = Query(False, description="Force refresh of search results")
):
    """
    Search for content based on query
    
    If results exist, returns them immediately.
    If results don't exist or refresh=True, initiates a background crawl task
    while still returning any available results.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    logger.info(f"Search request: query='{q}', page={page}, per_page={per_page}, refresh={refresh}")
    start_time = time.time()
    
    query_key = q.lower().strip()
    
    needs_refresh = refresh or query_key not in indexed_pages
    has_results = query_key in indexed_pages and len(indexed_pages[query_key]) > 0
    
    if needs_refresh:
        asyncio.create_task(crawl_for_query(query_key))
        logger.info(f"Started background crawl for query: '{query_key}'")
    
    if not has_results:
        indexed_pages[query_key] = []
        # Use first query's results as placeholder
        if len(indexed_pages) > 1: # Check if there are other results
            first_key = next(iter(k for k in indexed_pages if k != query_key)) # Get a different key
            logger.info(f"No results for '{query_key}' yet, showing placeholder results while crawling")
            temp_results = []
            for result in indexed_pages[first_key]:
                temp_result = result.copy()
                temp_result["confidence"] = max(50, temp_result["confidence"] * 0.7)
                temp_results.append(temp_result)
            indexed_pages[query_key] = temp_results
            
    results = indexed_pages.get(query_key, [])
    
    total = len(results)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_results = results[start_idx:end_idx]
    
    search_results = [
        SearchResult(
            id=result["id"],
            url=result["url"],
            title=result["title"],
            description=result["description"],
            content_preview=result["content_preview"],
            confidence=result["confidence"],
            source=result["source"],
            # Include new fields if available
            published_date=result.get("published_date"), 
            content_type=result.get("content_type")
        )
        for result in paginated_results
    ]
    
    response = SearchResponse(
        query=q,
        results=search_results,
        total=total,
        page=page,
        per_page=per_page,
        processing_time=time.time() - start_time
    )
    
    return response

async def crawl_for_query(query: str, num_results: int = 10):
    """
    Background task to crawl for a specific query and update indexed_pages
    
    Args:
        query: Search query string
        num_results: Number of search results to process
    """
    logger.info(f"Starting background crawl for query: '{query}'")
    
    try:
        search_results = await search_web(query, num_results=num_results)
        
        if query not in indexed_pages:
            indexed_pages[query] = []
        
        for result in search_results:
            page_id = str(uuid.uuid4())
            content = result.get("content", "")
            url = result["url"]
            
            if any(existing["url"] == url for existing in indexed_pages[query]):
                logger.info(f"URL already indexed, skipping: {url}")
                continue
                
            confidence = await classify_content(
                text=content,
                url=url
            )
            
            # Extract new metadata fields
            published_date = result.get("published_date")
            content_type = result.get("content_type")
            
            indexed_result = {
                "id": page_id,
                "url": url,
                "title": result["title"],
                "description": result["description"],
                "content": content,
                "content_preview": result["content_preview"],
                "confidence": confidence,
                "source": result["source"],
                "published_date": published_date,
                "content_type": content_type,
                "indexed_at": time.time()
            }
            indexed_pages[query].append(indexed_result)
            
            await index_content(
                url=url,
                title=result["title"],
                description=result["description"],
                content=content,
                confidence=confidence,
                published_date=published_date,
                content_type=content_type
            )
        
        logger.info(f"Background crawl completed for '{query}', found {len(search_results)} results")
        
    except Exception as e:
        logger.error(f"Error in background crawl for '{query}': {e}")

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
        
        successful_indexing = 0
        
        for page in crawled_data:
            page_id = str(uuid.uuid4())
            content = page.get("content", "")
            url = page["url"]
            title = page["title"]
            description = page["description"]
            published_date = page.get("published_date")
            content_type = page.get("content_type")
            
            confidence = await classify_content(
                text=content,
                url=url
            )
            
            # Store in memory (using a general key, perhaps URL, might be better than page_id)
            indexed_pages[url] = { # Use URL as key for simpler lookups
                "id": page_id,
                "url": url,
                "title": title,
                "description": description,
                "content": content,
                "content_preview": page["content_preview"],
                "confidence": confidence,
                "source": page["source"],
                "published_date": published_date,
                "content_type": content_type,
                "indexed_at": time.time()
            }
            
            if await index_content(
                url=url,
                title=title,
                description=description,
                content=content,
                confidence=confidence,
                published_date=published_date,
                content_type=content_type
            ):
                successful_indexing += 1
                
        return CrawlResponse(
            url=str(request.url),
            success=True,
            pages_crawled=len(crawled_data),
            pages_indexed=successful_indexing
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

# Website management endpoints
@app.post("/websites", response_model=WebsiteResponse)
async def add_website(website: Website):
    """
    Add a new website to be crawled regularly
    """
    try:
        # Convert to dict and ensure URL is a string
        website_data = website.dict()
        website_data["url"] = str(website.url)
        website_data["_id"] = str(uuid.uuid4())
        
        # Insert into database
        result = websites_collection.insert_one(website_data)
        
        # Create initial crawl job
        await schedule_crawl_job(
            website_id=website_data["_id"],
            url=website_data["url"],
            max_depth=website_data["max_depth"],
            priority=1  # Higher priority for new websites
        )
        
        logger.info(f"Added new website for crawling: {website_data['url']}")
        return {**website_data, "id": website_data["_id"]}
        
    except DuplicateKeyError:
        # Website URL already exists
        raise HTTPException(status_code=409, detail="Website with this URL already exists")
    except Exception as e:
        logger.error(f"Error adding website: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/websites", response_model=List[WebsiteResponse])
async def list_websites(skip: int = 0, limit: int = 100, active_only: bool = False):
    """
    List all websites in the crawl database
    """
    try:
        query = {"is_active": True} if active_only else {}
        websites = list(websites_collection.find(query).skip(skip).limit(limit))
        
        # Convert _id to id for response
        for website in websites:
            website["id"] = website.pop("_id")
            
        return websites
        
    except Exception as e:
        logger.error(f"Error listing websites: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/websites/{website_id}", response_model=WebsiteResponse)
async def get_website(website_id: str):
    """
    Get details for a specific website
    """
    website = websites_collection.find_one({"_id": website_id})
    
    if not website:
        raise HTTPException(status_code=404, detail="Website not found")
        
    website["id"] = website.pop("_id")
    return website

@app.put("/websites/{website_id}", response_model=WebsiteResponse)
async def update_website(website_id: str, update: WebsiteUpdate):
    """
    Update website settings
    """
    try:
        # Filter out None values
        update_data = {k: v for k, v in update.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid update fields provided")
            
        # Update the website
        result = websites_collection.update_one(
            {"_id": website_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Website not found")
            
        # Get the updated website
        updated = websites_collection.find_one({"_id": website_id})
        updated["id"] = updated.pop("_id")
        
        return updated
        
    except Exception as e:
        logger.error(f"Error updating website: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/websites/{website_id}")
async def delete_website(website_id: str):
    """
    Delete a website from the crawl database
    """
    result = websites_collection.delete_one({"_id": website_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Website not found")
        
    return {"status": "deleted"}

@app.post("/websites/{website_id}/crawl")
async def trigger_website_crawl(website_id: str, priority: int = Query(2, ge=0, le=5)):
    """
    Trigger an immediate crawl for a specific website
    """
    website = websites_collection.find_one({"_id": website_id})
    
    if not website:
        raise HTTPException(status_code=404, detail="Website not found")
        
    # Schedule crawl job with high priority
    await schedule_crawl_job(
        website_id=website_id,
        url=website["url"],
        max_depth=website["max_depth"],
        priority=priority
    )
    
    # Update last crawl scheduled time
    websites_collection.update_one(
        {"_id": website_id},
        {"$set": {"last_crawl_scheduled": datetime.now()}}
    )
    
    return {
        "status": "scheduled",
        "website_id": website_id,
        "url": website["url"]
    }

# Example pages for initial data (Optional, helpful for development)
def add_example_pages():
    # ... (Add example pages with new fields if needed)
    pass

# Message queue functions
async def schedule_crawl_job(website_id: str, url: str, max_depth: int = 2, priority: int = 0, delay_seconds: int = 0):
    """
    Schedule a crawl job to be executed by the crawler worker
    
    Args:
        website_id: ID of the website in the database
        url: URL to crawl
        max_depth: Maximum crawl depth
        priority: Job priority (higher = more important)
        delay_seconds: Delay before job should be executed
    """
    try:
        # Calculate scheduled time
        scheduled_time = datetime.now() + timedelta(seconds=delay_seconds)
        
        # Create crawl job
        job = {
            "_id": str(uuid.uuid4()),
            "website_id": website_id,
            "url": url,
            "max_depth": max_depth,
            "priority": priority,
            "scheduled_time": scheduled_time,
            "created_at": datetime.now(),
            "status": "pending"
        }
        
        # Add to database
        crawl_queue_collection.insert_one(job)
        
        # Add to Redis queue for immediate processing if no delay
        if delay_seconds == 0:
            redis_pool = await get_redis_pool()
            await redis_pool.lpush(CRAWL_QUEUE, json.dumps({
                "job_id": job["_id"],
                "website_id": website_id,
                "url": url,
                "max_depth": max_depth
            }))
            
        logger.info(f"Scheduled crawl job for {url}, priority={priority}, delay={delay_seconds}s")
        return job["_id"]
        
    except Exception as e:
        logger.error(f"Error scheduling crawl job: {e}")
        raise

async def process_crawl_job(job_data):
    """
    Process a crawl job from the queue
    
    Args:
        job_data: Job data from Redis queue
    """
    try:
        job_id = job_data.get("job_id")
        website_id = job_data.get("website_id")
        url = job_data.get("url")
        max_depth = job_data.get("max_depth", 2)
        
        logger.info(f"Processing crawl job {job_id} for {url} (depth={max_depth})")
        
        # Update job status
        crawl_queue_collection.update_one(
            {"_id": job_id},
            {"$set": {"status": "processing", "started_at": datetime.now()}}
        )
        
        # Initialize crawler
        crawler = WebCrawler()
        
        # Crawl the URL
        crawled_data = await crawler.crawl_url(url, max_depth=max_depth)
        
        # Process and index crawled data
        pages_indexed = 0
        for data in crawled_data:
            content = data.get("content", "")
            
            # Classify content
            confidence = await classify_content(
                text=content, 
                url=data["url"]
            )
            
            # Only index high-quality content
            if confidence > 50:
                success = await index_content(
                    url=data["url"],
                    title=data["title"],
                    description=data["description"],
                    content=content,
                    confidence=confidence,
                    published_date=data.get("published_date"),
                    content_type=data.get("content_type")
                )
                
                if success:
                    pages_indexed += 1
                    
        # Update job status
        crawl_queue_collection.update_one(
            {"_id": job_id},
            {
                "$set": {
                    "status": "completed", 
                    "completed_at": datetime.now(),
                    "pages_crawled": len(crawled_data),
                    "pages_indexed": pages_indexed
                }
            }
        )
        
        # Update website last crawl time
        if website_id:
            websites_collection.update_one(
                {"_id": website_id},
                {"$set": {"last_crawled": datetime.now()}}
            )
        
        logger.info(f"Completed crawl job {job_id}: crawled {len(crawled_data)} pages, indexed {pages_indexed}")
        
    except Exception as e:
        logger.error(f"Error processing crawl job: {e}")
        if job_id:
            crawl_queue_collection.update_one(
                {"_id": job_id},
                {"$set": {"status": "failed", "error": str(e), "failed_at": datetime.now()}}
            )

async def schedule_pending_jobs():
    """
    Check for scheduled jobs and add them to the Redis queue
    """
    try:
        # Find jobs that are scheduled to run now
        current_time = datetime.now()
        pending_jobs = crawl_queue_collection.find({
            "status": "pending",
            "scheduled_time": {"$lte": current_time}
        })
        
        redis_pool = await get_redis_pool()
        job_count = 0
        
        for job in pending_jobs:
            # Add to Redis queue
            await redis_pool.lpush(CRAWL_QUEUE, json.dumps({
                "job_id": job["_id"],
                "website_id": job.get("website_id"),
                "url": job["url"],
                "max_depth": job.get("max_depth", 2)
            }))
            
            # Update status
            crawl_queue_collection.update_one(
                {"_id": job["_id"]},
                {"$set": {"status": "queued", "queued_at": datetime.now()}}
            )
            
            job_count += 1
        
        if job_count > 0:
            logger.info(f"Added {job_count} pending jobs to the crawl queue")
        
    except Exception as e:
        logger.error(f"Error scheduling pending jobs: {e}")

async def queue_worker():
    """
    Worker process that monitors the Redis queue and processes crawl jobs
    """
    logger.info("Starting queue worker process")
    
    while True:
        try:
            redis_pool = await get_redis_pool()
            
            # Get next job from queue (blocking with 5s timeout)
            _, job_data = await redis_pool.blpop(CRAWL_QUEUE, timeout=5)
            
            if job_data:
                # Process the job
                job = json.loads(job_data)
                await process_crawl_job(job)
                
        except asyncio.TimeoutError:
            # No jobs in queue, check pending scheduled jobs
            await schedule_pending_jobs()
            
        except Exception as e:
            logger.error(f"Error in queue worker: {e}")
            await asyncio.sleep(5)  # Wait before retrying

async def schedule_future_crawls():
    """
    Schedule crawls for websites based on their crawl frequency
    """
    try:
        # Get all active websites
        active_websites = websites_collection.find({"is_active": True})
        
        for website in active_websites:
            website_id = website["_id"]
            
            # Get crawl frequency (hours)
            crawl_frequency = website.get("crawl_frequency", 24)
            
            # Get last crawl time
            last_crawled = website.get("last_crawled")
            
            # Calculate next crawl time
            if last_crawled:
                next_crawl = last_crawled + timedelta(hours=crawl_frequency)
            else:
                # If never crawled, schedule for now
                next_crawl = datetime.now()
                
            # If it's time to crawl, schedule a job
            if next_crawl <= datetime.now():
                # Calculate delay to distribute load (random 0-60 min)
                delay_seconds = random.randint(0, 3600)
                
                await schedule_crawl_job(
                    website_id=website_id,
                    url=website["url"],
                    max_depth=website.get("max_depth", 2),
                    priority=0,  # Normal priority
                    delay_seconds=delay_seconds
                )
                
                logger.info(f"Scheduled future crawl for {website['url']} with {delay_seconds}s delay")
        
    except Exception as e:
        logger.error(f"Error scheduling future crawls: {e}")

# Add continuous crawling background task
async def continuous_crawler():
    """
    Background task that continuously checks for scheduled crawls
    This runs independently of user searches to keep the index fresh
    """
    logger.info("Starting continuous crawler background task")
    
    while True:
        try:
            # Check for websites that need to be crawled based on their frequency
            await schedule_future_crawls()
            
            # Sleep for 15 minutes before next check
            await asyncio.sleep(900)
            
        except Exception as e:
            logger.error(f"Error in continuous crawler scheduler: {e}")
            await asyncio.sleep(60)

async def popular_terms_crawler():
    """
    Background task that crawls popular search terms
    This helps maintain a fresh index of popular content
    """
    global current_term_index, popular_search_terms
    
    logger.info("Starting popular terms crawler background task")
    
    while True:
        try:
            # Get next popular term
            term = popular_search_terms[current_term_index]
            current_term_index = (current_term_index + 1) % len(popular_search_terms)
            
            logger.info(f"Popular terms crawler working on: '{term}'")
            
            # Search the web for this term
            search_results = await search_web(term, num_results=5)
            
            # Schedule crawl jobs for each result
            for i, result in enumerate(search_results):
                try:
                    # Create "virtual" website entry for this URL if it doesn't exist
                    url = result["url"]
                    existing = websites_collection.find_one({"url": url})
                    
                    if existing:
                        website_id = existing["_id"]
                    else:
                        # Add as a system website with longer crawl frequency
                        website_data = {
                            "_id": str(uuid.uuid4()),
                            "url": url,
                            "description": f"Auto-discovered for term: {term}",
                            "crawl_frequency": 72,  # Crawl every 3 days
                            "max_depth": 2,
                            "is_active": True,
                            "added_at": datetime.now(),
                            "added_by": "system"
                        }
                        websites_collection.insert_one(website_data)
                        website_id = website_data["_id"]
                    
                    # Schedule with delay to distribute load
                    delay = i * 300  # 5 minutes between each job
                    await schedule_crawl_job(
                        website_id=website_id,
                        url=url,
                        max_depth=2,
                        priority=0,  # Normal priority
                        delay_seconds=delay
                    )
                    
                except Exception as e:
                    logger.error(f"Error scheduling crawl for search result {url}: {e}")
            
            # Wait longer between popular term crawls (30 minutes)
            await asyncio.sleep(1800)
            
        except Exception as e:
            logger.error(f"Error in popular terms crawler: {e}")
            await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Crawler Service")
    
    # Start the background processes
    asyncio.create_task(queue_worker())
    logger.info("Started queue worker process")
    
    asyncio.create_task(continuous_crawler())
    logger.info("Started continuous crawler scheduler")
    
    asyncio.create_task(popular_terms_crawler())
    logger.info("Started popular terms crawler")
    
    # Create necessary indexes for MongoDB collections
    logger.info("Setting up MongoDB indexes")
    try:
        # Create TTL index for completed jobs (keep for 7 days)
        crawl_queue_collection.create_index(
            [("completed_at", 1)],
            expireAfterSeconds=604800  # 7 days in seconds
        )
        
        # Create indexes for websites collection
        websites_collection.create_index([("url", 1)], unique=True)
        websites_collection.create_index([("is_active", 1)])
        websites_collection.create_index([("last_crawled", 1)])
        
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8084))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    ) 
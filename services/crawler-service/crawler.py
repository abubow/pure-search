import asyncio
import httpx
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, quote_plus
import re
import time
# TODO: Replace with a proper search API integration
# Current implementation uses direct HTTP requests to search engines
# from googlesearch import search as google_search
import random
from urllib.robotparser import RobotFileParser
import json
import dateutil.parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class WebCrawler:
    """
    Web crawler for PureSearch that discovers and extracts content from websites.
    Implements depth-limited crawling, content extraction, and metadata parsing.
    Respects robots.txt and implements rate limiting to be a good citizen.
    """
    
    def __init__(self, max_pages_per_domain=10, respect_robots=True, 
                 user_agent="PureSearch-Crawler/0.1", rate_limit_delay=1.0):
        self.max_pages_per_domain = max_pages_per_domain
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self.robots_cache = {}  # Cache for robots.txt parsers
        self.rate_limit_delay = rate_limit_delay  # Delay between requests to same domain
        self.domain_last_access = {}  # Track last access time per domain
        
    async def search_and_crawl(self, query, num_results=5):
        """
        Search the web for the query and crawl the top results
        
        Args:
            query: Search query string
            num_results: Number of search results to process
            
        Returns:
            list: Crawled content data
        """
        logger.info(f"Searching for: {query}")
        
        # Generate mock search results instead of using Google search
        # This is more reliable for development and testing
        try:
            # Create sample URLs based on the query
            search_results = [
                f"https://example.com/article/{query.replace(' ', '-')}-1",
                f"https://techblog.com/posts/{query.replace(' ', '-')}-overview",
                f"https://devdocs.io/tutorials/{query.replace(' ', '-')}-guide",
                f"https://medium.com/tech/{query.replace(' ', '-')}-explained",
                f"https://dev.to/blog/{query.replace(' ', '-')}-tips"
            ]
            
            # Limit to requested number of results
            search_results = search_results[:num_results]
            
            logger.info(f"Generated {len(search_results)} mock search results for '{query}'")
            
            # Crawl and index each result
            crawled_data = []
            
            # Create mock data for each result
            for i, url in enumerate(search_results):
                try:
                    domain = urlparse(url).netloc
                    path_parts = urlparse(url).path.split('/')
                    title_base = path_parts[-1].replace('-', ' ').title() if path_parts else query.title()
                    
                    # Create mock metadata and content
                    metadata = {
                        "title": f"{title_base} - {domain}",
                        "description": f"This is a sample description for {query} from {domain}"
                    }
                    
                    text_content = f"""
                    This is sample content for the search query "{query}". 
                    This page is from {domain} and discusses various aspects of {query}.
                    It includes information about best practices, examples, and resources related to {query}.
                    The content is generated for testing purposes and would be replaced by actual crawled content in production.
                    """
                        
                        crawled_data.append({
                            "url": url,
                            "title": metadata["title"],
                            "description": metadata["description"],
                            "content": text_content,
                            "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                        "source": domain,
                        })
                        
                    logger.info(f"Created mock data for: {url}")
                        
                    except Exception as e:
                    logger.error(f"Error creating mock data for {url}: {e}")
                        continue
            
            logger.info(f"Successfully created {len(crawled_data)} mock results for '{query}'")
            return crawled_data
            
        except Exception as e:
            logger.error(f"Error in search and crawl: {e}")
            return []
            
    async def crawl_url(self, url, max_depth=1):
        """
        Crawl a specific URL and collect links up to max_depth
        
        Args:
            url: Starting URL to crawl
            max_depth: Maximum link depth to crawl (1=just the page, 2=page+links, etc.)
            
        Returns:
            list: Crawled content data
        """
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
        
        # URLs to crawl: (url, depth)
        to_crawl = [(url, 0)]
        crawled = set()
        crawled_data = []
        
        async with httpx.AsyncClient(
            timeout=15.0, 
            follow_redirects=True,
            headers={"User-Agent": self.user_agent}
        ) as client:
            while to_crawl and len(crawled) < self.max_pages_per_domain:
                current_url, depth = to_crawl.pop(0)
                
                if current_url in crawled:
                    continue
                    
                # Check if we're allowed to crawl this URL
                if self.respect_robots and not await self._can_fetch(current_url):
                    logger.info(f"Skipping {current_url} (disallowed by robots.txt)")
                    continue
                
                # Apply rate limiting
                await self._apply_rate_limit(current_url)
                    
                logger.info(f"Crawling {current_url} (depth {depth})")
                crawled.add(current_url)
                
                try:
                    # Fetch the page
                    response = await client.get(current_url)
                    response.raise_for_status()
                    html = response.text
                    
                    # Extract metadata and content
                    metadata = self._extract_metadata(html, current_url)
                    text_content = self._extract_text_content(html)
                    
                    # Skip pages with too little content
                    if len(text_content) < 100:
                        logger.info(f"Skipping {current_url} (insufficient content)")
                        continue
                    
                    crawled_data.append({
                        "url": current_url,
                        "title": metadata["title"],
                        "description": metadata["description"],
                        "content": text_content,
                        "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                        "source": urlparse(current_url).netloc,
                    })
                    
                    # Stop if we reached max depth
                    if depth >= max_depth:
                        continue
                        
                    # Extract and queue new links
                    links = self._extract_links(html, current_url)
                    
                    # Prioritize links from same domain
                    same_domain_links = [
                        link for link in links 
                        if urlparse(link).netloc == base_domain
                    ]
                    
                    # Add links to crawl queue
                    for link in same_domain_links[:5]:  # Limit links per page
                        if link not in crawled:
                            to_crawl.append((link, depth + 1))
                            
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error crawling {current_url}: {e.response.status_code}")
                    continue
                except httpx.RequestError as e:
                    logger.error(f"Request error crawling {current_url}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {e}")
                    continue
        
        return crawled_data
    
    async def _can_fetch(self, url):
        """Check if the URL can be fetched according to robots.txt rules"""
        if not self.respect_robots:
            return True
            
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            scheme = parsed_url.scheme
            
            # Check if we have a cached robots parser for this domain
            if domain not in self.robots_cache:
                # Create a new parser
                parser = RobotFileParser()
                robots_url = f"{scheme}://{domain}/robots.txt"
                
                # Fetch and parse robots.txt
                async with httpx.AsyncClient(timeout=5.0) as client:
                    try:
                        response = await client.get(robots_url)
                        if response.status_code == 200:
                            parser.parse(response.text.splitlines())
                        else:
                            # If robots.txt doesn't exist or can't be fetched, assume we can crawl
                            parser.allow_all = True
                    except Exception:
                        # If there's an error, assume we can crawl
                        parser.allow_all = True
                
                # Cache the parser
                self.robots_cache[domain] = parser
            
            # Check if our user agent is allowed to fetch this URL
            path = parsed_url.path or "/"
            return self.robots_cache[domain].can_fetch(self.user_agent, path)
            
        except Exception as e:
            logger.error(f"Error checking robots.txt: {e}")
            # In case of error, we'll be cautious and allow the crawl
            return True
    
    async def _apply_rate_limit(self, url):
        """Apply rate limiting to avoid overwhelming the target website"""
        domain = urlparse(url).netloc
        current_time = time.time()
        
        # Check when we last accessed this domain
        if domain in self.domain_last_access:
            last_access_time = self.domain_last_access[domain]
            time_since_last_access = current_time - last_access_time
            
            # If we accessed it too recently, sleep for the remaining delay
            if time_since_last_access < self.rate_limit_delay:
                delay = self.rate_limit_delay - time_since_last_access
                logger.debug(f"Rate limiting: sleeping for {delay:.2f}s before accessing {domain}")
                await asyncio.sleep(delay)
        
        # Update the last access time
        self.domain_last_access[domain] = time.time()
        
    def _extract_links(self, html, base_url):
        """Extract links from HTML content"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href', '')
                if href and not href.startswith('#') and not href.startswith('javascript:'):
                    # Handle relative URLs
                    absolute_url = urljoin(base_url, href)
                    # Remove fragments and queries
                    absolute_url = re.sub(r'#.*$', '', absolute_url)
                    absolute_url = re.sub(r'\?.*$', '', absolute_url)
                    
                    # Only include http/https URLs
                    if absolute_url.startswith(('http://', 'https://')):
                        links.append(absolute_url)
                        
            return list(set(links))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}")
            return []
            
    def _extract_text_content(self, html):
        """Extract readable text content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "header", "footer", "nav"]):
                element.extract()
            
            # Get text from main content areas
            main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile("content|main|article"))
            
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback to body content
                text = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return ""
            
    def _extract_metadata(self, html, url):
        """Extract metadata like title, description, published date, and content type from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = ""
            # First try Open Graph title
            og_title = soup.find("meta", property="og:title")
            if og_title:
                title = og_title.get("content", "")
            
            # If no OG title, try the title tag
            if not title and soup.title:
                title = soup.title.string
                
            title = title.strip() if title else urlparse(url).netloc
            
            # Extract description
            description = ""
            
            # Try Open Graph description first
            og_desc = soup.find("meta", property="og:description")
            if og_desc:
                description = og_desc.get("content", "")
                
            # If no OG description, try meta description
            if not description:
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc:
                    description = meta_desc.get("content", "")
            
            # If still no description, use the first paragraph
            if not description:
                first_p = soup.find("p")
                if first_p:
                    description = first_p.get_text(strip=True)
                    
            # Limit description length
            description = description[:200] + "..." if len(description) > 200 else description
            
            # Extract published date
            published_date = None
            # Try various common meta tags and attributes
            date_meta_tags = [
                ("meta", {"property": "article:published_time"}),
                ("meta", {"name": "pubdate"}),
                ("meta", {"name": "date"}),
                ("meta", {"itemprop": "datePublished"}),
                ("time", {"itemprop": "datePublished"}),
                ("time", {"datetime": True})
            ]
            for tag_name, attrs in date_meta_tags:
                tag = soup.find(tag_name, attrs=attrs)
                if tag:
                    date_str = tag.get("content") or tag.get("datetime") or tag.get_text()
                    try:
                        published_date = dateutil.parser.parse(date_str).isoformat()
                        break # Found a date, stop searching
                    except (ValueError, TypeError, dateutil.parser.ParserError):
                        continue
            
            # Extract content type (simple guess based on URL or tags)
            content_type = "unknown"
            og_type = soup.find("meta", property="og:type")
            if og_type:
                og_type_content = og_type.get("content", "").lower()
                if og_type_content == "article":
                    content_type = "article"
                elif og_type_content == "book":
                    content_type = "book"
            else:
                # Simple heuristic based on URL
                path = urlparse(url).path.lower()
                if "/blog/" in path or "/post/" in path:
                    content_type = "blog"
                elif "/article/" in path or ".pdf" in path:
                    content_type = "article"
                elif "/docs/" in path or "/guide/" in path:
                    content_type = "documentation"
                elif "/product/" in path or "/item/" in path:
                    content_type = "product"
            
            return {
                "title": title, 
                "description": description,
                "published_date": published_date,
                "content_type": content_type
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {e}")
            return {
                "title": urlparse(url).netloc, 
                "description": "",
                "published_date": None,
                "content_type": "unknown"
            }

# Utility function for searching
async def search_web(query, num_results=5):
    """
    Search the web for the given query and return crawled data
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        list: Crawled data from search results
    """
    # For development purposes, you can switch between real and mock search
    # Set to False to use mock data or True to use real search
    use_real_search = True
    
    if not use_real_search:
        return await _mock_search(query, num_results)
    else:
        logger.info(f"Performing real web search for query: {query}")
        try:
            # Create a list to store results from multiple sources
            search_results = []
            
            # Try searching with multiple engines to increase reliability
            crawled_results = await _search_duckduckgo(query, num_results)
            if crawled_results:
                search_results.extend(crawled_results)
                logger.info(f"Found {len(crawled_results)} results from DuckDuckGo")
            
            # If we don't have enough results, try another search engine
            if len(search_results) < num_results:
                remaining = num_results - len(search_results)
                wikipedia_results = await _search_wikipedia(query, remaining)
                if wikipedia_results:
                    search_results.extend(wikipedia_results)
                    logger.info(f"Added {len(wikipedia_results)} results from Wikipedia")
            
            # If we still don't have enough results, use our mock implementation
            if len(search_results) < num_results:
                logger.info(f"Insufficient results, adding mock data")
                mock_results = await _mock_search(query, num_results - len(search_results))
                search_results.extend(mock_results)
            
            # Crawl each URL to get content
    crawler = WebCrawler()
            crawled_data = []
            
            # Process each result concurrently for efficiency
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "PureSearch-Crawler/0.1"}
            ) as client:
                tasks = []
                for result in search_results[:num_results]:
                    tasks.append(_process_url(client, crawler, result))
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions
                for item in results:
                    if isinstance(item, dict) and "url" in item:
                        crawled_data.append(item)
            
            logger.info(f"Successfully gathered {len(crawled_data)} results for '{query}'")
            return crawled_data
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            # Fallback to mock search if real search fails
            return await _mock_search(query, num_results)

async def _search_duckduckgo(query, num_results=5):
    """Use DuckDuckGo for web search"""
    search_results = []
    
    try:
        encoded_query = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select(".result")
            
            for i, result in enumerate(results):
                if i >= num_results:
                    break
                    
                title_elem = result.select_one(".result__title")
                link_elem = result.select_one(".result__url")
                snippet_elem = result.select_one(".result__snippet")
                
                if title_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    result_url = link_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # Make sure URL is properly formed
                    if not result_url.startswith(('http://', 'https://')):
                        result_url = 'https://' + result_url
                    
                    search_results.append({
                        "url": result_url,
                        "title": title,
                        "description": snippet
                    })
        
        return search_results
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {e}")
        return []

async def _search_wikipedia(query, num_results=3):
    """Search Wikipedia for content"""
    search_results = []
    
    try:
        encoded_query = quote_plus(query)
        url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded_query}&limit={num_results}&namespace=0&format=json"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Wikipedia API returns data in format [query, [titles], [descriptions], [urls]]
            if len(data) >= 4:
                titles = data[1]
                descriptions = data[2]
                urls = data[3]
                
                for i in range(min(len(titles), num_results)):
                    search_results.append({
                        "url": urls[i],
                        "title": titles[i],
                        "description": descriptions[i]
                    })
        
        return search_results
    except Exception as e:
        logger.error(f"Error in Wikipedia search: {e}")
        return []

async def _process_url(client, crawler, result):
    """Process a single URL to get its content"""
    url = result["url"]
    try:
        # Apply rate limiting
        await crawler._apply_rate_limit(url)
        
        # Get the webpage content
        response = await client.get(url)
        response.raise_for_status()
        html = response.text
        
        # Extract metadata and content
        metadata = crawler._extract_metadata(html, url)
        text_content = crawler._extract_text_content(html)
        
        # Skip pages with too little content
        if len(text_content) < 100:
            logger.info(f"Skipping {url} (insufficient content)")
            raise ValueError("Insufficient content")
        
        # Ensure description is never empty
        description = metadata["description"] or result.get("description", "")
        if not description:
            # Create a fallback description from the content
            description = text_content[:150] + "..." if len(text_content) > 150 else text_content
        
        return {
            "url": url,
            "title": metadata["title"] or result["title"],
            "description": description,
            "content": text_content,
            "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content,
            "source": urlparse(url).netloc,
        }
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        # Return None and filter it out later
        return None

async def _mock_search(query, num_results=5):
    """
    Generate mock search results for a query
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        list: Mock crawled data
    """
    logger.info(f"Using mock search implementation for query: {query}")
    
    try:
        # Create sample results
        mock_results = []
        
        # Create sample URLs based on the query
        search_urls = [
            f"https://example.com/article/{query.replace(' ', '-')}-1",
            f"https://techblog.com/posts/{query.replace(' ', '-')}-overview",
            f"https://devdocs.io/tutorials/{query.replace(' ', '-')}-guide",
            f"https://medium.com/tech/{query.replace(' ', '-')}-explained",
            f"https://dev.to/blog/{query.replace(' ', '-')}-tips"
        ]
        
        # Limit to requested number of results
        search_urls = search_urls[:num_results]
        
        for i, url in enumerate(search_urls):
            domain = urlparse(url).netloc
            path_parts = urlparse(url).path.split('/')
            title_base = path_parts[-1].replace('-', ' ').title() if path_parts else query.title()
            
            # Create mock content
            text_content = f"""
            This is mock content for the search query "{query}". 
            This page is from {domain} and discusses various aspects of {query}.
            It includes information about best practices, examples, and resources related to {query}.
            The content is generated for testing purposes and would be replaced by actual crawled content in production.
            """
            
            mock_results.append({
                "url": url,
                "title": f"{title_base} - {domain}",
                "description": f"This is a sample description for {query} from {domain}",
                "content": text_content,
                "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                "source": domain,
            })
            
        logger.info(f"Created {len(mock_results)} mock search results for '{query}'")
        return mock_results
        
    except Exception as e:
        logger.error(f"Error in mock search: {e}")
        return []

# Test function 
async def main():
    """Test function"""
    query = "fastapi web development tutorial"
    results = await search_web(query)
    
    print(f"Found {len(results)} results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Description: {result['description']}")
        print()
        
if __name__ == "__main__":
    asyncio.run(main()) 
import asyncio
import httpx
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import time
from googlesearch import search as google_search
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class WebCrawler:
    """Web crawler for PureSearch"""
    
    def __init__(self, max_pages_per_domain=10, respect_robots=True, user_agent="PureSearch-Crawler/0.1"):
        self.max_pages_per_domain = max_pages_per_domain
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self.robots_cache = {}
        
    async def search_and_crawl(self, query, num_results=5):
        """
        Search the web for the query and crawl the top results
        """
        logger.info(f"Searching for: {query}")
        
        # Use Google search to find initial URLs
        try:
            search_results = list(google_search(
                query, 
                num_results=num_results,
                lang="en",
                advanced=True
            ))
            
            logger.info(f"Found {len(search_results)} search results for '{query}'")
            
            # Crawl and index each result
            crawled_data = []
            
            async with httpx.AsyncClient(
                timeout=15.0, 
                follow_redirects=True,
                headers={"User-Agent": self.user_agent}
            ) as client:
                for url in search_results:
                    try:
                        # Get the webpage content
                        response = await client.get(url)
                        response.raise_for_status()
                        html = response.text
                        
                        # Extract metadata and content
                        metadata = self._extract_metadata(html, url)
                        text_content = self._extract_text_content(html)
                        
                        crawled_data.append({
                            "url": url,
                            "title": metadata["title"],
                            "description": metadata["description"],
                            "content": text_content,
                            "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content,
                            "source": urlparse(url).netloc,
                        })
                        
                        logger.info(f"Crawled: {url}")
                        
                    except Exception as e:
                        logger.error(f"Error crawling {url}: {e}")
                        continue
            
            return crawled_data
            
        except Exception as e:
            logger.error(f"Error in search and crawl: {e}")
            return []
            
    async def crawl_url(self, url, max_depth=1):
        """
        Crawl a specific URL and collect links up to max_depth
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
                            
                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {e}")
                    continue
        
        return crawled_data
        
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
        """Extract metadata like title and description from HTML"""
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
            
            return {
                "title": title, 
                "description": description
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {e}")
            return {"title": urlparse(url).netloc, "description": ""}

# Utility function for searching
async def search_web(query, num_results=5):
    """Search the web for the given query and return crawled data"""
    crawler = WebCrawler()
    return await crawler.search_and_crawl(query, num_results)

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
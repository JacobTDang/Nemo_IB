from ddgs import DDGS
from typing import List, Dict, Any, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse, urljoin
import re
from collections import Counter
import hashlib
from datetime import datetime
import sys
class SessionManager:
  """Manages persistent HTTP sessions for improved performance"""

  def __init__(self):
    self.sessions = {}

  def get_session(self, domain: str) -> requests.Session:
    """Get or create a session for a specific domain"""
    if domain not in self.sessions:
      session = requests.Session()
      # Configure session with adapter for retries at connection level
      adapter = HTTPAdapter(
        max_retries=Retry(
          total=3,
          backoff_factor=0.3,
          status_forcelist=[500, 502, 503, 504]
        )
      )
      session.mount('http://', adapter)
      session.mount('https://', adapter)
      self.sessions[domain] = session
    return self.sessions[domain]

  def close_all(self):
    """Close all open sessions"""
    for session in self.sessions.values():
      session.close()
    self.sessions.clear()

# Global session manager
_session_manager = SessionManager()

def search_duckduckgo(query: str, max_results = 5) -> List[Dict]:
  """Enhanced DuckDuckGo search with URL validation"""
  try:
    search_results = []

    # Create targeted financial queries using advanced operators
    financial_queries = [
      f"{query} site:sec.gov",                    # SEC filings
      f"{query} site:investor.",                  # Company investor relations
      f"{query} site:reuters.com",                # Reuters financial news
      f"{query} site:apnews.com",                 # AP News business
      f"{query} -site:bloomberg.com -site:wsj.com"  # Exclude paywalls
    ]

    # Search each targeted query
    results_per_query = max(1, max_results // len(financial_queries))

    with DDGS() as ddgs:
      for targeted_query in financial_queries:
        try:
          for result in ddgs.text(targeted_query, max_results=results_per_query):
            link = result.get('href', '')
            # Validate URL before adding
            if link and is_valid_url(link):
              search_results.append({
                'title': result.get('title', ''),
                'link': link,
                'snippet': result.get('body',''),
                'source': 'duckduckgo',
                'query_type': targeted_query.split()[0] if 'site:' in targeted_query else 'general'
              })
        except:
          continue  # Skip if individual query fails

    # Remove duplicates based on URL
    seen_urls = set()
    unique_results = []
    for result in search_results:
      if result['link'] not in seen_urls:
        seen_urls.add(result['link'])
        unique_results.append(result)

    return unique_results[:max_results]

  except Exception as e:
    print(f"DuckDuckGo search failed: {e}", file=sys.stderr, flush=True)
    return []

def is_valid_url(url: str) -> bool:
  """Validate URL has proper scheme and structure"""
  try:
    result = urlparse(url)
    return all([result.scheme, result.netloc])
  except:
    return False

def calculate_content_quality_score(content: str) -> Dict[str, Any]:
  """
  Calculate quality metrics for extracted content
  Returns a score and detailed metrics
  """
  if not content:
    return {'score': 0, 'metrics': {}, 'quality': 'empty'}

  # Calculate various metrics
  words = content.split()
  word_count = len(words)
  char_count = len(content)

  # Sentence detection (basic)
  sentences = re.split(r'[.!?]+', content)
  sentence_count = len([s for s in sentences if len(s.strip()) > 10])

  # Average word length (indicator of content vs navigation)
  avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0

  # Unique words ratio (higher is better, indicates diverse content)
  unique_words = len(set(w.lower() for w in words))
  unique_ratio = unique_words / word_count if word_count > 0 else 0

  # Check for financial keywords
  financial_keywords = [
    'revenue', 'earnings', 'profit', 'loss', 'shares', 'stock', 'market',
    'quarter', 'fiscal', 'guidance', 'outlook', 'eps', 'ebitda', 'cash flow',
    'debt', 'equity', 'dividend', 'investor', 'shareholder', 'sec', 'filing'
  ]
  content_lower = content.lower()
  keyword_matches = sum(1 for keyword in financial_keywords if keyword in content_lower)

  # Calculate composite score (0-100)
  score = 0

  # Word count (0-40 points)
  if word_count >= 500:
    score += 40
  elif word_count >= 200:
    score += 30
  elif word_count >= 100:
    score += 20
  elif word_count >= 50:
    score += 10

  # Unique ratio (0-20 points)
  score += int(unique_ratio * 20)

  # Sentence structure (0-20 points)
  if sentence_count >= 10:
    score += 20
  elif sentence_count >= 5:
    score += 15
  elif sentence_count >= 2:
    score += 10

  # Financial relevance (0-20 points)
  score += min(keyword_matches * 2, 20)

  # Determine quality level
  if score >= 80:
    quality = 'excellent'
  elif score >= 60:
    quality = 'good'
  elif score >= 40:
    quality = 'fair'
  elif score >= 20:
    quality = 'poor'
  else:
    quality = 'very_poor'

  return {
    'score': score,
    'quality': quality,
    'metrics': {
      'word_count': word_count,
      'char_count': char_count,
      'sentence_count': sentence_count,
      'unique_words': unique_words,
      'unique_ratio': round(unique_ratio, 3),
      'avg_word_length': round(avg_word_length, 2),
      'financial_keywords': keyword_matches
    }
  }

def clean_extracted_text(text: str) -> str:
  """Advanced text cleaning and normalization"""
  if not text:
    return ""

  # Remove excessive whitespace
  text = re.sub(r'\s+', ' ', text)

  # Remove common navigation artifacts
  noise_patterns = [
    r'skip to (?:main )?content',
    r'click here to (?:read more|continue)',
    r'share this article',
    r'sign up for (?:our )?newsletter',
    r'follow us on \w+',
    r'cookie (?:policy|notice|preferences)',
    r'accept (?:all )?cookies',
    r'privacy policy',
    r'terms (?:of service|and conditions)',
    r'all rights reserved',
    r'back to top',
    r'related (?:articles|stories)',
  ]

  for pattern in noise_patterns:
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

  # Remove email addresses (privacy)
  text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email]', text)

  # Remove phone numbers
  text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[phone]', text)

  # Normalize multiple punctuation
  text = re.sub(r'([.!?])\1+', r'\1', text)

  # Clean up spaces around punctuation
  text = re.sub(r'\s+([.,!?;:])', r'\1', text)
  text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)

  # Remove leading/trailing whitespace
  text = text.strip()

  return text

def get_headers_for_site(url: str) -> Dict[str, str]:
  """Return appropriate headers based on the target site"""

  # SEC.gov requires specific headers
  if 'sec.gov' in url:
    return {
      'User-Agent': 'MyFinancialApp/1.0 (yourname@example.com)',  # SEC requires email in User-Agent
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
      'Connection': 'keep-alive',
      'Upgrade-Insecure-Requests': '1',
      'Cache-Control': 'max-age=0',
    }

  # Generic headers for other sites
  return {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Cache-Control': 'max-age=0',
  }

def web_scrape(url: str, max_retries: int = 3, retry_delay: float = 1.0, verbose: bool = True) -> Dict[str, Any]:
  """
  Enhanced web scraping with retry logic and site-specific handling

  Args:
    url: URL to scrape
    max_retries: Maximum number of retry attempts
    retry_delay: Delay between retries in seconds
    verbose: Whether to print detailed extraction information
  """

  # Validate URL first
  if not url or not is_valid_url(url):
    return {
      'success': False,
      'error': f'Invalid URL: {url}',
      'url': url
    }

  # Extract domain for session management
  parsed_url = urlparse(url)
  domain = parsed_url.netloc

  # Try scraping with retries
  last_error = None

  for attempt in range(max_retries):
    try:
      # Get session for this domain
      session = _session_manager.get_session(domain)

      # Get appropriate headers for this site
      headers = get_headers_for_site(url)

      # Add delay between retries (but not on first attempt)
      if attempt > 0:
        time.sleep(retry_delay * attempt)  # Exponential backoff
        if verbose:
          print(f"  Retry attempt {attempt + 1}/{max_retries}", file=sys.stderr, flush=True)

      # Make request with timeout
      if verbose:
        print(f"  Fetching URL...", file=sys.stderr, flush=True)

      response = session.get(
        url,
        headers=headers,
        timeout=20,
        allow_redirects=True
      )

      # Handle different status codes
      if response.status_code == 403:
        # Try with different User-Agent on 403
        if attempt < max_retries - 1:
          headers['User-Agent'] = f'Mozilla/5.0 (compatible; FinancialBot/1.0; +http://example.com/bot)'
          continue

      response.raise_for_status()  # Raise exception for bad status codes

      if verbose:
        print(f"  Response received: {response.status_code} ({len(response.content)} bytes)", file=sys.stderr, flush=True)

      # Check if content is HTML
      content_type = response.headers.get('content-type', '').lower()
      if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
        return {
          'success': False,
          'error': f'Not HTML content: {content_type}',
          'url': url
        }

      # Detect encoding
      encoding = response.encoding if response.encoding else 'utf-8'

      if verbose:
        print(f"  Parsing HTML (encoding: {encoding})...", file=sys.stderr, flush=True)

      # Parse HTML content
      soup = BeautifulSoup(response.content, 'html.parser', from_encoding=encoding)

      # Get page title
      page_title = soup.title.string.strip() if soup.title and soup.title.string else 'No title'

      # Remove unwanted elements
      removed_elements = 0
      for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'iframe', 'noscript']):
        element.decompose()
        removed_elements += 1

      if verbose:
        print(f"  Removed {removed_elements} non-content elements", file=sys.stderr, flush=True)
        print(f"  Extracting main content...", file=sys.stderr, flush=True)

      # Extract main content based on site type
      main_content, extraction_method = extract_financial_content_advanced(soup, url, verbose=verbose)

      # Clean the extracted text
      main_content = clean_extracted_text(main_content)

      # Calculate content quality
      quality_info = calculate_content_quality_score(main_content)

      if verbose:
        print(f"  Extraction method: {extraction_method}", file=sys.stderr, flush=True)
        print(f"  Content quality: {quality_info['quality']} (score: {quality_info['score']}/100)", file=sys.stderr, flush=True)
        print(f"  Words: {quality_info['metrics']['word_count']}, Sentences: {quality_info['metrics']['sentence_count']}", file=sys.stderr, flush=True)
        print(f"  Financial keywords found: {quality_info['metrics']['financial_keywords']}", file=sys.stderr, flush=True)

      # Basic content validation
      if len(main_content) < 100:
        if attempt < max_retries - 1:
          last_error = 'Content too short, retrying...'
          if verbose:
            print(f"  WARNING: Content too short ({len(main_content)} chars), retrying...", file=sys.stderr, flush=True)
          continue
        return {
          'success': False,
          'error': 'Content too short, likely blocked or empty',
          'url': url,
          'content_length': len(main_content)
        }

      # Show content preview
      if verbose:
        print(f"\n  --- Content Preview (first 500 chars) ---", file=sys.stderr, flush=True)
        print(f"  {main_content[:500]}...", file=sys.stderr, flush=True)
        print(f"  --- End Preview ---\n", file=sys.stderr, flush=True)

      # Success!
      return {
        'success': True,
        'content': main_content,
        'url': url,
        'title': page_title,
        'word_count': quality_info['metrics']['word_count'],
        'char_count': quality_info['metrics']['char_count'],
        'quality_score': quality_info['score'],
        'quality_level': quality_info['quality'],
        'extraction_method': extraction_method,
        'financial_keywords': quality_info['metrics']['financial_keywords'],
        'attempts': attempt + 1,
        'timestamp': datetime.now().isoformat()
      }

    except requests.exceptions.Timeout:
      last_error = f'Request timeout (attempt {attempt + 1}/{max_retries})'
      if verbose:
        print(f"  ERROR: {last_error}", file=sys.stderr, flush=True)
      if attempt == max_retries - 1:
        return {'success': False, 'error': 'Request timeout after retries', 'url': url}

    except requests.exceptions.TooManyRedirects:
      if verbose:
        print(f"  ERROR: Too many redirects", file=sys.stderr, flush=True)
      return {'success': False, 'error': 'Too many redirects', 'url': url}

    except requests.exceptions.SSLError as e:
      last_error = f'SSL Error: {str(e)}'
      if verbose:
        print(f"  ERROR: {last_error}", file=sys.stderr, flush=True)
      if attempt == max_retries - 1:
        return {'success': False, 'error': f'SSL Error: {str(e)}', 'url': url}

    except requests.exceptions.ConnectionError as e:
      last_error = f'Connection error: {str(e)}'
      if verbose:
        print(f"  ERROR: {last_error}", file=sys.stderr, flush=True)
      if attempt == max_retries - 1:
        return {'success': False, 'error': f'Connection error: {str(e)}', 'url': url}

    except requests.exceptions.HTTPError as e:
      status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
      last_error = f'HTTP {status_code}: {str(e)}'
      if verbose:
        print(f"  ERROR: {last_error}", file=sys.stderr, flush=True)

      # Don't retry on certain error codes
      if status_code in [404, 410, 451]:  # Not Found, Gone, Unavailable for Legal Reasons
        return {'success': False, 'error': f'HTTP {status_code}: Page not available', 'url': url}

      if attempt == max_retries - 1:
        return {'success': False, 'error': f'HTTP error after retries: {str(e)}', 'url': url}

    except requests.exceptions.RequestException as e:
      last_error = f'Request failed: {str(e)}'
      if verbose:
        print(f"  ERROR: {last_error}", file=sys.stderr, flush=True)
      if attempt == max_retries - 1:
        return {'success': False, 'error': f'Request failed: {str(e)}', 'url': url}

    except Exception as e:
      last_error = f'Parsing failed: {str(e)}'
      if verbose:
        print(f"  ERROR: {last_error}", file=sys.stderr, flush=True)
      if attempt == max_retries - 1:
        return {'success': False, 'error': f'Parsing failed: {str(e)}', 'url': url}

  # Should not reach here, but just in case
  return {'success': False, 'error': f'Failed after {max_retries} attempts: {last_error}', 'url': url}

def extract_financial_content_advanced(soup, url: str, verbose: bool = False) -> Tuple[str, str]:
  """
  Advanced content extraction that tries multiple methods and picks the best result
  Returns: (content, extraction_method)
  """

  extraction_results = []

  # Method 1: Site-specific selectors
  site_specific_content = extract_with_site_selectors(soup, url)
  if site_specific_content:
    score = score_content(site_specific_content)
    extraction_results.append(('site_specific_selectors', site_specific_content, score))
    if verbose:
      print(f"    Site-specific selectors: {len(site_specific_content)} chars (score: {score})", file=sys.stderr, flush=True)

  # Method 2: Semantic HTML5 elements
  semantic_content = extract_with_semantic_html(soup)
  if semantic_content:
    score = score_content(semantic_content)
    extraction_results.append(('semantic_html5', semantic_content, score))
    if verbose:
      print(f"    Semantic HTML5: {len(semantic_content)} chars (score: {score})", file=sys.stderr, flush=True)

  # Method 3: Common content classes
  class_based_content = extract_with_common_classes(soup)
  if class_based_content:
    score = score_content(class_based_content)
    extraction_results.append(('common_classes', class_based_content, score))
    if verbose:
      print(f"    Common classes: {len(class_based_content)} chars (score: {score})", file=sys.stderr, flush=True)

  # Method 4: Longest text block heuristic
  longest_block_content = extract_longest_text_blocks(soup)
  if longest_block_content:
    score = score_content(longest_block_content)
    extraction_results.append(('longest_blocks', longest_block_content, score))
    if verbose:
      print(f"    Longest blocks: {len(longest_block_content)} chars (score: {score})", file=sys.stderr, flush=True)

  # Method 5: Paragraph density analysis
  density_content = extract_by_paragraph_density(soup)
  if density_content:
    score = score_content(density_content)
    extraction_results.append(('paragraph_density', density_content, score))
    if verbose:
      print(f"    Paragraph density: {len(density_content)} chars (score: {score})", file=sys.stderr, flush=True)

  # Pick the best result
  if extraction_results:
    extraction_results.sort(key=lambda x: x[2], reverse=True)
    best_method, best_content, best_score = extraction_results[0]

    if verbose:
      print(f"    BEST: {best_method} (score: {best_score})", file=sys.stderr, flush=True)

    return best_content, best_method

  # Absolute fallback
  if verbose:
    print(f"    WARNING: All methods failed, using raw text fallback", file=sys.stderr, flush=True)
  return extract_raw_text_fallback(soup), 'raw_fallback'

def score_content(content: str) -> int:
  """Score content quality for extraction method comparison"""
  if not content:
    return 0

  words = content.split()
  word_count = len(words)

  score = 0

  # Length scoring
  if word_count >= 200:
    score += 50
  elif word_count >= 100:
    score += 30
  elif word_count >= 50:
    score += 15

  # Financial keyword density
  financial_keywords = ['revenue', 'earnings', 'profit', 'quarter', 'fiscal', 'shares', 'eps']
  keyword_count = sum(1 for kw in financial_keywords if kw in content.lower())
  score += min(keyword_count * 5, 25)

  # Sentence structure (presence of periods indicates real content)
  sentence_markers = content.count('.') + content.count('!') + content.count('?')
  score += min(sentence_markers * 2, 25)

  return score

def extract_with_site_selectors(soup, url: str) -> str:
  """Extract using site-specific selectors"""

  # SEC filings
  if 'sec.gov' in url:
    selectors = ['document', 'text', '#formDiv', '.FormGrouping', 'pre', '[id*="document"]']

  # Company investor relations
  elif 'investor.' in url:
    selectors = ['article', '.content', '.press-release', '.earnings', '.financial-data', 'main']

  # Reuters
  elif 'reuters.com' in url:
    selectors = ['[data-testid="ArticleBody"]', 'article', '.article-body', '.ArticleBody']

  # AP News
  elif 'apnews.com' in url:
    selectors = ['.Article', 'article', '.RichTextStoryBody', '[class*="Body"]']

  # Yahoo Finance
  elif 'finance.yahoo.com' in url:
    selectors = ['.caas-body', 'article', '.body', '[class*="article"]']

  # MarketWatch
  elif 'marketwatch.com' in url:
    selectors = ['[class*="article__body"]', 'article', '.article__body']

  # CNBC
  elif 'cnbc.com' in url:
    selectors = ['.ArticleBody-articleBody', 'article', '.group']

  else:
    return ""

  for selector in selectors:
    try:
      elements = soup.select(selector)
      if elements:
        content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
        content = re.sub(r'\s+', ' ', content)
        if len(content) > 200:
          return content
    except:
      continue

  return ""

def extract_with_semantic_html(soup) -> str:
  """Extract using HTML5 semantic elements"""
  semantic_elements = ['article', 'main', '[role="main"]', 'section']

  for elem_selector in semantic_elements:
    try:
      elements = soup.select(elem_selector)
      if elements:
        content_parts = []
        for elem in elements:
          text = elem.get_text(separator=' ', strip=True)
          if text and len(text) > 50:
            content_parts.append(text)

        if content_parts:
          content = ' '.join(content_parts)
          content = re.sub(r'\s+', ' ', content)
          if len(content) > 200:
            return content
    except:
      continue

  return ""

def extract_with_common_classes(soup) -> str:
  """Extract using common content class patterns"""
  class_patterns = [
    'content', 'article', 'post', 'entry', 'story', 'text',
    'body', 'main', 'primary', 'detail', 'news'
  ]

  for pattern in class_patterns:
    try:
      # Look for classes containing the pattern
      elements = soup.find_all(class_=re.compile(pattern, re.I))
      if elements:
        content_parts = []
        for elem in elements:
          text = elem.get_text(separator=' ', strip=True)
          if text and len(text) > 50:
            content_parts.append(text)

        if content_parts:
          content = ' '.join(content_parts)
          content = re.sub(r'\s+', ' ', content)
          if len(content) > 200:
            return content
    except:
      continue

  return ""

def extract_longest_text_blocks(soup) -> str:
  """Find and combine the longest text blocks"""
  try:
    # Find all divs and paragraphs
    elements = soup.find_all(['div', 'p', 'section', 'article'])

    # Score each element by text length
    scored_elements = []
    for elem in elements:
      text = elem.get_text(separator=' ', strip=True)
      if text and len(text) > 100:
        scored_elements.append((elem, len(text)))

    # Sort by length
    scored_elements.sort(key=lambda x: x[1], reverse=True)

    # Take top elements that together give us good content
    content_parts = []
    total_length = 0
    for elem, length in scored_elements[:10]:  # Top 10 longest
      text = elem.get_text(separator=' ', strip=True)
      content_parts.append(text)
      total_length += length
      if total_length > 2000:  # Enough content
        break

    if content_parts:
      content = ' '.join(content_parts)
      content = re.sub(r'\s+', ' ', content)
      return content
  except:
    pass

  return ""

def extract_by_paragraph_density(soup) -> str:
  """Extract based on paragraph density (areas with many paragraphs likely contain main content)"""
  try:
    # Find containers with multiple paragraphs
    containers = soup.find_all(['div', 'section', 'article'])

    scored_containers = []
    for container in containers:
      paragraphs = container.find_all('p', recursive=True)
      if len(paragraphs) >= 3:  # At least 3 paragraphs
        total_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        score = len(paragraphs) * len(total_text)
        scored_containers.append((container, score))

    if scored_containers:
      # Get best container
      scored_containers.sort(key=lambda x: x[1], reverse=True)
      best_container = scored_containers[0][0]

      content = best_container.get_text(separator=' ', strip=True)
      content = re.sub(r'\s+', ' ', content)
      return content
  except:
    pass

  return ""

def extract_raw_text_fallback(soup) -> str:
  """Last resort: extract all text with basic filtering"""
  try:
    all_text = soup.get_text(separator='\n')
    lines = [line.strip() for line in all_text.splitlines() if line.strip()]

    # Filter out very short lines
    content_lines = [line for line in lines if len(line) > 20]

    # Remove common navigation phrases
    filtered_lines = []
    skip_phrases = ['click here', 'sign up', 'subscribe', 'newsletter', 'cookie',
                   'privacy', 'terms', 'rights reserved', '©', 'copyright']

    for line in content_lines:
      line_lower = line.lower()
      if not any(phrase in line_lower for phrase in skip_phrases):
        filtered_lines.append(line)

    content = ' '.join(filtered_lines)
    content = re.sub(r'\s+', ' ', content)
    return content
  except:
    return ""

def batch_scrape(urls: List[str], delay: float = 0.5, max_retries: int = 3, verbose: bool = True) -> List[Dict[str, Any]]:
  """
  Scrape multiple URLs with delay between requests

  Args:
    urls: List of URLs to scrape
    delay: Delay between requests in seconds (be respectful!)
    max_retries: Maximum retries per URL
    verbose: Whether to print detailed information during scraping
  """
  results = []

  print(f"\nStarting batch scrape of {len(urls)} URLs...", file=sys.stderr, flush=True)
  print(f"Settings: delay={delay}s, max_retries={max_retries}, verbose={verbose}", file=sys.stderr, flush=True)
  print("=" * 80, file=sys.stderr, flush=True)

  for i, url in enumerate(urls):
    print(f"\n[{i+1}/{len(urls)}] {url}", file=sys.stderr, flush=True)

    # Add delay between requests (except for first)
    if i > 0:
      time.sleep(delay)

    result = web_scrape(url, max_retries=max_retries, verbose=verbose)
    results.append(result)

    # Print summary
    if result['success']:
      print(f"SUCCESS - {result['word_count']} words, quality: {result['quality_level']} ({result['quality_score']}/100)", file=sys.stderr, flush=True)
    else:
      print(f"FAILED - {result['error']}", file=sys.stderr, flush=True)

  return results

def get_scraping_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
  """Generate comprehensive summary statistics from scraping results"""
  successful = [r for r in results if r.get('success', False)]
  failed = [r for r in results if not r.get('success', False)]

  total_words = sum(r.get('word_count', 0) for r in successful)
  total_chars = sum(r.get('char_count', 0) for r in successful)

  # Quality distribution
  quality_dist = Counter(r.get('quality_level', 'unknown') for r in successful)

  # Extraction method distribution
  method_dist = Counter(r.get('extraction_method', 'unknown') for r in successful)

  # Average quality score
  quality_scores = [r.get('quality_score', 0) for r in successful]
  avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

  # Financial keyword statistics
  total_financial_keywords = sum(r.get('financial_keywords', 0) for r in successful)

  # Error analysis
  error_types = {}
  for result in failed:
    error = result.get('error', 'Unknown error')
    # Extract error type
    error_type = error.split(':')[0] if ':' in error else error[:50]
    error_types[error_type] = error_types.get(error_type, 0) + 1

  # Retry statistics
  retry_counts = [r.get('attempts', 1) for r in results]
  avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0

  return {
    'total_urls': len(results),
    'successful': len(successful),
    'failed': len(failed),
    'success_rate': len(successful) / len(results) if results else 0,
    'total_words': total_words,
    'total_chars': total_chars,
    'avg_words_per_page': total_words / len(successful) if successful else 0,
    'avg_chars_per_page': total_chars / len(successful) if successful else 0,
    'avg_quality_score': avg_quality,
    'quality_distribution': dict(quality_dist),
    'extraction_methods': dict(method_dist),
    'total_financial_keywords': total_financial_keywords,
    'avg_retries': avg_retries,
    'error_breakdown': error_types
  }

if __name__ == "__main__":
  print("=" * 80, file=sys.stderr, flush=True)
  print("ADVANCED WEB SCRAPING TOOL - TEST SUITE", file=sys.stderr, flush=True)
  print("=" * 80, file=sys.stderr, flush=True)

  # Test search functionality
  print("\n" + "=" * 80, file=sys.stderr, flush=True)
  print("PHASE 1: SEARCH", file=sys.stderr, flush=True)
  print("=" * 80, file=sys.stderr, flush=True)

  query = "MSFT earnings guidance"
  print(f"\nSearching for: '{query}'", file=sys.stderr, flush=True)

  search_results = search_duckduckgo(query, 10)
  print(f"\nFound {len(search_results)} search results", file=sys.stderr, flush=True)

  if search_results:
    print("\nSearch Results Summary:", file=sys.stderr, flush=True)
    print("-" * 80, file=sys.stderr, flush=True)
    for i, result in enumerate(search_results, 1):
      print(f"{i}. {result['title']}", file=sys.stderr, flush=True)
      print(f"   URL: {result['link']}", file=sys.stderr, flush=True)
      print(f"   Query Type: {result['query_type']}", file=sys.stderr, flush=True)
      print(f"   Snippet: {result['snippet'][:100]}...", file=sys.stderr, flush=True)
      print(file=sys.stderr, flush=True)

  # Test scraping with verbose output
  print("\n" + "=" * 80, file=sys.stderr, flush=True)
  print("PHASE 2: CONTENT EXTRACTION", file=sys.stderr, flush=True)
  print("=" * 80, file=sys.stderr, flush=True)

  urls_to_test = [result['link'] for result in search_results[:5]]  # Test first 5
  scrape_results = batch_scrape(urls_to_test, delay=1.0, max_retries=3, verbose=True)

  # Generate and display comprehensive summary
  print("\n" + "=" * 80, file=sys.stderr, flush=True)
  print("PHASE 3: ANALYSIS & STATISTICS", file=sys.stderr, flush=True)
  print("=" * 80, file=sys.stderr, flush=True)

  summary = get_scraping_summary(scrape_results)

  print("\nOVERALL STATISTICS:", file=sys.stderr, flush=True)
  print("-" * 80, file=sys.stderr, flush=True)
  print(f"Total URLs Attempted:      {summary['total_urls']}", file=sys.stderr, flush=True)
  print(f"Successful Scrapes:        {summary['successful']}", file=sys.stderr, flush=True)
  print(f"Failed Scrapes:            {summary['failed']}", file=sys.stderr, flush=True)
  print(f"Success Rate:              {summary['success_rate']:.1%}", file=sys.stderr, flush=True)
  print(f"Average Retries per URL:   {summary['avg_retries']:.1f}", file=sys.stderr, flush=True)

  print("\nCONTENT STATISTICS:", file=sys.stderr, flush=True)
  print("-" * 80, file=sys.stderr, flush=True)
  print(f"Total Words Extracted:     {summary['total_words']:,}", file=sys.stderr, flush=True)
  print(f"Total Characters:          {summary['total_chars']:,}", file=sys.stderr, flush=True)
  print(f"Average Words per Page:    {summary['avg_words_per_page']:.0f}", file=sys.stderr, flush=True)
  print(f"Average Characters/Page:   {summary['avg_chars_per_page']:.0f}", file=sys.stderr, flush=True)

  print("\nQUALITY ANALYSIS:", file=sys.stderr, flush=True)
  print("-" * 80, file=sys.stderr, flush=True)
  print(f"Average Quality Score:     {summary['avg_quality_score']:.1f}/100", file=sys.stderr, flush=True)
  print(f"Financial Keywords Found:  {summary['total_financial_keywords']}", file=sys.stderr, flush=True)

  if summary['quality_distribution']:
    print("\nQuality Distribution:", file=sys.stderr, flush=True)
    for quality, count in sorted(summary['quality_distribution'].items()):
      print(f"  {quality.ljust(15)}: {count} pages", file=sys.stderr, flush=True)

  if summary['extraction_methods']:
    print("\nExtraction Methods Used:", file=sys.stderr, flush=True)
    for method, count in sorted(summary['extraction_methods'].items()):
      print(f"  {method.ljust(25)}: {count} pages", file=sys.stderr, flush=True)

  if summary['error_breakdown']:
    print("\nError Breakdown:", file=sys.stderr, flush=True)
    for error_type, count in summary['error_breakdown'].items():
      print(f"  {error_type[:50]}: {count}", file=sys.stderr, flush=True)

  # Display detailed content from successful scrapes
  successful_scrapes = [r for r in scrape_results if r.get('success', False)]
  if successful_scrapes:
    print("\n" + "=" * 80, file=sys.stderr, flush=True)
    print("PHASE 4: EXTRACTED CONTENT SAMPLES", file=sys.stderr, flush=True)
    print("=" * 80, file=sys.stderr, flush=True)

    for i, sample in enumerate(successful_scrapes[:3], 1):  # Show first 3
      print(f"\n--- SAMPLE {i} ---", file=sys.stderr, flush=True)
      print(f"Title:             {sample['title']}", file=sys.stderr, flush=True)
      print(f"URL:               {sample['url']}", file=sys.stderr, flush=True)
      print(f"Word Count:        {sample['word_count']:,}", file=sys.stderr, flush=True)
      print(f"Quality Score:     {sample['quality_score']}/100 ({sample['quality_level']})", file=sys.stderr, flush=True)
      print(f"Extraction Method: {sample['extraction_method']}", file=sys.stderr, flush=True)
      print(f"Financial Keywords: {sample['financial_keywords']}", file=sys.stderr, flush=True)
      print(f"Attempts:          {sample['attempts']}", file=sys.stderr, flush=True)
      print(f"\nContent Preview (first 800 characters):", file=sys.stderr, flush=True)
      print("-" * 80, file=sys.stderr, flush=True)
      print(sample['content'][:800], file=sys.stderr, flush=True)
      if len(sample['content']) > 800:
        print(f"... [{len(sample['content']) - 800:,} more characters]", file=sys.stderr, flush=True)
      print("-" * 80, file=sys.stderr, flush=True)

  # Final summary
  print("\n" + "=" * 80, file=sys.stderr, flush=True)
  print("TEST COMPLETE", file=sys.stderr, flush=True)
  print("=" * 80, file=sys.stderr, flush=True)
  print(f"\nResults: {summary['successful']}/{summary['total_urls']} successful ({summary['success_rate']:.1%})", file=sys.stderr, flush=True)
  print(f"Total content extracted: {summary['total_words']:,} words", file=sys.stderr, flush=True)
  print(f"Average quality: {summary['avg_quality_score']:.1f}/100", file=sys.stderr, flush=True)

  # Close sessions
  _session_manager.close_all()
  print("\nAll sessions closed. Test suite finished successfully.", file=sys.stderr, flush=True)
  print("=" * 80, file=sys.stderr, flush=True)

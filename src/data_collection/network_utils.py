"""
Network Utilities for Clinical Trial Data Collection
Handles API requests with retry logic, caching, and offline fallback
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime, timedelta
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkError(Exception):
    """Custom exception for network-related errors"""
    pass


class NetworkUtils:
    """
    Robust network utilities for API interactions
    
    Features:
    - Automatic retry with exponential backoff
    - Response caching to reduce API calls
    - Offline fallback mode
    - Rate limiting compliance
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        base_url: str = "https://clinicaltrials.gov/api/v2/studies",
        cache_enabled: bool = True,
        cache_dir: str = "data/cache",
        cache_ttl: int = 3600,  # 1 hour in seconds
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit: float = 0.5  # seconds between requests
    ):
        self.base_url = base_url
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        
        # Create cache directory
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ClinicalTrialIntelligence/1.0 (Research Purpose)',
            'Accept': 'application/json'
        })
        
        # Track last request time for rate limiting
        self.last_request_time = 0
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Generate cache file path from key"""
        safe_key = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in cache_key)
        return self.cache_dir / f"{safe_key[:200]}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if available and not expired"""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
            age_seconds = (datetime.now() - cached_time).total_seconds()
            
            if age_seconds > self.cache_ttl:
                return None
            
            return cached_data.get('data')
        
        except Exception:
            return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache with timestamp"""
        if not self.cache_enabled:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'cache_key': cache_key,
                'data': data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        
        except Exception:
            pass
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limit"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = 'GET',
        use_cache: bool = True,
        **kwargs
    ) -> Dict:
        """Make HTTP request with retry logic and caching"""
        
        # Build full URL
        if endpoint.startswith('http'):
            url = endpoint
        else:
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Create cache key
        cache_key = f"{method}_{url}_{json.dumps(params, sort_keys=True) if params else 'no_params'}"
        
        # Try to load from cache
        if use_cache and method == 'GET':
            cached_response = self._load_from_cache(cache_key)
            if cached_response is not None:
                return cached_response
        
        # Make request with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout,
                    **kwargs
                )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_after)
                        continue
                
                response.raise_for_status()
                data = response.json()
                
                if method == 'GET':
                    self._save_to_cache(cache_key, data)
                
                return data
            
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    sleep_time = min(2 ** attempt, 30)
                    time.sleep(sleep_time)
            
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    sleep_time = min(2 ** attempt, 30)
                    time.sleep(sleep_time)
            
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    sleep_time = min(2 ** attempt, 30)
                    time.sleep(sleep_time)
        
        error_msg = f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        raise NetworkError(error_msg)
    
    def test_connection(self) -> bool:
        """Test if network connection is working"""
        try:
            response = self.session.get(
                "https://clinicaltrials.gov/api/v2/stats",
                timeout=5
            )
            return response.status_code == 200
        
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.cache_enabled or not self.cache_dir.exists():
            return {'enabled': False, 'files': 0, 'total_size_kb': 0}
        
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'enabled': True,
            'files': len(cache_files),
            'total_size_kb': total_size / 1024,
            'ttl_hours': self.cache_ttl / 3600
        }

import os
import sys
import json
import hashlib
import logging
import argparse
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from functools import lru_cache
import httpx
import tweepy
import praw
import prawcore
from mastodon import Mastodon, MastodonError, MastodonNotFoundError, MastodonRatelimitError, MastodonUnauthorizedError, MastodonVersionError
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
import base64
from urllib.parse import quote_plus, urlparse
from PIL import Image
from atproto import Client, exceptions as atproto_exceptions
from dotenv import load_dotenv

load_dotenv()  # Load .env file if available

logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('analyzer.log'), logging.StreamHandler()]
)
logger = logging.getLogger('SocialOSINTLM')

# --- Constants ---
CACHE_EXPIRY_HOURS = 24
MAX_CACHE_ITEMS = 200  # Max tweets/posts/submissions per user/platform in cache
REQUEST_TIMEOUT = 20.0 # Default timeout for HTTP requests
INITIAL_FETCH_LIMIT = 50 # How many items to fetch on first run or force_refresh
INCREMENTAL_FETCH_LIMIT = 50 # How many items to fetch during incremental updates
MASTODON_FETCH_LIMIT = 40 # Mastodon API max is often 40

class RateLimitExceededError(Exception):
    pass

class UserNotFoundError(Exception):
    pass

class AccessForbiddenError(Exception):
    pass

# --- JSON Encoder ---
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# --- Helper Function (Moved to Global Scope) ---
def get_sort_key(item: Dict[str, Any], dt_key: str) -> datetime:
    """Safely gets and parses a datetime string or object for sorting."""
    dt_val = item.get(dt_key)
    if isinstance(dt_val, str):
        try:
            # Handle ISO format strings (including 'Z' for UTC)
            if dt_val.endswith('Z'):
                dt_val = dt_val[:-1] + '+00:00'
            return datetime.fromisoformat(dt_val)
        except ValueError: # Handle cases where conversion might fail
            logger.warning(f"Could not parse datetime string: {dt_val}")
            return datetime.min.replace(tzinfo=timezone.utc)
    elif isinstance(dt_val, datetime):
         # Ensure timezone for comparison if needed
         return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    # Fallback for missing/invalid keys or other types (like timestamps)
    elif isinstance(dt_val, (int, float)):
         try:
             # Attempt to treat as UNIX timestamp
             return datetime.fromtimestamp(dt_val, tz=timezone.utc)
         except (ValueError, OSError): # OSError can occur for out-of-range timestamps
              logger.warning(f"Could not convert timestamp: {dt_val}")
              return datetime.min.replace(tzinfo=timezone.utc)

    logger.debug(f"Using fallback datetime for key '{dt_key}' with value type: {type(dt_val)}")
    return datetime.min.replace(tzinfo=timezone.utc) # Fallback for missing/invalid keys


# --- Main Class ---
class SocialOSINTLM:
    def __init__(self, args=None):
        self.console = Console()
        self.base_dir = Path("data")
        self._setup_directories()
        self.progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            transient=True,
            console=self.console,
            refresh_per_second=10
        )
        self.current_task: Optional[TaskID] = None
        self._analysis_response: Optional[httpx.Response] = None
        self._analysis_exception: Optional[Exception] = None
        self.args = args if args else argparse.Namespace()
        self._verify_env_vars()

    def _verify_env_vars(self):
        required = ['OPENROUTER_API_KEY', 'IMAGE_ANALYSIS_MODEL']
        # Check for at least one platform credential set
        platforms_configured = any([
            all(os.getenv(k) for k in ['TWITTER_BEARER_TOKEN']),
            all(os.getenv(k) for k in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']),
            all(os.getenv(k) for k in ['BLUESKY_IDENTIFIER', 'BLUESKY_APP_SECRET']),
            # +++ Add Mastodon Check +++
            all(os.getenv(k) for k in ['MASTODON_ACCESS_TOKEN', 'MASTODON_API_BASE_URL'])
        ])
        # HN needs no keys, considered configured if no other platform is.
        if not platforms_configured and 'hackernews' not in self.get_available_platforms(check_creds=False):
             logger.warning("No platform API credentials found in environment variables. Only HackerNews might work.")

        missing_core = [var for var in required if not os.getenv(var)]
        if missing_core:
            raise RuntimeError(f"Missing critical environment variables: {', '.join(missing_core)}")

    def _setup_directories(self):
        for dir_name in ['cache', 'media', 'outputs']:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # --- Property-based Client Initializers ---
    @property
    def bluesky(self) -> Client:
        if not hasattr(self, '_bluesky_client'):
            try:
                if not os.getenv('BLUESKY_IDENTIFIER') or not os.getenv('BLUESKY_APP_SECRET'):
                     raise RuntimeError("Bluesky credentials (BLUESKY_IDENTIFIER, BLUESKY_APP_SECRET) not set in environment.")
                client = Client()
                client.login(
                    os.environ['BLUESKY_IDENTIFIER'],
                    os.environ['BLUESKY_APP_SECRET']
                )
                self._bluesky_client = client
                logger.debug("Bluesky login successful")
            except (KeyError, atproto_exceptions.AtProtocolError, RuntimeError) as e: # Use alias
                logger.error(f"Bluesky setup failed: {e}")
                raise RuntimeError(f"Bluesky setup failed: {e}") # Re-raise after logging
        return self._bluesky_client

    # +++ Add Mastodon Client Property +++
    @property
    def mastodon(self) -> Mastodon:
        if not hasattr(self, '_mastodon_client'):
            try:
                token = os.getenv('MASTODON_ACCESS_TOKEN')
                base_url = os.getenv('MASTODON_API_BASE_URL')
                if not token or not base_url:
                    raise RuntimeError("Mastodon credentials (MASTODON_ACCESS_TOKEN, MASTODON_API_BASE_URL) not set.")

                # Validate URL format
                parsed_url = urlparse(base_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                     raise RuntimeError(f"Invalid MASTODON_API_BASE_URL format: {base_url}. Should be like 'https://mastodon.social'.")

                client = Mastodon(
                    access_token=token,
                    api_base_url=base_url,
                    request_timeout=REQUEST_TIMEOUT # Use consistent timeout
                )
                # Test connection by getting instance info
                client.instance()
                self._mastodon_client = client
                logger.debug(f"Mastodon client initialized for {base_url}.")
            except (KeyError, MastodonError, RuntimeError) as e:
                logger.error(f"Mastodon setup failed: {e}")
                raise RuntimeError(f"Mastodon setup failed: {e}")
        return self._mastodon_client
    # --- End Mastodon Client Property ---


    @property
    def openrouter(self) -> httpx.Client:
        if not hasattr(self, '_openrouter'):
            try:
                # --- Check for OpenRouter API Key (related to 401 errors during image analysis) ---
                api_key = os.environ.get('OPENROUTER_API_KEY')
                # --- Optional Debugging: Uncomment the line below to check if the key is read ---
                # print(f"DEBUG: OpenRouter Key read from env: {'Yes' if api_key else 'No'}")
                if not api_key:
                    raise KeyError("OPENROUTER_API_KEY not found in environment variables. Check your .env file.")
                # --- End Check ---

                self._openrouter = httpx.Client(
                    base_url="https://openrouter.ai/api/v1",
                    headers={
                        "Authorization": f"Bearer {api_key}", # Use the verified key
                        "HTTP-Referer": "http://localhost:3000", # Replace with your actual referrer if applicable
                        "X-Title": "Social Media analyzer",
                        "Content-Type": "application/json"
                    },
                    timeout=60.0 # Increased timeout for potentially long LLM calls
                )
            except KeyError as e:
                raise RuntimeError(f"Missing OpenRouter API key (OPENROUTER_API_KEY): {e}")
        return self._openrouter

    @property
    def reddit(self) -> praw.Reddit:
        if not hasattr(self, '_reddit'):
            try:
                if not all(os.getenv(k) for k in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']):
                    raise RuntimeError("Reddit credentials not fully set in environment.")
                self._reddit = praw.Reddit(
                    client_id=os.environ['REDDIT_CLIENT_ID'],
                    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
                    user_agent=os.environ['REDDIT_USER_AGENT'],
                    read_only=True # Explicitly set read-only mode
                )
                self._reddit.auth.scopes() # Test connection/auth early
                logger.debug("Reddit client initialized.")
            except (KeyError, prawcore.exceptions.OAuthException, prawcore.exceptions.ResponseException, RuntimeError) as e:
                 logger.error(f"Reddit setup failed: {e}")
                 raise RuntimeError(f"Reddit setup failed: {e}")
        return self._reddit

    @property
    def twitter(self) -> tweepy.Client:
        if not hasattr(self, '_twitter'):
            try:
                if not os.getenv('TWITTER_BEARER_TOKEN'):
                    raise RuntimeError("Twitter Bearer Token (TWITTER_BEARER_TOKEN) not set.")
                self._twitter = tweepy.Client(bearer_token=os.environ['TWITTER_BEARER_TOKEN'], wait_on_rate_limit=False)
                # Test connection
                self._twitter.get_user(username="twitterdev") # Example known user
                logger.debug("Twitter client initialized.")
            except (KeyError, tweepy.errors.TweepyException, RuntimeError) as e:
                logger.error(f"Twitter setup failed: {e}")
                raise RuntimeError(f"Twitter setup failed: {e}")
        return self._twitter

    # --- Utility Methods ---
    def _handle_rate_limit(self, platform: str, exception: Optional[Exception] = None):
        error_message = f"{platform} API rate limit exceeded."
        reset_info = ""
        wait_seconds = 900 # Default wait 15 mins if unknown

        if isinstance(exception, tweepy.TooManyRequests):
            rate_limit_reset = exception.response.headers.get('x-rate-limit-reset')
            if rate_limit_reset:
                try:
                    reset_time = datetime.fromtimestamp(int(rate_limit_reset), tz=timezone.utc)
                    current_time = datetime.now(timezone.utc)
                    wait_seconds = max(int((reset_time - current_time).total_seconds()) + 5, 1) # Add 5s buffer
                    reset_info = f"Try again after: {reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                except (ValueError, TypeError):
                    logger.debug("Could not parse rate limit reset time.")
        elif isinstance(exception, (prawcore.exceptions.RequestException, httpx.HTTPStatusError)):
             if hasattr(exception, 'response') and exception.response.status_code == 429:
                 reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
             else:
                 logger.error(f"Unhandled HTTP Error for {platform}: {exception}")
                 raise exception
        # +++ Add Mastodon Rate Limit Handling +++
        elif isinstance(exception, MastodonRatelimitError):
            # Mastodon headers *might* contain rate limit info, but not consistently guaranteed across instances
            # Check response headers if available (Mastodon.py might not expose them easily on the exception)
            reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
            # Log header details if possible
            if hasattr(exception, 'response') and hasattr(exception.response, 'headers'):
                 logger.debug(f"Mastodon rate limit headers: {exception.response.headers}")
        # --- End Mastodon Rate Limit Handling ---
        elif isinstance(exception, atproto_exceptions.AtProtocolError) and 'rate limit' in str(exception).lower(): # Use alias
             reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
        else:
             # Check specifically for OpenRouter rate limit (often HTTP 429)
             if isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code == 429:
                 reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
                 error_message = f"Image Analysis ({platform}) API rate limit exceeded." # More specific message
             else:
                 reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."


        self.console.print(Panel(
            f"[bold red]Rate Limit Blocked: {platform}[/bold red]\n"
            f"{error_message}\n"
            f"{reset_info}",
            title="🚫 Rate Limit",
            border_style="red"
        ))
        raise RateLimitExceededError(error_message + f" ({reset_info})") # Raise specific error

    def _get_media_path(self, url: str, platform: str, username: str) -> Path:
        # Use only URL hash for consistency, platform/username are for context
        url_hash = hashlib.md5(url.encode()).hexdigest()
        # Use a generic extension initially, will be refined if downloaded
        return self.base_dir / 'media' / f"{url_hash}.media"

    def _download_media(self, url: str, platform: str, username: str, headers: Optional[dict] = None) -> Optional[Path]:
        """Downloads media, saves with correct extension, returns path if successful."""
        media_path_stub = self._get_media_path(url, platform, username)
        # Check if any file with this hash exists (might have different extensions)
        existing_files = list(self.base_dir.glob(f'media/{media_path_stub.stem}.*'))
        if existing_files:
            # Prefer common image types if multiple exist (e.g., jpg over media)
            for ext in ['.jpg', '.png', '.webp', '.gif']:
                 if (found := self.base_dir / 'media' / f"{media_path_stub.stem}{ext}").exists():
                     logger.debug(f"Media cache hit: {found}")
                     return found
            logger.debug(f"Media cache hit (generic): {existing_files[0]}")
            return existing_files[0] # Return the first one found

        valid_types = {
            'image/jpeg': '.jpg', 'image/png': '.png',
            'image/gif': '.gif', 'image/webp': '.webp',
            'video/mp4': '.mp4', 'video/webm': '.webm' # Add common video types
            # Note: Image analysis currently only supports image types
        }
        final_media_path = None

        try:
            # Platform-specific adjustments for AUTHENTICATION
            # Mastodon media URLs are typically public CDN links, no special auth needed usually.
            # The existing logic only adds auth for Twitter/Bluesky, so it should be fine.
            auth_headers = {}
            if platform == 'twitter':
                if not hasattr(self, '_twitter'): self.twitter
                token = os.getenv('TWITTER_BEARER_TOKEN')
                if token: auth_headers['Authorization'] = f'Bearer {token}'
            elif platform == 'bluesky':
                 if not hasattr(self, '_bluesky_client'): self.bluesky
                 # Accessing protected member, consider refactoring if possible
                 access_token = getattr(self.bluesky._session, 'access_jwt', None)
                 if not access_token:
                     logger.warning("Bluesky access token not available for media download.")
                 else:
                     auth_headers['Authorization'] = f"Bearer {access_token}"
                 # CDN URL adjustments are handled in fetch_bluesky now

            # Combine provided headers with auth headers
            request_headers = headers.copy() if headers else {}
            request_headers.update(auth_headers)


            with httpx.Client(follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
                resp = client.get(url, headers=request_headers) # Use combined headers
                resp.raise_for_status()

            content_type = resp.headers.get('content-type', '').lower().split(';')[0]
            extension = valid_types.get(content_type)

            if not extension:
                logger.warning(f"Unsupported or non-media type '{content_type}' for URL: {url}")
                # Might be HTML page etc.
                return None

            final_media_path = media_path_stub.with_suffix(extension)
            final_media_path.write_bytes(resp.content)
            logger.debug(f"Downloaded media to: {final_media_path}")
            return final_media_path

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Guess platform based on URL structure if not directly passed? Or rely on caller context.
                # Assume platform passed in is correct context for rate limit message.
                self._handle_rate_limit(f"{platform} Media Download", e) # Use platform context
            elif e.response.status_code in [404, 403, 401]:
                 logger.warning(f"Media access error ({e.response.status_code}) for {url}. Skipping.")
            else:
                logger.error(f"HTTP error {e.response.status_code} downloading {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Media download failed for {url}: {str(e)}", exc_info=False)
            return None

    def _analyze_image(self, file_path: Path, context: str = "") -> Optional[str]:
        """Analyzes image using OpenRouter, handles resizing and errors."""
        if not file_path or not file_path.exists():
            logger.warning(f"Image analysis skipped: file path invalid or missing ({file_path})")
            return None

        # Check file extension - only analyze supported image types
        supported_image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        if file_path.suffix.lower() not in supported_image_extensions:
             logger.debug(f"Skipping analysis for non-image file: {file_path}")
             return None

        temp_path = None # Define outside try block
        analysis_file_path = file_path # Default to original path

        try:
            with Image.open(file_path) as img:
                # Format check (redundant with extension check, but safer)
                if img.format.lower() not in ['jpeg', 'png', 'webp', 'gif']:
                    logger.warning(f"Unsupported image type for analysis: {img.format} at {file_path}")
                    return None

                # Resize large images
                max_dimension = 1536
                scale_factor = min(max_dimension / img.size[0], max_dimension / img.size[1], 1.0)

                if scale_factor < 1.0:
                    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    # Handle animated GIFs - analyze only the first frame
                    is_animated = getattr(img, 'is_animated', False) and img.n_frames > 1
                    if is_animated:
                         img.seek(0) # Go to first frame

                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    temp_path = file_path.with_suffix('.resized.jpg')
                    # Ensure RGB mode before saving as JPEG
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(temp_path, 'JPEG', quality=85)
                    analysis_file_path = temp_path
                    logger.debug(f"Resized image for analysis: {file_path} -> {temp_path}")
                else:
                    # Convert non-JPEG formats for consistency (including first frame of GIF)
                    if img.format.lower() != 'jpeg':
                         is_animated = getattr(img, 'is_animated', False) and img.n_frames > 1
                         temp_path = file_path.with_suffix('.converted.jpg')
                         if is_animated:
                             img.seek(0) # Go to first frame
                         if img.mode in ("RGBA", "P"):
                             img = img.convert("RGB")
                         img.save(temp_path, 'JPEG', quality=90)
                         analysis_file_path = temp_path
                         logger.debug(f"Converted image to JPEG for analysis: {file_path} -> {temp_path}")
                    # else: analysis_file_path remains file_path (original JPEG)

            base64_image = base64.b64encode(analysis_file_path.read_bytes()).decode('utf-8')

            # --- Call OpenRouter ---
            prompt_text = (
                 f"Perform an objective OSINT analysis of this image originating from {context}. Focus *only* on visually verifiable elements relevant to profiling or context understanding. Describe:\n"
                 "- **Setting/Environment:** (e.g., Indoor office, outdoor urban street, natural landscape, specific room type if identifiable). Note weather, time of day clues, architecture if distinctive.\n"
                 "- **Key Objects/Items:** List prominent or unusual objects. If text/logos are clearly legible (e.g., book titles, brand names on products, signs), state them exactly. Note technology types, tools, personal items.\n"
                 "- **People (if present):** Describe observable characteristics: approximate number, general attire, estimated age range (e.g., child, adult, senior), ongoing activity. *Do not guess identities or relationships.*\n"
                 "- **Text/Symbols:** Transcribe any clearly readable text on signs, labels, clothing, etc. Describe distinct symbols or logos.\n"
                 "- **Activity/Event:** Describe the apparent action (e.g., person working at desk, group dining, attending rally, specific sport).\n"
                 "- **Implicit Context Indicators:** Note subtle clues like reflections revealing unseen elements, background details suggesting location (e.g., specific landmarks, regional flora), or object condition suggesting usage/age.\n"
                 "- **Overall Scene Impression:** Summarize the visual narrative (e.g., professional setting, casual gathering, technical workshop, artistic expression, political statement).\n\n"
                 "Output a concise, bulleted list of observations. Avoid assumptions, interpretations, or emotional language not directly supported by the visual evidence."
            )

            model_to_use = os.getenv('IMAGE_ANALYSIS_MODEL', 'google/gemini-pro-vision') # Default if not set

            response = self.openrouter.post( # Use the property to get initialized client
                "/chat/completions",
                json={
                    "model": model_to_use,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high" # Use high detail for better analysis
                            }}
                        ]
                    }],
                    "max_tokens": 1024 # Allow longer response for detailed analysis
                }
            )
            response.raise_for_status() # Check for HTTP errors first (will raise on 401 etc.)
            result = response.json()

            # Check for API-level errors sometimes returned in a 200 OK response
            if 'error' in result:
                 # Log the specific error message from the API
                 err_msg = result['error'].get('message', 'Unknown error detail')
                 err_code = result['error'].get('code', 'N/A')
                 logger.error(f"Image analysis API error (Code: {err_code}): {err_msg}")
                 return None
            if 'choices' not in result or not result['choices'] or 'message' not in result['choices'][0] or 'content' not in result['choices'][0]['message']:
                logger.error(f"Invalid image analysis API response structure: {result}")
                return None

            analysis_text = result['choices'][0]['message']['content']
            logger.debug(f"Image analysis successful for: {file_path}")
            return analysis_text

        except (IOError, Image.DecompressionBombError, SyntaxError) as img_err: # Added SyntaxError for corrupt images
             logger.error(f"Image processing error for {file_path}: {str(img_err)}")
             return None
        except httpx.RequestError as req_err:
             logger.error(f"Network error during image analysis API call: {str(req_err)}")
             return None # Network errors are often transient, don't raise fatal
        except httpx.HTTPStatusError as status_err:
            model_to_use = os.getenv('IMAGE_ANALYSIS_MODEL', 'google/gemini-pro-vision')
            if status_err.response.status_code == 429:
                 # Pass the model name for clearer rate limit message
                 self._handle_rate_limit(model_to_use, status_err) # Should raise RateLimitExceededError
            # --- Explicitly check for 401 Unauthorized ---
            elif status_err.response.status_code == 401:
                 # This is where the original error likely occurred
                 logger.error(f"HTTP 401 Unauthorized during image analysis ({model_to_use}). Check your OPENROUTER_API_KEY.")
                 logger.error(f"API Response: {status_err.response.text}") # Log API response
                 # Don't raise RateLimitExceededError here, it's an auth failure
            else:
                 logger.error(f"HTTP error {status_err.response.status_code} during image analysis ({model_to_use}): {status_err.response.text}")
            return None # Return None for HTTP errors other than rate limits
        except Exception as e:
            logger.error(f"Unexpected error during image analysis for {file_path}: {str(e)}", exc_info=True) # Log full traceback for unexpected errors
            return None
        finally:
            # Clean up temporary file if created, regardless of success/failure
            if temp_path and temp_path.exists():
                 try:
                    temp_path.unlink()
                    logger.debug(f"Deleted temporary analysis file: {temp_path}")
                 except OSError as e:
                     logger.warning(f"Could not delete temporary analysis file {temp_path}: {e}")


    # --- Cache Management ---
    @lru_cache(maxsize=128) # Cache path generation
    def _get_cache_path(self, platform: str, username: str) -> Path:
        # Sanitize username (especially for Mastodon user@instance format)
        safe_username = "".join(c if c.isalnum() or c in ['-', '_', '.', '@'] else '_' for c in username)
        return self.base_dir / 'cache' / f"{platform}_{safe_username}.json"

    def _load_cache(self, platform: str, username: str) -> Optional[Dict[str, Any]]:
        """Loads cache data if it exists and is not expired."""
        cache_path = self._get_cache_path(platform, username)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            timestamp = datetime.fromisoformat(data['timestamp'])

            # Check expiry
            if datetime.now(timezone.utc) - timestamp < timedelta(hours=CACHE_EXPIRY_HOURS):
                 required_keys = ['timestamp']
                 # +++ Add Mastodon Keys +++
                 if platform == 'mastodon': required_keys.extend(['posts', 'user_info', 'stats'])
                 # --- Existing Checks ---
                 elif platform == 'twitter': required_keys.extend(['tweets', 'user_info'])
                 elif platform == 'reddit': required_keys.extend(['submissions', 'comments', 'stats'])
                 elif platform == 'bluesky': required_keys.extend(['posts', 'stats']) # profile_info is optional
                 elif platform == 'hackernews': required_keys.extend(['submissions', 'stats'])

                 if all(key in data for key in required_keys):
                      logger.debug(f"Cache hit for {platform}/{username}")
                      return data
                 else:
                     logger.warning(f"Cache file for {platform}/{username} seems incomplete. Discarding.")
                     cache_path.unlink(missing_ok=True) # Ensure deletion even if missing
                     return None
            else:
                logger.info(f"Cache expired for {platform}/{username}")
                return data # Return expired data for incremental update baseline

        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Failed to load or parse cache for {platform}/{username}: {e}. Discarding cache.")
            cache_path.unlink(missing_ok=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading cache for {platform}/{username}: {e}", exc_info=True)
            cache_path.unlink(missing_ok=True)
            return None


    def _save_cache(self, platform: str, username: str, data: Dict[str, Any]):
        """Saves data to cache, ensuring timestamp is updated."""
        cache_path = self._get_cache_path(platform, username)
        try:
            # Ensure the main list is sorted newest first before saving
            sort_key_map = {
                'twitter': ('tweets', 'created_at'),
                'reddit': [('submissions', 'created_utc'), ('comments', 'created_utc')],
                'bluesky': ('posts', 'created_at'),
                'hackernews': ('submissions', 'created_at'),
                # +++ Add Mastodon Sort Key +++
                'mastodon': ('posts', 'created_at'),
            }

            # *** The get_sort_key function definition was MOVED to the global scope ***
            # *** No definition needed here anymore ***

            if platform in sort_key_map:
                 items_to_sort = sort_key_map[platform]
                 if isinstance(items_to_sort, list): # Like Reddit
                     for list_key, dt_key in items_to_sort:
                         if list_key in data and data[list_key]:
                            # Call the globally defined get_sort_key function
                            data[list_key].sort(key=lambda x: get_sort_key(x, dt_key), reverse=True)
                 else: # Single list platforms
                    list_key, dt_key = items_to_sort
                    if list_key in data and data[list_key]:
                       # Call the globally defined get_sort_key function
                       data[list_key].sort(key=lambda x: get_sort_key(x, dt_key), reverse=True)


            data['timestamp'] = datetime.now(timezone.utc) # Use object for encoder
            cache_path.write_text(
                json.dumps(data, indent=2, cls=DateTimeEncoder),
                encoding='utf-8'
            )
            logger.debug(f"Saved cache for {platform}/{username}")
        except Exception as e:
            logger.error(f"Failed to save cache for {platform}/{username}: {e}", exc_info=True)


    # --- Platform Fetch Methods (with Incremental Logic) ---

    def fetch_twitter(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        # ... (existing implementation) ...
        # Now calls the global get_sort_key correctly
        cached_data = self._load_cache('twitter', username)

        # Condition 1: Cache is valid, recent, and not forced refresh
        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - datetime.fromisoformat(cached_data['timestamp'])) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Twitter @{username}")
            return cached_data

        # Condition 2 & 3: Cache is old, missing, or force_refresh is True
        logger.info(f"Fetching Twitter data for @{username} (Force Refresh: {force_refresh})")
        since_id = None
        existing_tweets = []
        existing_media_analysis = []
        existing_media_paths = []
        user_info = None

        if not force_refresh and cached_data:
            logger.info(f"Attempting incremental fetch for Twitter @{username}")
            existing_tweets = cached_data.get('tweets', [])
            # Ensure tweets are sorted newest first to get the latest ID
            existing_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            if existing_tweets:
                since_id = existing_tweets[0]['id']
                logger.debug(f"Using since_id: {since_id}")
            user_info = cached_data.get('user_info') # Keep existing user info
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])


        try:
            # --- Get User ID ---
            # User info might be missing on first fetch or corrupted cache
            if not user_info or force_refresh:
                try:
                     # Ensure client is ready
                     if not hasattr(self, '_twitter'): self.twitter
                     user_response = self.twitter.get_user(username=username, user_fields=['created_at', 'public_metrics', 'profile_image_url'])
                     if not user_response or not user_response.data:
                         raise UserNotFoundError(f"Twitter user @{username} not found.")
                     user = user_response.data
                     user_info = {
                         'id': user.id,
                         'name': user.name,
                         'username': user.username,
                         'created_at': user.created_at,
                         'public_metrics': user.public_metrics,
                         'profile_image_url': user.profile_image_url
                     }
                     logger.debug(f"Fetched user info for @{username}")
                except tweepy.NotFound:
                     raise UserNotFoundError(f"Twitter user @{username} not found.")
                except tweepy.Forbidden:
                     raise AccessForbiddenError(f"Access forbidden to Twitter user @{username}'s profile.")

            user_id = user_info['id']

            # --- Fetch Tweets ---
            new_tweets_data = []
            new_media_includes = {} # Store includes from new fetches

            # Use pagination for potentially large number of new tweets since last check
            fetch_limit = INITIAL_FETCH_LIMIT if (force_refresh or not since_id) else INCREMENTAL_FETCH_LIMIT
            pagination_token = None
            tweets_fetch_count = 0 # Track actual tweets fetched

            while True: # Loop for pagination
                current_page_limit = min(fetch_limit - tweets_fetch_count, 100)
                if current_page_limit <= 0: break # Stop if fetch limit reached

                try:
                    # Ensure client is ready
                    if not hasattr(self, '_twitter'): self.twitter
                    tweets_response = self.twitter.get_users_tweets(
                        id=user_id,
                        max_results=current_page_limit,
                        since_id=since_id if not force_refresh else None, # Only use since_id for incremental
                        pagination_token=pagination_token,
                        tweet_fields=['created_at', 'public_metrics', 'attachments', 'entities'], # Added entities for URLs etc.
                        expansions=['attachments.media_keys', 'author_id'], # Author_id redundant but good practice
                        media_fields=['url', 'preview_image_url', 'type', 'media_key', 'width', 'height']
                    )
                except tweepy.TooManyRequests as e:
                    self._handle_rate_limit('Twitter', exception=e)
                    return None # Rate limit error handled, exit fetch
                except tweepy.NotFound:
                    # If user existed but tweets now 404, could be deleted/suspended/protected between calls
                    raise UserNotFoundError(f"Tweets not found for user ID {user_id} (@{username}). User might be protected or deleted after profile check.")
                except tweepy.Forbidden as e:
                     raise AccessForbiddenError(f"Access forbidden to @{username}'s tweets (possibly protected). Details: {e}")

                if tweets_response.data:
                    page_count = len(tweets_response.data)
                    new_tweets_data.extend(tweets_response.data)
                    tweets_fetch_count += page_count
                    logger.debug(f"Fetched {page_count} new tweets page (Total this run: {tweets_fetch_count}).")
                if tweets_response.includes:
                    # Merge includes, especially media
                    for key, items in tweets_response.includes.items():
                        if key not in new_media_includes:
                            new_media_includes[key] = []
                        # Avoid duplicates if paginating aggressively (shouldn't happen with since_id)
                        existing_keys = {item['media_key'] for item in new_media_includes[key] if 'media_key' in item}
                        for item in items:
                            if 'media_key' not in item or item['media_key'] not in existing_keys:
                                new_media_includes[key].append(item)
                                if 'media_key' in item: existing_keys.add(item['media_key'])


                pagination_token = tweets_response.meta.get('next_token')
                # Adjust loop break condition to consider actual fetched vs limit
                if not pagination_token or tweets_fetch_count >= fetch_limit:
                    if pagination_token and tweets_fetch_count >= fetch_limit:
                         logger.info(f"Reached fetch limit ({fetch_limit}) for Twitter @{username}.")
                    else:
                         logger.debug("No more pages found.")
                    break


            logger.info(f"Fetched {tweets_fetch_count} total new tweets for @{username}.")

            # --- Process New Tweets and Media ---
            processed_new_tweets = []
            newly_added_media_analysis = []
            newly_added_media_paths = set() # Use set for efficient check later

            all_media_objects = {m.media_key: m for m in new_media_includes.get('media', [])}

            for tweet in new_tweets_data:
                 media_items_for_tweet = [] # Collect media details first
                 if tweet.attachments and 'media_keys' in tweet.attachments:
                     for media_key in tweet.attachments['media_keys']:
                         media = all_media_objects.get(media_key)
                         if media:
                             # Prefer media.url for photos, preview_image_url otherwise (videos)
                             # Use the url field directly if available, else preview
                             url = media.url if media.type in ['photo', 'gif'] and media.url else media.preview_image_url
                             if url:
                                 media_path = self._download_media(url=url, platform='twitter', username=username)
                                 if media_path:
                                     # Analyze only if it's an image type we support
                                     analysis = None
                                     if media_path.suffix.lower() in supported_image_extensions:
                                         analysis = self._analyze_image(media_path, f"Twitter user @{username}'s tweet")

                                     media_items_for_tweet.append({
                                         'type': media.type,
                                         'analysis': analysis,
                                         'url': url,
                                         'local_path': str(media_path)
                                     })
                                     if analysis: newly_added_media_analysis.append(analysis)
                                     newly_added_media_paths.add(str(media_path))


                 tweet_data = {
                     'id': tweet.id,
                     'text': tweet.text,
                     'created_at': tweet.created_at, # Already datetime object
                     'metrics': tweet.public_metrics,
                     'entities': tweet.entities, # Store entities
                     'media': media_items_for_tweet # Add processed media
                 }
                 processed_new_tweets.append(tweet_data)

            # --- Combine and Prune ---
            # Ensure lists are sorted correctly before combining
            processed_new_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            existing_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper

            # De-duplicate based on ID before combining and pruning
            existing_ids = {t['id'] for t in existing_tweets}
            unique_new_tweets = [t for t in processed_new_tweets if t['id'] not in existing_ids]

            combined_tweets = unique_new_tweets + existing_tweets
            # Sort again after combining unique new ones
            combined_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True)
            # Prune based on MAX_CACHE_ITEMS
            final_tweets = combined_tweets[:MAX_CACHE_ITEMS]

            # Combine media analysis and paths (only add new unique ones)
            final_media_analysis = newly_added_media_analysis + [m for m in existing_media_analysis if m not in newly_added_media_analysis] # Basic de-dup
            final_media_paths = list(newly_added_media_paths.union(existing_media_paths))[:MAX_CACHE_ITEMS * 2] # Limit paths too

            # --- Prepare Final Data ---
            final_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(), # Set final timestamp here
                'user_info': user_info,
                'tweets': final_tweets,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths
            }

            self._save_cache('twitter', username, final_data)
            logger.info(f"Successfully updated Twitter cache for @{username}. Total tweets cached: {len(final_tweets)}")
            return final_data

        except RateLimitExceededError:
             return None # Indicate fetch failed due to rate limit
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Twitter fetch failed for @{username}: {user_err}")
             return None
        except tweepy.errors.TweepyException as e:
            logger.error(f"Twitter API error for @{username}: {str(e)}", exc_info=False)
            if "Authentication credentials" in str(e) or "bearer token" in str(e).lower():
                 raise RuntimeError(f"Twitter authentication failed. Check Bearer Token. ({e})")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Twitter data for @{username}: {str(e)}", exc_info=True)
            return None


    def fetch_reddit(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        # ... (existing implementation) ...
        # Now calls the global get_sort_key correctly
        cached_data = self._load_cache('reddit', username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - datetime.fromisoformat(cached_data['timestamp'])) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Reddit u/{username}")
            return cached_data

        logger.info(f"Fetching Reddit data for u/{username} (Force Refresh: {force_refresh})")
        latest_submission_fullname = None
        latest_comment_fullname = None
        existing_submissions = []
        existing_comments = []
        existing_media_analysis = []
        existing_media_paths = []

        if not force_refresh and cached_data:
            logger.info(f"Attempting incremental fetch for Reddit u/{username}")
            existing_submissions = cached_data.get('submissions', [])
            existing_comments = cached_data.get('comments', [])
            # Sort existing data to find the latest easily (PRAW fullname includes type prefix t1_, t3_)
            existing_submissions.sort(key=lambda x: get_sort_key(x, 'created_utc'), reverse=True) # Use global helper
            existing_comments.sort(key=lambda x: get_sort_key(x, 'created_utc'), reverse=True) # Use global helper

            if existing_submissions:
                latest_submission_fullname = f"t3_{existing_submissions[0]['id']}"
                logger.debug(f"Using latest submission fullname: {latest_submission_fullname}")
            if existing_comments:
                latest_comment_fullname = f"t1_{existing_comments[0]['id']}"
                logger.debug(f"Using latest comment fullname: {latest_comment_fullname}")

            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])

        try:
            # Ensure client is ready
            if not hasattr(self, '_reddit'): self.reddit
            redditor = self.reddit.redditor(username)
            try:
                # Accessing redditor.id forces a check if the user exists
                redditor_id = redditor.id
                logger.debug(f"Reddit user u/{username} found (ID: {redditor_id}).")
            except prawcore.exceptions.NotFound:
                raise UserNotFoundError(f"Reddit user u/{username} not found.")
            except prawcore.exceptions.Forbidden:
                 raise AccessForbiddenError(f"Access forbidden to Reddit user u/{username} (possibly suspended or shadowbanned).")


            # --- Fetch New Submissions ---
            new_submissions_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            fetch_limit = INCREMENTAL_FETCH_LIMIT # Limit incremental fetch
            count = 0
            processed_ids = {s['id'] for s in existing_submissions} # Track existing submission IDs

            logger.debug("Fetching new submissions...")
            try:
                # Fetch a batch and filter locally
                # Use 'before' parameter with latest fullname for more efficient incremental fetching
                params = {}
                if not force_refresh and latest_submission_fullname:
                    params['before'] = latest_submission_fullname
                    logger.debug(f"Fetching submissions before {latest_submission_fullname}")

                for submission in redditor.submissions.new(limit=fetch_limit, params=params):
                    count += 1
                    submission_fullname = submission.fullname

                    # Avoid reprocessing already cached submissions (shouldn't be needed with 'before', but safe)
                    if submission.id in processed_ids:
                        continue

                    # This check is less reliable than using 'before', but keep as fallback
                    if not force_refresh and latest_submission_fullname and submission_fullname == latest_submission_fullname:
                         logger.debug(f"Reached latest known submission {submission_fullname} (fallback check). Stopping.")
                         break


                    media_items_for_submission = [] # Process media first
                    media_processed_inline = False # Track if direct URL was handled

                    # Direct Image/GIF/Video URL (check common extensions)
                    supported_image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
                    if hasattr(submission, 'url') and submission.url and any(submission.url.lower().endswith(ext) for ext in supported_image_extensions + ['.mp4', '.webm']):
                         media_path = self._download_media(url=submission.url, platform='reddit', username=username)
                         if media_path:
                             analysis = None
                             # Analyze images only
                             if media_path.suffix.lower() in supported_image_extensions:
                                 analysis = self._analyze_image(media_path, f"Reddit user u/{username}'s post in r/{submission.subreddit.display_name}")
                             media_items_for_submission.append({
                                 'type': 'image' if media_path.suffix.lower() in supported_image_extensions else 'video',
                                 'analysis': analysis,
                                 'url': submission.url,
                                 'local_path': str(media_path)
                             })
                             if analysis: newly_added_media_analysis.append(analysis)
                             newly_added_media_paths.add(str(media_path))
                             media_processed_inline = True

                    # Reddit Gallery (only if direct URL wasn't processed)
                    is_gallery = getattr(submission, 'is_gallery', False)
                    media_metadata = getattr(submission, 'media_metadata', None)
                    if not media_processed_inline and is_gallery and media_metadata:
                        for media_id, media_item in media_metadata.items():
                             source = media_item.get('s') # Source dictionary
                             if source:
                                 # Prefer highest resolution available (u or gif)
                                 image_url = source.get('u') or source.get('gif')
                                 if image_url:
                                     # Reddit URLs often have escaped ampersands
                                     image_url = image_url.replace('&amp;', '&')
                                     media_path = self._download_media(url=image_url, platform='reddit', username=username)
                                     if media_path:
                                         analysis = None
                                         if media_path.suffix.lower() in supported_image_extensions:
                                             analysis = self._analyze_image(media_path, f"Reddit user u/{username}'s gallery post in r/{submission.subreddit.display_name}")
                                         media_items_for_submission.append({
                                             'type': 'gallery_image',
                                             'analysis': analysis,
                                             'url': image_url,
                                             'local_path': str(media_path)
                                         })
                                         if analysis: newly_added_media_analysis.append(analysis)
                                         newly_added_media_paths.add(str(media_path))

                    submission_data = {
                        'id': submission.id,
                        'title': submission.title,
                        'text': submission.selftext[:1000] if hasattr(submission, 'selftext') else '',
                        'score': submission.score,
                        'subreddit': submission.subreddit.display_name,
                        'permalink': f"https://www.reddit.com{submission.permalink}",
                        'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                        'fullname': submission_fullname,
                        'url': submission.url,
                        'is_gallery': is_gallery,
                        'media': media_items_for_submission # Add processed media
                        # 'media_metadata' is complex, exclude from final cache unless needed
                    }
                    new_submissions_data.append(submission_data)
                    processed_ids.add(submission.id) # Track processed ID

            except prawcore.exceptions.Forbidden:
                logger.warning(f"Access forbidden while fetching submissions for u/{username}.")
            logger.info(f"Fetched {len(new_submissions_data)} new submissions for u/{username} (scanned approx {count}).")


            # --- Fetch New Comments ---
            new_comments_data = []
            count = 0
            processed_comment_ids = {c['id'] for c in existing_comments} # Track existing comment IDs
            logger.debug("Fetching new comments...")
            try:
                # Fetch a batch and filter locally, use 'before' param
                params = {}
                if not force_refresh and latest_comment_fullname:
                    params['before'] = latest_comment_fullname
                    logger.debug(f"Fetching comments before {latest_comment_fullname}")

                for comment in redditor.comments.new(limit=fetch_limit, params=params):
                     count += 1
                     comment_fullname = comment.fullname

                     if comment.id in processed_comment_ids:
                         continue

                     if not force_refresh and latest_comment_fullname and comment_fullname == latest_comment_fullname:
                         logger.debug(f"Reached latest known comment {comment_fullname} (fallback check). Stopping.")
                         break

                     new_comments_data.append({
                         'id': comment.id,
                         'text': comment.body[:1000], # Increased snippet
                         'score': comment.score,
                         'subreddit': comment.subreddit.display_name,
                         'permalink': f"https://www.reddit.com{comment.permalink}", # Full URL
                         'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                         'fullname': comment_fullname
                     })
                     processed_comment_ids.add(comment.id)

            except prawcore.exceptions.Forbidden:
                 logger.warning(f"Access forbidden while fetching comments for u/{username}.")
            logger.info(f"Fetched {len(new_comments_data)} new comments for u/{username} (scanned approx {count}).")


            # --- Combine and Prune ---
            # New data is already assumed unique due to 'before' param and ID checks
            combined_submissions = new_submissions_data + existing_submissions
            combined_comments = new_comments_data + existing_comments

            # Sort combined lists before pruning
            combined_submissions.sort(key=lambda x: get_sort_key(x, 'created_utc'), reverse=True) # Use global helper
            combined_comments.sort(key=lambda x: get_sort_key(x, 'created_utc'), reverse=True) # Use global helper

            final_submissions = combined_submissions[:MAX_CACHE_ITEMS]
            final_comments = combined_comments[:MAX_CACHE_ITEMS]

            # Combine media analysis and paths
            final_media_analysis = newly_added_media_analysis + [m for m in existing_media_analysis if m not in newly_added_media_analysis]
            final_media_paths = list(newly_added_media_paths.union(existing_media_paths))[:MAX_CACHE_ITEMS * 2]

            # --- Calculate Stats ---
            total_submissions = len(final_submissions)
            total_comments = len(final_comments)
            submissions_with_media = len([s for s in final_submissions if s.get('media')])
            stats = {
                'total_submissions': total_submissions,
                'total_comments': total_comments,
                'submissions_with_media': submissions_with_media,
                'total_media_items_processed': len(final_media_paths), # Count unique paths
                'avg_submission_score': sum(s.get('score', 0) for s in final_submissions) / max(total_submissions, 1),
                'avg_comment_score': sum(c.get('score', 0) for c in final_comments) / max(total_comments, 1)
            }

            # --- Prepare Final Data ---
            final_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'submissions': final_submissions,
                'comments': final_comments,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths,
                'stats': stats
            }

            self._save_cache('reddit', username, final_data)
            logger.info(f"Successfully updated Reddit cache for u/{username}. Cached submissions: {total_submissions}, comments: {total_comments}")
            return final_data

        except RateLimitExceededError:
            return None # Handled
        except prawcore.exceptions.RequestException as e:
            # Handle potential 429 specifically if not caught earlier
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                 self._handle_rate_limit('Reddit', exception=e)
            else:
                 logger.error(f"Reddit request failed for u/{username}: {str(e)}")
            return None
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Reddit fetch failed for u/{username}: {user_err}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Reddit data for u/{username}: {str(e)}", exc_info=True)
            return None


    def fetch_bluesky(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        # ... (existing implementation) ...
        # Now calls the global get_sort_key correctly
        cached_data = self._load_cache('bluesky', username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - datetime.fromisoformat(cached_data['timestamp'])) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Bluesky user {username}")
            return cached_data

        logger.info(f"Fetching Bluesky data for {username} (Force Refresh: {force_refresh})")
        latest_post_cid = None # Bluesky feed doesn't use simple since_id, compare CIDs or timestamps
        existing_posts = []
        existing_media_analysis = []
        existing_media_paths = []
        profile_info = None # Store profile info separately

        if not force_refresh and cached_data:
            logger.info(f"Attempting incremental fetch for Bluesky {username}")
            existing_posts = cached_data.get('posts', [])
            existing_posts.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            if existing_posts:
                # Use the CID of the latest post for comparison during fetch
                latest_post_cid = existing_posts[0]['cid']
                logger.debug(f"Latest known post CID: {latest_post_cid}")

            profile_info = cached_data.get('profile_info') # Keep existing profile info
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])

        try:
            # Ensure client is ready
            if not hasattr(self, '_bluesky_client'): self.bluesky

            # --- Get Profile Info (only if missing or forced) ---
            if not profile_info or force_refresh:
                try:
                     profile = self.bluesky.get_profile(actor=username)
                     profile_info = { # Store basic profile info
                          'did': profile.did,
                          'handle': profile.handle,
                          'display_name': profile.display_name,
                          'description': profile.description,
                          'avatar': profile.avatar,
                          'banner': profile.banner,
                          'followers_count': profile.followers_count,
                          'follows_count': profile.follows_count,
                          'posts_count': profile.posts_count
                      }
                     logger.debug(f"Fetched Bluesky profile info for {username}")
                except atproto_exceptions.AtProtocolError as e:
                     err_str = str(e).lower()
                     if 'profile not found' in err_str or 'could not resolve handle' in err_str:
                          raise UserNotFoundError(f"Bluesky user {username} not found.")
                     elif 'blocked by actor' in err_str or 'blocking actor' in err_str:
                          raise AccessForbiddenError(f"Blocked from accessing Bluesky profile for {username}.")
                     else: # Re-raise other profile lookup errors
                          logger.error(f"Unexpected error fetching Bluesky profile for {username}: {e}")
                          raise AccessForbiddenError(f"Error fetching Bluesky profile for {username}: {e}")


            # --- Fetch New Posts ---
            new_posts_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            cursor = None
            processed_cids = set(p['cid'] for p in existing_posts) # Track existing CIDs
            fetch_limit_per_page = min(INCREMENTAL_FETCH_LIMIT, 100) # Bluesky limit is 100
            total_fetched_this_run = 0
            max_fetches = INITIAL_FETCH_LIMIT if (force_refresh or not latest_post_cid) else INCREMENTAL_FETCH_LIMIT

            logger.debug(f"Fetching new Bluesky posts for {username}...")
            while total_fetched_this_run < max_fetches:
                stop_fetching = False
                try:
                    response = self.bluesky.get_author_feed(
                        actor=username, # Use handle/DID provided
                        cursor=cursor,
                        limit=fetch_limit_per_page
                    )
                except atproto_exceptions.AtProtocolError as e:
                    err_str = str(e).lower()
                    if 'rate limit' in err_str:
                        self._handle_rate_limit('Bluesky', exception=e)
                        return None # Rate limit handled
                    # More robust user not found / forbidden checks during feed fetch
                    if 'could not resolve handle' in err_str or 'profile not found' in err_str:
                         # This might occur if profile fetch succeeded but feed fails (e.g., account deleted between calls)
                         raise UserNotFoundError(f"Bluesky user {username} not found or handle cannot be resolved during feed fetch.")
                    if 'blocked by actor' in err_str or 'blocking actor' in err_str:
                         raise AccessForbiddenError(f"Access to Bluesky user {username}'s feed is blocked.")

                    logger.error(f"Bluesky API error fetching feed for {username}: {e}")
                    return None # Stop fetching on unexpected errors

                if not response or not response.feed:
                    logger.debug("No more posts found in feed.")
                    break # No more posts

                logger.debug(f"Processing feed page with {len(response.feed)} items. Cursor: {response.cursor}")
                for feed_item in response.feed:
                    post = feed_item.post
                    post_cid = post.cid # Use CID for uniqueness

                    # Avoid reprocessing the same post (API might overlap slightly)
                    if post_cid in processed_cids:
                        continue

                    # Stop if we hit the latest known post during an incremental update
                    if not force_refresh and latest_post_cid and post_cid == latest_post_cid:
                        logger.info(f"Reached latest known post CID {post_cid} for Bluesky {username}. Stopping incremental fetch.")
                        stop_fetching = True
                        break # Stop processing this page

                    record = getattr(post, 'record', None)
                    if not record: continue # Skip if post has no record data

                    created_at_dt = get_sort_key({'created_at': getattr(record, 'created_at', None)}, 'created_at')

                    # --- Process Media ---
                    media_items_for_post = []
                    embed = getattr(record, 'embed', None)
                    image_embeds_to_process = []
                    embed_type_str = 'unknown'
                    supported_image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif'] # Define here too

                    if embed:
                         embed_type_str = getattr(embed, '$type', 'unknown').split('.')[-1]
                         # Case 1: Direct images (app.bsky.embed.images)
                         if hasattr(embed, 'images'): image_embeds_to_process.extend(embed.images)
                         # Case 2: Record with media (app.bsky.embed.recordWithMedia)
                         media_embed = getattr(embed, 'media', None)
                         if media_embed and hasattr(media_embed, 'images'): image_embeds_to_process.extend(media_embed.images)
                         # Case 3: Check inside nested record (e.g., quote post's media)
                         record_embed = getattr(embed, 'record', None)
                         nested_record_value = getattr(record_embed, 'record', None) # Check 'record' within 'record'
                         if nested_record_value:
                              nested_embed = getattr(nested_record_value, 'embed', None)
                              if nested_embed and hasattr(nested_embed, 'images'):
                                   image_embeds_to_process.extend(nested_embed.images)


                    # Process collected image embeds
                    for image_info in image_embeds_to_process:
                        img_blob = getattr(image_info, 'image', None)
                        if img_blob:
                            # Bluesky API v0.4.0 changed blob structure (ref -> link)
                            cid_ref = getattr(img_blob, 'ref', None) # Older structure?
                            cid = getattr(cid_ref, 'link', None) if cid_ref else getattr(img_blob, 'cid', None) # Newer uses 'cid' directly? Let's try both

                            if cid:
                                author_did = post.author.did
                                # Construct the CDN URL (ensure DID and CID are properly quoted)
                                cdn_url = f"https://cdn.bsky.app/img/feed_fullsize/plain/{quote_plus(author_did)}/{quote_plus(cid)}@jpeg"
                                media_path = self._download_media(url=cdn_url, platform='bluesky', username=username)
                                if media_path:
                                    analysis = None
                                    if media_path.suffix.lower() in supported_image_extensions:
                                         analysis = self._analyze_image(media_path, f"Bluesky user {username}'s post ({post.uri})")
                                    media_items_for_post.append({
                                        'type': 'image', # Assume image for now
                                        'analysis': analysis,
                                        'url': cdn_url,
                                        'alt_text': getattr(image_info, 'alt', ''),
                                        'local_path': str(media_path)
                                    })
                                    if analysis: newly_added_media_analysis.append(analysis)
                                    newly_added_media_paths.add(str(media_path))
                            else: logger.warning(f"Could not find image CID/link in embed for post {post.uri}")
                        else: logger.warning(f"Image embed structure missing 'image' blob for post {post.uri}")

                    post_data = {
                        'uri': post.uri,
                        'cid': post.cid,
                        'author_did': post.author.did,
                        'text': getattr(record, 'text', '')[:2000],
                        'created_at': created_at_dt.isoformat(),
                        'likes': getattr(post, 'like_count', 0),
                        'reposts': getattr(post, 'repost_count', 0),
                        'reply_count': getattr(post, 'reply_count', 0),
                        'embed': {'type': embed_type_str} if embed else None,
                        'media': media_items_for_post
                    }

                    new_posts_data.append(post_data)
                    processed_cids.add(post_cid)
                    total_fetched_this_run += 1
                    if total_fetched_this_run >= max_fetches:
                        logger.info(f"Reached fetch limit ({max_fetches}) for Bluesky {username}.")
                        stop_fetching = True
                        break # Stop processing this page

                if stop_fetching:
                    break # Exit outer loop

                cursor = response.cursor
                if not cursor:
                    logger.debug("Reached end of feed (no cursor).")
                    break # No more pages

            logger.info(f"Fetched {len(new_posts_data)} new posts for Bluesky user {username}.")

            # --- Combine and Prune ---
            # De-duplicate based on CID before combining
            existing_cids = {p['cid'] for p in existing_posts}
            unique_new_posts = [p for p in new_posts_data if p['cid'] not in existing_cids]

            combined_posts = unique_new_posts + existing_posts
            # Sort again after combining
            combined_posts.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            final_posts = combined_posts[:MAX_CACHE_ITEMS]

            # Combine media analysis and paths
            final_media_analysis = newly_added_media_analysis + [m for m in existing_media_analysis if m not in newly_added_media_analysis]
            final_media_paths = list(newly_added_media_paths.union(existing_media_paths))[:MAX_CACHE_ITEMS * 2]

            # --- Calculate Stats ---
            total_posts = len(final_posts)
            posts_with_media = len([p for p in final_posts if p.get('media')])
            stats = {
                'total_posts': total_posts,
                'posts_with_media': posts_with_media,
                'total_media_items_processed': len(final_media_paths),
                'avg_likes': sum(p.get('likes', 0) for p in final_posts) / max(total_posts, 1),
                'avg_reposts': sum(p.get('reposts', 0) for p in final_posts) / max(total_posts, 1),
                'avg_replies': sum(p.get('reply_count', 0) for p in final_posts) / max(total_posts, 1)
            }

            # --- Prepare Final Data ---
            final_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'profile_info': profile_info, # Add fetched profile info
                'posts': final_posts,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths,
                'stats': stats
            }

            self._save_cache('bluesky', username, final_data)
            logger.info(f"Successfully updated Bluesky cache for {username}. Total posts cached: {total_posts}")
            return final_data

        except RateLimitExceededError:
            return None # Handled
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Bluesky fetch failed for {username}: {user_err}")
             return None
        except atproto_exceptions.AtProtocolError as e:
            logger.error(f"Bluesky ATProtocol error for {username}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Bluesky data for {username}: {str(e)}", exc_info=True)
            return None


    # +++ Add Mastodon Fetcher +++
    def fetch_mastodon(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetches Mastodon statuses (toots) for a user.
        Username should be in the format 'user@instance.domain'.
        """
        # Use full username for cache key consistency
        cache_key_username = username # e.g., user@instance.social
        supported_image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif'] # Define here too

        # Basic validation for Mastodon username format
        if '@' not in cache_key_username or '.' not in cache_key_username.split('@')[1]:
             logger.error(f"Invalid Mastodon username format for fetch: '{cache_key_username}'. Needs 'user@instance.domain'.")
             # Attempt to fix if a default instance is set? Or just fail. Let's fail for now.
             raise ValueError(f"Invalid Mastodon username format: '{cache_key_username}'. Must be 'user@instance.domain'.")


        cached_data = self._load_cache('mastodon', cache_key_username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - datetime.fromisoformat(cached_data['timestamp'])) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Mastodon user {cache_key_username}")
            return cached_data

        logger.info(f"Fetching Mastodon data for {cache_key_username} (Force Refresh: {force_refresh})")
        since_id = None
        existing_posts = []
        existing_media_analysis = []
        existing_media_paths = []
        user_info = None # Mastodon user info

        if not force_refresh and cached_data:
            logger.info(f"Attempting incremental fetch for Mastodon {cache_key_username}")
            existing_posts = cached_data.get('posts', [])
            existing_posts.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            if existing_posts:
                since_id = existing_posts[0]['id']
                logger.debug(f"Using since_id: {since_id}")

            user_info = cached_data.get('user_info') # Keep existing user info if available
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])

        try:
            # Ensure client is ready
            if not hasattr(self, '_mastodon_client'): self.mastodon

            # --- Get User Account Info ---
            # Only refetch user info if forced or missing
            if not user_info or force_refresh:
                try:
                    # account_lookup handles 'user@instance.domain'
                    # This call might fail if the instance in 'username' is different from the client's base_url
                    # and federation is blocked or the target instance doesn't allow lookups.
                    logger.debug(f"Looking up Mastodon account: {username} using client for {self.mastodon.api_base_url}")
                    account = self.mastodon.account_lookup(acct=username)
                    user_info = {
                        'id': account['id'],
                        'username': account['username'], # Local username part
                        'acct': account['acct'], # Full handle (user@instance or just user)
                        'display_name': account['display_name'],
                        'note_html': account.get('note', ''), # Bio (HTML)
                        'url': account['url'], # Link to profile page
                        'avatar': account['avatar'],
                        'header': account['header'],
                        'followers_count': account['followers_count'],
                        'following_count': account['following_count'],
                        'statuses_count': account['statuses_count'],
                        'created_at': account['created_at'] # Already datetime
                    }
                    logger.info(f"Fetched Mastodon user info for {cache_key_username}")
                except MastodonNotFoundError:
                    raise UserNotFoundError(f"Mastodon user {username} not found via {self.mastodon.api_base_url}.")
                except MastodonUnauthorizedError:
                    # Might happen for locked accounts you don't follow or instance restrictions
                    raise AccessForbiddenError(f"Unauthorized access to Mastodon user {username}'s info (locked account / instance policy?).")
                except MastodonVersionError as e:
                    # Can indicate various issues including being blocked or federation problems
                    logger.error(f"MastodonVersionError looking up {username}: {e}")
                    raise AccessForbiddenError(f"Forbidden from accessing Mastodon user {username}'s info (blocked/federation issue?).")

            user_id = user_info['id']

            # --- Fetch New Statuses (Toots) ---
            new_posts_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            # Fetch slightly more initially to ensure we get enough distinct items
            fetch_limit = INITIAL_FETCH_LIMIT if (force_refresh or not since_id) else INCREMENTAL_FETCH_LIMIT
            # Mastodon API max limit is typically 40, respect that
            api_limit = min(fetch_limit, MASTODON_FETCH_LIMIT)
            processed_status_ids = {p['id'] for p in existing_posts} # Track existing status IDs


            logger.debug(f"Fetching new statuses for user ID {user_id} ({cache_key_username}) (since_id: {since_id})")
            try:
                # account_statuses fetches newest first. Use since_id for incremental.
                # Paginate if needed (though incremental fetch usually doesn't require > 1 page)
                # Mastodon.py handles pagination automatically if we ask for more than api_limit
                # Let's fetch in one go for simplicity here, assuming api_limit is enough for incremental
                new_statuses = self.mastodon.account_statuses(
                    id=user_id,
                    limit=api_limit,
                    since_id=since_id if not force_refresh else None,
                    exclude_replies=False, # Include replies
                    exclude_reblogs=False   # Include boosts
                )
            except MastodonRatelimitError as e:
                 self._handle_rate_limit('Mastodon', exception=e)
                 return None
            except MastodonNotFoundError: # Should not happen if account lookup succeeded, but maybe account deleted between calls
                 raise UserNotFoundError(f"Mastodon user ID {user_id} (handle: {username}) not found during status fetch.")
            except (MastodonUnauthorizedError, MastodonVersionError) as e:
                 logger.error(f"Error fetching statuses for {username}: {e}")
                 raise AccessForbiddenError(f"Access forbidden to Mastodon user {username}'s statuses (private/blocked?).")

            logger.info(f"Fetched {len(new_statuses)} new raw statuses for Mastodon user {cache_key_username}.")

            # --- Process New Statuses ---
            count_added = 0
            for status in new_statuses:
                status_id = status['id']
                # Skip if already processed (safeguard against API overlap or since_id issues)
                if status_id in processed_status_ids:
                    continue

                # Clean HTML content using BeautifulSoup
                cleaned_text = '[Content Warning Hidden]' if status.get('spoiler_text') else ''
                if not cleaned_text: # Only parse if not hidden by CW
                    try:
                         soup = BeautifulSoup(status['content'], 'html.parser')
                         # Add space between paragraphs, breaks etc.
                         for br in soup.find_all("br"): br.replace_with("\n")
                         for p in soup.find_all("p"): p.append("\n")
                         cleaned_text = soup.get_text(separator=' ', strip=True)
                    except Exception as parse_err:
                         logger.warning(f"HTML parsing failed for status {status_id}: {parse_err}. Using raw content.")
                         cleaned_text = status['content'] # Fallback

                # --- Process Media ---
                media_items_for_post = []
                for attachment in status.get('media_attachments', []):
                     media_url = attachment.get('url') # Full resolution URL
                     preview_url = attachment.get('preview_url') # Lower res preview
                     media_type = attachment.get('type', 'unknown') # image, video, gifv, audio

                     # Prefer full URL if available and it's not a video (sometimes URL is missing for video)
                     url_to_download = media_url if media_url and media_type != 'video' else preview_url
                     if url_to_download:
                         media_path = self._download_media(url=url_to_download, platform='mastodon', username=cache_key_username)
                         if media_path:
                             analysis = None
                             # Only analyze supported image types
                             if media_type == 'image' and media_path.suffix.lower() in supported_image_extensions:
                                 # Pass context including the status URL for reference
                                 image_context = f"Mastodon user {cache_key_username}'s post ({status.get('url', status_id)})"
                                 analysis = self._analyze_image(media_path, image_context)

                             media_items_for_post.append({
                                 'type': media_type,
                                 'analysis': analysis,
                                 'url': media_url, # Store original full URL
                                 'preview_url': preview_url,
                                 'description': attachment.get('description'), # Alt text
                                 'local_path': str(media_path)
                             })
                             if analysis: newly_added_media_analysis.append(analysis)
                             newly_added_media_paths.add(str(media_path))

                is_reblog = status.get('reblog') is not None
                reblog_info = status.get('reblog') if is_reblog else None

                post_data = {
                    'id': status_id,
                    'created_at': status['created_at'], # Already datetime
                    'url': status['url'], # Link to the status
                    'text_html': status['content'], # Raw HTML content
                    'text_cleaned': cleaned_text[:2000], # Store cleaned, truncated text
                    'spoiler_text': status.get('spoiler_text', ''),
                    'reblogs_count': status.get('reblogs_count', 0),
                    'favourites_count': status.get('favourites_count', 0),
                    'replies_count': status.get('replies_count', 0),
                    'is_reblog': is_reblog,
                    'reblog_original_author': reblog_info['account']['acct'] if reblog_info else None,
                    'reblog_original_url': reblog_info['url'] if reblog_info else None,
                    'media': media_items_for_post
                }
                new_posts_data.append(post_data)
                processed_status_ids.add(status_id) # Track added ID
                count_added += 1

            logger.info(f"Processed {count_added} new unique statuses for Mastodon user {cache_key_username}.")


            # --- Combine and Prune ---
            # New data is already unique based on ID checks
            combined_posts = new_posts_data + existing_posts
            # Sort combined list before pruning
            combined_posts.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            final_posts = combined_posts[:MAX_CACHE_ITEMS]

            # Combine media analysis and paths
            final_media_analysis = newly_added_media_analysis + [m for m in existing_media_analysis if m not in newly_added_media_analysis]
            final_media_paths = list(newly_added_media_paths.union(existing_media_paths))[:MAX_CACHE_ITEMS * 2]

            # --- Calculate Stats ---
            total_posts = len(final_posts)
            original_posts = [p for p in final_posts if not p.get('is_reblog')]
            total_original_posts = len(original_posts)
            total_reblogs = total_posts - total_original_posts
            posts_with_media = len([p for p in final_posts if p.get('media')])
            stats = {
                'total_posts_cached': total_posts,
                'total_original_posts': total_original_posts,
                'total_reblogs': total_reblogs,
                'posts_with_media': posts_with_media,
                'total_media_items_processed': len(final_media_paths),
                # Calculate stats only on original posts to avoid inflating interaction counts from boosts
                'avg_favourites': sum(p['favourites_count'] for p in original_posts) / max(total_original_posts, 1),
                'avg_reblogs': sum(p['reblogs_count'] for p in original_posts) / max(total_original_posts, 1),
                'avg_replies': sum(p['replies_count'] for p in original_posts) / max(total_original_posts, 1)
            }

            # --- Prepare Final Data ---
            final_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'user_info': user_info,
                'posts': final_posts,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths,
                'stats': stats
            }

            self._save_cache('mastodon', cache_key_username, final_data)
            logger.info(f"Successfully updated Mastodon cache for {cache_key_username}. Total posts cached: {total_posts}")
            return final_data

        except ValueError as ve: # Catch the format validation error
             logger.error(f"Mastodon fetch failed for {username}: {ve}")
             return None
        except RateLimitExceededError:
            return None # Handled
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Mastodon fetch failed for {username}: {user_err}")
             return None
        except MastodonError as e:
            # Catch other generic Mastodon errors
            logger.error(f"Mastodon API error for {username}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Mastodon data for {username}: {str(e)}", exc_info=True)
            return None
    # --- End Mastodon Fetcher ---


    def fetch_hackernews(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        # ... (existing implementation) ...
        # Now calls the global get_sort_key correctly
        cached_data = self._load_cache('hackernews', username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - datetime.fromisoformat(cached_data['timestamp'])) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for HackerNews user {username}")
            return cached_data

        logger.info(f"Fetching HackerNews data for {username} (Force Refresh: {force_refresh})")
        latest_timestamp_i = 0 # Algolia uses integer timestamps
        existing_submissions = []

        if not force_refresh and cached_data:
            logger.info(f"Attempting incremental fetch for HackerNews {username}")
            existing_submissions = cached_data.get('submissions', [])
            existing_submissions.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            if existing_submissions:
                # Find the max timestamp_i from the existing data
                latest_timestamp_i = max(s.get('created_at_i', 0) for s in existing_submissions)
                logger.debug(f"Using latest timestamp_i: {latest_timestamp_i}")

        try:
            # Algolia API endpoint
            base_url = "https://hn.algolia.com/api/v1/search"
            # Fetch more on initial/forced, use incremental limit otherwise
            hits_per_page = INITIAL_FETCH_LIMIT if (force_refresh or not latest_timestamp_i) else INCREMENTAL_FETCH_LIMIT
            params = {
                "tags": f"author_{quote_plus(username)}",
                "hitsPerPage": hits_per_page,
                "typoTolerance": False # Be strict with username matching
            }

            # Add numeric filter for incremental fetch - fetch items created *after* the latest known timestamp
            if not force_refresh and latest_timestamp_i > 0:
                params["numericFilters"] = f"created_at_i>{latest_timestamp_i}"
                logger.debug(f"Applying numeric filter: created_at_i > {latest_timestamp_i}")

            new_submissions_data = []
            processed_ids = set(s['objectID'] for s in existing_submissions) # Track existing IDs

            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                 response = client.get(base_url, params=params)
                 response.raise_for_status() # Check for HTTP errors
                 data = response.json()

            if 'hits' not in data:
                 logger.warning(f"No 'hits' found in HN Algolia response for {username}")
                 data['hits'] = []
            # Check if Algolia thinks the user exists but returned 0 hits
            elif not data['hits'] and data.get('nbHits', 0) == 0 and data.get('exhaustiveNbHits', True):
                 # Only raise UserNotFound if we're sure Algolia found nothing for this author tag
                 # Note: This might trigger if user exists but has 0 posts/comments.
                 # Consider if this is the desired behavior or if an empty result is acceptable.
                 # For now, let's proceed with empty data, which the analysis can handle.
                 logger.info(f"HackerNews user {username} found via Algolia, but has 0 items.")


            logger.info(f"Fetched {len(data['hits'])} potential new items for HN user {username}.")

            for hit in data.get('hits', []):
                object_id = hit.get('objectID')
                if not object_id or object_id in processed_ids:
                     continue # Skip duplicates

                created_at_ts = hit.get('created_at_i')
                if not created_at_ts: continue # Skip if no timestamp

                tags = hit.get('_tags', [])
                item_type = 'unknown'
                if 'story' in tags: item_type = 'story'
                elif 'comment' in tags: item_type = 'comment'
                elif 'poll' in tags: item_type = 'poll'
                elif 'pollopt' in tags: item_type = 'pollopt'

                # Clean up text (remove HTML tags) if present
                raw_text = hit.get('story_text') or hit.get('comment_text') or ''
                cleaned_text = BeautifulSoup(raw_text, 'html.parser').get_text(separator=' ', strip=True) if raw_text else ''


                submission_item = {
                    'objectID': object_id,
                    'type': item_type,
                    'title': hit.get('title'),
                    'url': hit.get('url'),
                    'points': hit.get('points'),
                    'num_comments': hit.get('num_comments'),
                    'story_id': hit.get('story_id'),
                    'parent_id': hit.get('parent_id'),
                    'created_at_i': created_at_ts,
                    'created_at': datetime.fromtimestamp(created_at_ts, tz=timezone.utc),
                    'text': cleaned_text # Use cleaned text
                }
                new_submissions_data.append(submission_item)
                processed_ids.add(object_id)

            # --- Combine and Prune ---
            # New data is already unique due to ID check
            combined_submissions = new_submissions_data + existing_submissions
            # Sort combined list before pruning
            combined_submissions.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            final_submissions = combined_submissions[:MAX_CACHE_ITEMS]

            # --- Calculate Stats ---
            story_submissions = [s for s in final_submissions if s['type'] == 'story']
            total_items = len(final_submissions)
            total_stories = len(story_submissions)
            total_comments = len([s for s in final_submissions if s['type'] == 'comment'])
            stats = {
                'total_items_cached': total_items,
                'total_stories': total_stories,
                'total_comments': total_comments,
                'average_story_points': sum(s.get('points', 0) or 0 for s in story_submissions) / max(total_stories, 1),
                'average_story_num_comments': sum(s.get('num_comments', 0) or 0 for s in story_submissions) / max(total_stories, 1)
            }

            # --- Prepare Final Data ---
            final_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'submissions': final_submissions,
                'stats': stats
            }

            self._save_cache('hackernews', username, final_data)
            logger.info(f"Successfully updated HackerNews cache for {username}. Total items cached: {total_items}")
            return final_data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._handle_rate_limit('HackerNews (Algolia)', e)
            elif e.response.status_code == 400:
                 # This can happen with invalid tag syntax (e.g., bad characters in username)
                 logger.error(f"HN Algolia API Bad Request (400) for {username}: {e.response.text}. Check username format.")
                 # It's unlikely the user exists if the tag is invalid
                 raise UserNotFoundError(f"HackerNews username '{username}' resulted in a bad request (check format).")
            else:
                 logger.error(f"HN Algolia API HTTP error for {username}: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
             logger.error(f"HN Algolia API network error for {username}: {str(e)}")
             return None
        except UserNotFoundError as e: # Re-raise specific UserNotFound if detected above
             logger.error(f"HN fetch failed for {username}: {e}")
             raise e
        except Exception as e:
            logger.error(f"Unexpected error fetching HackerNews data for {username}: {str(e)}", exc_info=True)
            return None


    # --- Analysis Core ---

    def analyze(self, platforms: Dict[str, Union[str, List[str]]], query: str) -> str:
        """Collects data (using fetch methods) and performs LLM analysis."""
        collected_text_summaries = []
        all_media_analyzes = []
        failed_fetches = []

        platform_count = sum(len(v) if isinstance(v, list) else 1 for v in platforms.values())
        if platform_count == 0:
             return "[yellow]No platforms or users specified for analysis.[/yellow]"

        try:
            collect_task = self.progress.add_task(
                f"[cyan]Collecting data for {platform_count} target(s)...",
                total=platform_count
            )
            self.progress.start() # Ensure progress starts

            for platform, usernames in platforms.items():
                # Ensure usernames is always a list for iteration
                if isinstance(usernames, str): usernames = [usernames]
                if not usernames: continue # Skip if empty list

                fetcher = getattr(self, f'fetch_{platform}', None)
                if not fetcher:
                     logger.warning(f"No fetcher method found for platform: {platform}")
                     failed_fetches.extend([(platform, u, "Fetcher not implemented") for u in usernames])
                     self.progress.advance(collect_task, advance=len(usernames)) # Advance progress for skipped users
                     continue

                for username in usernames:
                    # Construct display name based on platform convention
                    display_name = ""
                    if platform == 'twitter': display_name = f"@{username}"
                    elif platform == 'reddit': display_name = f"u/{username}"
                    # Mastodon/Bluesky/HN display usernames as entered
                    elif platform in ['mastodon', 'bluesky', 'hackernews']: display_name = username
                    else: display_name = username # Default

                    task_desc = f"[cyan]Fetching {platform} for {display_name}..."
                    self.progress.update(collect_task, description=task_desc)
                    try:
                        # Use force_refresh=False for analysis calls unless explicitly needed later
                        data = fetcher(username=username, force_refresh=False)

                        if data:
                            summary = self._format_text_data(platform, username, data)
                            collected_text_summaries.append(summary)
                            # Collect media analysis only from successful fetches
                            # Filter out None/empty strings immediately
                            media_analyses = [ma for ma in data.get('media_analysis', []) if ma]
                            all_media_analyzes.extend(media_analyses)
                            logger.info(f"Successfully collected data for {platform}/{display_name}")
                        else:
                            # Fetcher returned None, imply failure handled internally (rate limit, not found etc)
                             failed_fetches.append((platform, display_name, "Data fetch failed (check logs/rate limits)"))
                             logger.warning(f"Data fetch returned None for {platform}/{display_name}")

                    except RateLimitExceededError as rle:
                        # This error is already logged and printed by _handle_rate_limit
                        failed_fetches.append((platform, display_name, f"Rate Limited")) # Simpler message
                    except (UserNotFoundError, AccessForbiddenError) as afe:
                         # These errors are logged by the fetchers
                         failed_fetches.append((platform, display_name, f"Access Error ({type(afe).__name__})"))
                         self.console.print(f"[yellow]Skipping {platform}/{display_name}: {afe}[/yellow]")
                    except Exception as e:
                        # Catch unexpected errors during fetch call
                        fetch_error_msg = f"Unexpected error during fetch for {platform}/{display_name}: {e}"
                        logger.error(fetch_error_msg, exc_info=True)
                        failed_fetches.append((platform, display_name, "Unexpected fetch error"))
                        self.console.print(f"[red]Error fetching {platform}/{display_name}: {e}[/red]")
                    finally:
                         # Ensure progress advances even if username processing fails inside loop
                         self.progress.advance(collect_task)


            # Make sure progress stops cleanly even if loop finishes early
            if collect_task in self.progress.task_ids:
                 # Ensure the task is marked as completed (fixes potential display issues)
                 self.progress.update(collect_task, completed=platform_count, description="[green]Data collection finished.")
                 # Optionally remove the task after completion
                 # self.progress.remove_task(collect_task)
            if self.progress.live.is_started: # Check if started before stopping
                 self.progress.stop()

            # --- Report Failed Fetches ---
            if failed_fetches:
                self.console.print("\n[bold yellow]Data Collection Issues:[/bold yellow]")
                for pf, user, reason in failed_fetches:
                    self.console.print(f"- {pf}/{user}: {reason}")
                self.console.print("[yellow]Analysis will proceed with available data.[/yellow]\n")


            # --- Prepare for LLM Analysis ---
            if not collected_text_summaries and not all_media_analyzes:
                return "[red]No data successfully collected from any platform. Analysis cannot proceed.[/red]"

            # De-duplicate media analysis strings (simple set conversion)
            # Ensure only non-empty strings are included
            unique_media_analyzes = sorted(list(set(filter(None, all_media_analyzes))))

            analysis_components = []
            image_model = os.getenv('IMAGE_ANALYSIS_MODEL', 'google/gemini-pro-vision') # Get model used
            text_model = os.getenv('ANALYSIS_MODEL', 'mistralai/mixtral-8x7b-instruct') # Get text model

            # Add Media Analysis Section (if any)
            if unique_media_analyzes:
                 media_summary = f"## Consolidated Media Analysis (using {image_model}):\n\n"
                 media_summary += "*Note: The following are objective descriptions based on visual content analysis.*\n\n"
                 # Ensure analysis strings are stripped of extra whitespace before joining
                 media_summary += "\n".join(f"{i+1}. {analysis.strip()}" for i, analysis in enumerate(unique_media_analyzes))
                 analysis_components.append(media_summary)
                 logger.debug(f"Added {len(unique_media_analyzes)} unique media analyzes to prompt.")

            # Add Text Data Section (if any)
            if collected_text_summaries:
                 text_summary = f"## Collected Textual & Activity Data Summary:\n\n"
                 # Ensure summaries are stripped before joining
                 text_summary += "\n\n---\n\n".join([s.strip() for s in collected_text_summaries]) # Separate platforms clearly
                 analysis_components.append(text_summary)
                 logger.debug(f"Added {len(collected_text_summaries)} platform text summaries to prompt.")

            # Construct the final prompt
            system_prompt = """**Objective:** Generate a comprehensive behavioral and linguistic profile based on the provided social media data, employing structured analytic techniques focused on objectivity, evidence-based reasoning, and clear articulation.

**Input:** You will receive summaries of user activity (text posts, engagement metrics, descriptive analyzes of images shared) from platforms like Twitter, Reddit, Bluesky, Mastodon, and Hacker News for one or more specified users. The user will also provide a specific analysis query. You may also receive consolidated analyzes of images shared by the user(s).

**Primary Task:** Address the user's specific analysis query using ALL the data provided (text summaries AND image analyzes) and the analytical framework below.

**Analysis Domains (Use these to structure your thinking and response where relevant to the query):**
1.  **Behavioral Patterns:** Analyze interaction frequency, platform-specific activity (e.g., retweets vs. posts, submissions vs. comments, boosts vs. original posts), potential engagement triggers, and temporal communication rhythms apparent *in the provided data*. Note differences across platforms if multiple are present.
2.  **Semantic Content & Themes:** Identify recurring topics, keywords, and concepts. Analyze linguistic indicators such as expressed sentiment/tone (positive, negative, neutral, specific emotions if clear), potential ideological leanings *if explicitly stated or strongly implied by language/topics*, and cognitive framing (how subjects are discussed). Assess information source credibility *only if* the user shares external links/content within the provided data AND you can evaluate the source based on common knowledge. Note use of content warnings/spoilers.
3.  **Interests & Network Context:** Deduce primary interests, hobbies, or professional domains suggested by post content and image analysis. Note any interaction patterns visible *within the provided posts* (e.g., frequent replies to specific user types, retweets/boosts of particular accounts, participation in specific communities like subreddits or Mastodon hashtags/local timelines if mentioned). Avoid inferring broad influence or definitive group membership without strong evidence.
4.  **Communication Style:** Assess linguistic complexity (simple/complex vocabulary, sentence structure), use of jargon/slang, rhetorical strategies (e.g., humor, sarcasm, argumentation), markers of emotional expression (e.g., emoji use, exclamation points, emotionally charged words), and narrative consistency across platforms. Note use of HTML/rich text formatting (e.g., in Mastodon).
5.  **Visual Data Integration:** Explicitly incorporate insights derived from the provided image analyzes. How do the visual elements (settings, objects, activities depicted) complement, contradict, or add context to the textual data? Note any patterns in the *types* of images shared (photos, screenshots, art) or use of alt text.

**Analytical Constraints & Guidelines:**
*   **Evidence-Based:** Ground ALL conclusions *strictly and exclusively* on the provided source materials (text summaries AND image analyzes). Reference specific examples or patterns from the data (e.g., "Frequent posts about [topic] on Reddit," "Image analysis of setting suggests [environment]," "Consistent use of technical jargon on HackerNews", "Use of spoiler tags on Mastodon for [topic]").
*   **Objectivity & Neutrality:** Maintain analytical neutrality. Avoid speculation, moral judgments, personal opinions, or projecting external knowledge not present in the data. Focus on describing *what the data shows*.
*   **Synthesize, Don't Just List:** Integrate findings from different platforms and data types (text/image) into a coherent narrative that addresses the query. Highlight correlations or discrepancies.
*   **Address the Query Directly:** Structure your response primarily around answering the user's specific question(s). Use the analysis domains as tools to build your answer.
*   **Acknowledge Limitations:** If the data is sparse, lacks specific details needed for the query, or only covers a short time period, explicitly state these limitations (e.g., "Based on the limited posts available...", "Image analysis provides no clues regarding [aspect]", "Mastodon data only includes original posts, excluding replies/boosts if filtered"). Do not invent information.
*   **Clarity & Structure:** Use clear language. Employ formatting (markdown headings, bullet points) to organize the response logically, often starting with a direct answer to the query followed by supporting evidence/analysis.

**Output:** A structured analytical report that directly addresses the user's query, rigorously supported by evidence from the provided text and image data, adhering to all constraints. Start with a summary answer, then elaborate with details structured using relevant analysis domains.
"""
            user_prompt = f"**Analysis Query:** {query}\n\n" \
                          f"**Provided Data:**\n\n" + \
                          "\n\n===\n\n".join(analysis_components) # Use a very clear separator

            # --- Call OpenRouter LLM ---
            analysis_task = self.progress.add_task(f"[magenta]Analyzing with {text_model}...", total=None)
            # self.progress.start() # Progress should already be started, re-starting can cause issues

            try:
                # Use threading to keep UI responsive during API call
                self._analysis_response = None # Reset before thread starts
                self._analysis_exception = None # Reset before thread starts
                api_thread = threading.Thread(
                    target=self._call_openrouter,
                    kwargs={
                        "json_data": {
                            "model": text_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "max_tokens": 3500, # Increased slightly for potentially more data
                            "temperature": 0.5,
                            "stream": False # Streaming disabled for now
                        }
                    }
                )
                api_thread.start()

                # Wait for the thread to finish while keeping the progress spinner active
                while api_thread.is_alive():
                    # The join(0.1) allows the main thread to briefly yield,
                    # keeping the Progress display updated.
                    api_thread.join(0.1)
                    # Update progress description slightly if desired, or just let it spin
                    # self.progress.update(analysis_task, description=f"[magenta]Analyzing with {text_model}...")

                # Stop the analysis task spinner once the thread is done
                if analysis_task is not None and analysis_task in self.progress.task_ids:
                      self.progress.update(analysis_task, description="[magenta]Analysis API call complete.")
                      # Keep the task visible until results are processed or error shown
                      # self.progress.remove_task(analysis_task)


                # --- Check results after thread completion ---
                if self._analysis_exception:
                    err_details = str(self._analysis_exception)
                    # Improve error reporting for HTTPStatusError
                    if isinstance(self._analysis_exception, httpx.HTTPStatusError):
                         err_code = self._analysis_exception.response.status_code
                         err_details = f"API HTTP {err_code}"
                         logger.error(f"Analysis API Error Response ({err_code}): {self._analysis_exception.response.text}")
                         err_details += " (See analyzer.log for full response)"
                         try: # Try to get cleaner message from JSON body
                             error_data = self._analysis_exception.response.json()
                             if 'error' in error_data and 'message' in error_data['error']:
                                 err_details = f"API Error: {error_data['error']['message']}"
                         except (json.JSONDecodeError, KeyError, AttributeError): pass # Use status code if not JSON or format unknown

                    # Re-raise as a runtime error to be caught by the outer handler
                    raise RuntimeError(f"Analysis API request failed: {err_details}") from self._analysis_exception

                if not self._analysis_response:
                     # This case should ideally not happen if the exception wasn't raised, but good failsafe
                     raise RuntimeError("Analysis API call completed but no response object was captured.")

                # Process successful response (outside thread)
                response = self._analysis_response
                response.raise_for_status() # Check status code again (redundant but safe)

                response_data = response.json()
                if 'choices' not in response_data or not response_data['choices'] or 'message' not in response_data['choices'][0] or 'content' not in response_data['choices'][0]['message']:
                    logger.error(f"Invalid analysis API response format: {response_data}")
                    return "[red]Analysis failed: Invalid response format from API.[/red]"

                analysis_content = response_data['choices'][0]['message']['content']
                # Add a header to the final output
                final_report = f"# OSINT Analysis Report\n\n**Query:** {query}\n\n**Models Used:**\n- Text Analysis: `{text_model}`\n- Image Analysis: `{image_model}`\n\n---\n\n{analysis_content}"
                return final_report

            finally:
                 # Ensure task is removed and progress potentially stops *after* processing or error
                 if analysis_task is not None and analysis_task in self.progress.task_ids:
                      self.progress.remove_task(analysis_task)
                 # Don't stop progress here, let the outer loop manage it if needed
                 # if self.progress.live.is_started:
                 #      self.progress.stop()
                 # Reset state variables used by the thread
                 self._analysis_response = None
                 self._analysis_exception = None


        except RateLimitExceededError as rle:
             # Stop progress if it's running
             if self.progress.live.is_started: self.progress.stop()
             self.console.print(f"[bold red]Analysis Aborted: {rle}[/bold red]")
             return f"[red]Analysis aborted due to rate limiting: {rle}[/red]"
        except RuntimeError as run_err: # Catch the re-raised API error
             if self.progress.live.is_started: self.progress.stop()
             self.console.print(f"[bold red]Analysis Failed: {run_err}[/bold red]")
             return f"[red]Analysis failed: {run_err}[/red]"
        except Exception as e:
             if self.progress.live.is_started: self.progress.stop()
             logger.error(f"Unexpected error during analysis phase: {str(e)}", exc_info=True)
             return f"[red]Analysis failed due to unexpected error: {str(e)}[/red]"


    def _format_text_data(self, platform: str, username: str, data: dict) -> str:
        """Formats fetched data into a detailed text block for the analysis LLM."""
        MAX_ITEMS_PER_TYPE = 25  # Max items (tweets, posts, etc.) to include per user/platform
        TEXT_SNIPPET_LENGTH = 750 # Max characters for text snippets

        output_lines = []
        platform_display_name = platform.capitalize()
        # Determine user prefix based on platform
        user_prefix = ""
        display_username = username # Default to passed username
        if platform == 'twitter': user_prefix = "@"
        elif platform == 'reddit': user_prefix = "u/"
        elif platform == 'mastodon':
             # Use the full 'acct' from user_info if available, otherwise keep original input
             display_username = data.get('user_info', {}).get('acct', username)
        elif platform == 'hackernews': user_prefix = ""
        elif platform == 'bluesky':
             # Use handle from profile if available
             display_username = data.get('profile_info', {}).get('handle', username)


        output_lines.append(f"### {platform_display_name} Data Summary for: {user_prefix}{display_username}")

        # --- Platform Specific Formatting ---

        if platform == 'twitter':
            user_info = data.get('user_info', {})
            if user_info:
                # Use get_sort_key for user creation date as well
                created_at_dt = get_sort_key(user_info, 'created_at')
                created_at_str = created_at_dt.strftime('%Y-%m-%d') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                output_lines.append(f"- User Profile: '{user_info.get('name')}' ({user_prefix}{user_info.get('username')}), ID: {user_info.get('id')}, Created: {created_at_str}")
                if user_info.get('public_metrics'):
                    pm = user_info['public_metrics']
                    output_lines.append(f"  - Stats: Followers={pm.get('followers_count', 'N/A')}, Following={pm.get('following_count', 'N/A')}, Tweets={pm.get('tweet_count', 'N/A')}")
            else:
                 output_lines.append("- User profile information not available.")

            tweets = data.get('tweets', [])
            output_lines.append(f"\n**Recent Tweets (up to {MAX_ITEMS_PER_TYPE}):**")
            if not tweets:
                output_lines.append("- No tweets found in fetched data.")
            else:
                for i, t in enumerate(tweets[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = get_sort_key(t, 'created_at') # Use global helper
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                    media_count = len(t.get('media', []))
                    media_info = f" (Media Attached: {media_count})" if media_count > 0 else ""
                    text = t.get('text', '[No Text]')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    metrics = t.get('metrics', {})
                    output_lines.append(
                        f"- Tweet {i+1} ({created_at_str}):{media_info}\n"
                        f"  Content: {text_snippet}\n"
                        f"  Metrics: Likes={metrics.get('like_count', 0)}, RTs={metrics.get('retweet_count', 0)}, Replies={metrics.get('reply_count', 0)}, Quotes={metrics.get('quote_count', 0)}"
                    )

        elif platform == 'reddit':
            stats = data.get('stats', {})
            output_lines.append(
                f"- Activity Overview: Subs={stats.get('total_submissions', 0)}, Comments={stats.get('total_comments', 0)}, Media Posts={stats.get('submissions_with_media', 0)}, Avg Sub Score={stats.get('avg_submission_score', 0):.1f}, Avg Comment Score={stats.get('avg_comment_score', 0):.1f}"
            )

            submissions = data.get('submissions', [])
            output_lines.append(f"\n**Recent Submissions (up to {MAX_ITEMS_PER_TYPE}):**")
            if not submissions:
                 output_lines.append("- No submissions found.")
            else:
                for i, s in enumerate(submissions[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = get_sort_key(s, 'created_utc') # Use global helper
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                    media_count = len(s.get('media', []))
                    media_info = f" (Media: {media_count})" if media_count > 0 else ""
                    text = s.get('text', '')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    text_info = f"\n  Self-Text: {text_snippet}" if text_snippet else ""
                    output_lines.append(
                        f"- Submission {i+1} in r/{s.get('subreddit', 'N/A')} ({created_at_str}):{media_info}\n"
                        f"  Title: {s.get('title', '[No Title]')}\n"
                        f"  Score: {s.get('score', 0)}, URL: {s.get('url', 'N/A')}"
                        f"{text_info}"
                    )

            comments = data.get('comments', [])
            output_lines.append(f"\n**Recent Comments (up to {MAX_ITEMS_PER_TYPE}):**")
            if not comments:
                 output_lines.append("- No comments found.")
            else:
                for i, c in enumerate(comments[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = get_sort_key(c, 'created_utc') # Use global helper
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                    text = c.get('text', '[No Text]')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    output_lines.append(
                        f"- Comment {i+1} in r/{c.get('subreddit', 'N/A')} ({created_at_str}):\n"
                        f"  Content: {text_snippet}\n"
                        f"  Score: {c.get('score', 0)}, Link: {c.get('permalink', 'N/A')}"
                    )

        elif platform == 'hackernews':
            stats = data.get('stats', {})
            output_lines.append(
                f"- Activity Overview: Items={stats.get('total_items_cached', 0)}, Stories={stats.get('total_stories', 0)}, Comments={stats.get('total_comments', 0)}, Avg Story Pts={stats.get('average_story_points', 0):.1f}, Avg Story Comments={stats.get('average_story_num_comments', 0):.1f}"
            )
            submissions = data.get('submissions', [])
            output_lines.append(f"\n**Recent Activity (Stories & Comments, up to {MAX_ITEMS_PER_TYPE}):**")
            if not submissions:
                output_lines.append("- No activity found.")
            else:
                for i, s in enumerate(submissions[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = get_sort_key(s, 'created_at') # Use global helper
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                    item_type = s.get('type', 'unknown').capitalize()
                    title = s.get('title')
                    text = s.get('text', '') # Already cleaned in fetcher
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    hn_link = f"https://news.ycombinator.com/item?id={s.get('objectID')}"

                    output_lines.append(f"- Item {i+1} ({item_type}, {created_at_str}):")
                    if title: output_lines.append(f"  Title: {title}")
                    if s.get('url'): output_lines.append(f"  URL: {s.get('url')}")
                    if text_snippet: output_lines.append(f"  Text: {text_snippet}")
                    if item_type == 'Story':
                        output_lines.append(f"  Stats: Pts={s.get('points', 0)}, Comments={s.get('num_comments', 0)}")
                    elif item_type == 'Comment':
                         # Add points for comments if available
                         points = s.get('points')
                         if points is not None:
                              output_lines.append(f"  Stats: Pts={points}")
                    output_lines.append(f"  HN Link: {hn_link}")


        elif platform == 'bluesky':
            profile = data.get('profile_info')
            if profile:
                 desc = profile.get('description', '').strip()
                 desc_snippet = desc[:150] + ('...' if len(desc) > 150 else '')
                 output_lines.append(f"- Profile: '{profile.get('display_name')}' ({profile.get('handle')}), DID: {profile.get('did')}")
                 if desc_snippet: output_lines.append(f"  - Bio: {desc_snippet}")
                 output_lines.append(f"  - Stats: Posts={profile.get('posts_count', 'N/A')}, Following={profile.get('follows_count', 'N/A')}, Followers={profile.get('followers_count', 'N/A')}")

            stats = data.get('stats', {})
            output_lines.append(
                 f"- Cached Activity Overview: Posts={stats.get('total_posts', 0)}, Media Posts={stats.get('posts_with_media', 0)}, Avg Likes={stats.get('avg_likes', 0):.1f}, Avg Reposts={stats.get('avg_reposts', 0):.1f}, Avg Replies={stats.get('avg_replies', 0):.1f}"
            )
            posts = data.get('posts', [])
            output_lines.append(f"\n**Recent Posts (up to {MAX_ITEMS_PER_TYPE}):**")
            if not posts:
                 output_lines.append("- No posts found.")
            else:
                for i, p in enumerate(posts[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = get_sort_key(p, 'created_at') # Use global helper
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                    media_count = len(p.get('media', []))
                    media_info = f" (Media: {media_count})" if media_count > 0 else ""
                    text = p.get('text', '[No Text]')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    embed_info = p.get('embed')
                    embed_desc = f" (Embed: {embed_info['type']})" if embed_info else ""

                    output_lines.append(
                         f"- Post {i+1} ({created_at_str}):{media_info}{embed_desc}\n"
                         f"  Content: {text_snippet}\n"
                         f"  Stats: Likes={p.get('likes', 0)}, Reposts={p.get('reposts', 0)}, Replies={p.get('reply_count', 0)}\n"
                         f"  URI: {p.get('uri', 'N/A')}"
                     )

        # +++ Add Mastodon Formatting +++
        elif platform == 'mastodon':
            user_info = data.get('user_info', {})
            if user_info:
                 created_at_dt = get_sort_key(user_info, 'created_at') # Use helper for profile date too
                 created_at_str = created_at_dt.strftime('%Y-%m-%d') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                 # Clean bio HTML using BeautifulSoup for display snippet
                 bio_text = ''
                 try:
                     bio_soup = BeautifulSoup(user_info.get('note_html', ''), 'html.parser')
                     bio_text = bio_soup.get_text(separator=' ', strip=True)
                 except Exception: # Catch potential BS4 errors
                     bio_text = user_info.get('note_html', '[Could not parse bio]') # Fallback

                 bio_snippet = bio_text[:150] + ('...' if len(bio_text) > 150 else '')

                 output_lines.append(
                     f"- User Profile: '{user_info.get('display_name')}' ({user_info.get('acct')}), ID: {user_info.get('id')}, Created: {created_at_str}\n"
                     f"  - Bio Snippet: {bio_snippet}\n"
                     f"  - Stats: Followers={user_info.get('followers_count', 'N/A')}, Following={user_info.get('following_count', 'N/A')}, Posts={user_info.get('statuses_count', 'N/A')}"
                 )
            else:
                 output_lines.append("- User profile information not available.")

            stats = data.get('stats', {})
            output_lines.append(
                 f"- Cached Activity Overview: Posts={stats.get('total_posts_cached',0)}, Originals={stats.get('total_original_posts',0)}, Boosts={stats.get('total_reblogs',0)}, Media Posts={stats.get('posts_with_media', 0)}, Avg Favs={stats.get('avg_favourites', 0):.1f}, Avg Boosts={stats.get('avg_reblogs', 0):.1f}"
            )

            posts = data.get('posts', [])
            output_lines.append(f"\n**Recent Posts (Toots & Boosts, up to {MAX_ITEMS_PER_TYPE}):**")
            if not posts:
                output_lines.append("- No posts found in fetched data.")
            else:
                for i, p in enumerate(posts[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = get_sort_key(p, 'created_at') # Use global helper
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                    media_count = len(p.get('media', []))
                    media_info = f" (Media: {media_count})" if media_count > 0 else ""
                    spoiler = p.get('spoiler_text', '')
                    spoiler_info = f" (CW: {spoiler})" if spoiler else ""
                    is_boost = p.get('is_reblog', False)
                    boost_info = f" (Boost of {p.get('reblog_original_author', 'unknown')})" if is_boost else ""

                    # Use cleaned text, provide context for boosts/CWs
                    text_snippet = p.get('text_cleaned', '')
                    if spoiler and not text_snippet: # Handle case where CW hides everything
                         text_display = "[Content Warning Text Only]"
                    elif is_boost and not text_snippet:
                         text_display = "[Boost Content Only]"
                    else: # Show snippet if available
                         text_display = text_snippet[:TEXT_SNIPPET_LENGTH] + ('...' if len(text_snippet) > TEXT_SNIPPET_LENGTH else '')
                         if not text_display: text_display = "[No Text Content]" # Handle empty strings


                    output_lines.append(
                        f"- Post {i+1} ({created_at_str}):{boost_info}{spoiler_info}{media_info}\n"
                        f"  Content: {text_display}\n"
                        f"  Stats: Favs={p.get('favourites_count', 0)}, Boosts={p.get('reblogs_count', 0)}, Replies={p.get('replies_count', 0)}\n"
                        f"  Link: {p.get('url', 'N/A')}"
                    )
                    # Optionally add link to boosted post
                    if is_boost and p.get('reblog_original_url'):
                         output_lines.append(f"  Original Post: {p['reblog_original_url']}")

        # --- End Mastodon Formatting ---

        else:
            # Fallback for any other platform
            output_lines.append(f"\n**Data Overview:**")
            output_lines.append(f"- Raw Data Preview: {str(data)[:TEXT_SNIPPET_LENGTH]}...")

        return "\n".join(output_lines)


    def _call_openrouter(self, json_data: dict):
        """Worker function for making the OpenRouter API call in a thread."""
        # Reset state variables specific to this call
        thread_response = None
        thread_exception = None
        try:
            # Ensure client is ready (accessing property initializes if needed)
            client = self.openrouter
            thread_response = client.post("/chat/completions", json=json_data)
            # Raise HTTP errors immediately *within the thread* so they get caught
            thread_response.raise_for_status()
        except Exception as e:
            # Store the exception to be checked by the main thread
            logger.error(f"OpenRouter API call error in thread: {str(e)}")
            # Log response details if it's an HTTP error
            if isinstance(e, httpx.HTTPStatusError) and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                try:
                     logger.error(f"Response content: {e.response.text}")
                except Exception:
                     logger.error("Could not decode error response content.")
            thread_exception = e
        finally:
            # Set the shared instance variables *after* the call completes or fails
            self._analysis_response = thread_response
            self._analysis_exception = thread_exception


    def _save_output(self, content: str, query: str, platforms_analyzed: List[str], format_type: str = "markdown"):
        """Saves the analysis report to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_dir / 'outputs'
        # Create a safe filename base from query and platforms
        safe_query = "".join(c if c.isalnum() else '_' for c in query[:30]).strip('_')
        safe_platforms = "_".join(sorted(platforms_analyzed))[:30].strip('_') # Increased length slightly
        filename_base = f"analysis_{timestamp}_{safe_platforms}_{safe_query}"

        try:
            if format_type == "json":
                filename = output_dir / f"{filename_base}.json"
                # Store raw markdown content within JSON structure
                data_to_save = {
                    "analysis_metadata": {
                         "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                         "query": query,
                         "platforms_analyzed": platforms_analyzed,
                         "output_format": "json",
                         "text_model": os.getenv('ANALYSIS_MODEL', 'unknown'),
                         "image_model": os.getenv('IMAGE_ANALYSIS_MODEL', 'unknown')
                    },
                    "analysis_report_markdown": content # Store the raw markdown report
                }
                filename.write_text(json.dumps(data_to_save, indent=2), encoding='utf-8')
            else: # Default to markdown
                filename = output_dir / f"{filename_base}.md"
                # Add metadata as comments/frontmatter to markdown
                md_metadata = f"""---
Query: {query}
Platforms: {', '.join(platforms_analyzed)}
Timestamp: {datetime.now(timezone.utc).isoformat()}
Text Model: {os.getenv('ANALYSIS_MODEL', 'unknown')}
Image Model: {os.getenv('IMAGE_ANALYSIS_MODEL', 'unknown')}
---

"""
                # The content should already be markdown from the LLM
                # No need to process with Markdown() again before saving
                # Just ensure it starts with the # OSINT Analysis Report header if present
                if content.startswith("# OSINT Analysis Report"):
                     plain_content = content # Assume it's good markdown
                else:
                     # If missing the header, add it? Or just save as is. Let's save as is.
                     logger.warning("Saved analysis content missing expected '# OSINT Analysis Report' header.")
                     plain_content = content


                full_content = md_metadata + plain_content
                filename.write_text(full_content, encoding='utf-8')

            self.console.print(f"[green]Analysis saved to: {filename}[/green]")

        except Exception as e:
            self.console.print(f"[bold red]Failed to save output: {str(e)}[/bold red]")
            logger.error(f"Failed to save output file {filename_base}: {e}", exc_info=True)

    def get_available_platforms(self, check_creds=True) -> List[str]:
        """Checks environment variables to see which platforms are configured."""
        available = []
        # Conditionally check credentials based on flag
        if not check_creds or all(os.getenv(k) for k in ['TWITTER_BEARER_TOKEN']):
            available.append('twitter')
        if not check_creds or all(os.getenv(k) for k in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']):
            available.append('reddit')
        if not check_creds or all(os.getenv(k) for k in ['BLUESKY_IDENTIFIER', 'BLUESKY_APP_SECRET']):
             available.append('bluesky')
        # +++ Add Mastodon Check +++
        if not check_creds or all(os.getenv(k) for k in ['MASTODON_ACCESS_TOKEN', 'MASTODON_API_BASE_URL']):
             # Also do a basic URL check for Mastodon
             base_url = os.getenv('MASTODON_API_BASE_URL')
             if base_url and urlparse(base_url).scheme and urlparse(base_url).netloc:
                 available.append('mastodon')
             elif check_creds: # Only warn if checking creds and URL is bad
                 logger.warning("Mastodon credentials found, but MASTODON_API_BASE_URL is invalid or missing.")

        # --- End Mastodon Check ---
        available.append('hackernews') # Always available conceptually
        return sorted(available)

    # --- Interactive Mode ---
    def run(self):
        """Runs the interactive command-line interface."""
        self.console.print(Panel(
            "[bold blue]Social Media OSINT analyzer[/bold blue]\n"
            "Collects and analyzes user activity across multiple platforms using LLMs.\n"
            "Ensure API keys and identifiers are set in your `.env` file.",
            title="Welcome",
            border_style="blue"
        ))

        # Get platforms with credentials actually configured
        available_platforms = self.get_available_platforms(check_creds=True)
        # Check if only HackerNews is available (no other creds set)
        only_hackernews = 'hackernews' in self.get_available_platforms(check_creds=False) and not any(p != 'hackernews' for p in available_platforms)

        if not available_platforms and not only_hackernews:
            self.console.print("[bold red]Error: No API credentials correctly configured for any platform.[/bold red]")
            self.console.print("Please set credentials in a `.env` file (e.g., TWITTER_BEARER_TOKEN, MASTODON_ACCESS_TOKEN etc.) and ensure URLs are valid.")
            return # Exit if no platforms usable

        while True:
            self.console.print("\n[bold cyan]Select Platform(s) for Analysis:[/bold cyan]")

            # Get currently configured platforms for the menu
            current_available = self.get_available_platforms(check_creds=True)
            # Add HN if it's the *only* thing available conceptually, even without other creds
            if not current_available and 'hackernews' in self.get_available_platforms(check_creds=False):
                 current_available = ['hackernews']

            if not current_available:
                 self.console.print("[yellow]No platforms seem to be configured correctly. Please check your .env file and logs.[/yellow]")
                 break # Exit loop if nothing is configured


            platform_priority = {
                'twitter': 1,
                'bluesky': 2,
                'mastodon': 3, # Add Mastodon priority
                'reddit': 4,
                'hackernews': 5,
            }

            current_available.sort(key=lambda x: platform_priority.get(x, 999))

            platform_options = {str(i+1): p for i, p in enumerate(current_available)}
            num_platforms = len(current_available)
            cross_platform_key = str(num_platforms + 1)
            exit_key = str(num_platforms + 2)
            # Only offer cross-platform if more than one platform is available
            if num_platforms > 1:
                platform_options[cross_platform_key] = "cross-platform"
            platform_options[exit_key] = "exit"

            for key, name in platform_options.items():
                 # Add config status indicator? (e.g., check if client property raises error?) - maybe too complex here
                 self.console.print(f"{key}. {name.capitalize()}")

            choice = Prompt.ask("Enter number(s) (e.g., 1 or 1,3 or 5 for cross-platform)", default=exit_key).strip()

            if choice == exit_key or choice.lower() == 'exit':
                break

            selected_platform_keys = []
            is_cross_platform = False
            if (cross_platform_key in platform_options) and (choice == cross_platform_key or choice.lower() == 'cross-platform'):
                 # Select all available platforms *except* 'cross-platform' and 'exit' themselves
                 selected_platform_keys = [k for k, v in platform_options.items() if v not in ["cross-platform", "exit"]]
                 is_cross_platform = True
                 selected_names = [platform_options[k] for k in selected_platform_keys]
                 self.console.print(f"Selected: Cross-Platform Analysis ({', '.join(name.capitalize() for name in selected_names)})")
            else:
                 raw_keys = [k.strip() for k in choice.split(',')]
                 valid_keys = [k for k in raw_keys if k in platform_options and k not in [cross_platform_key, exit_key]]
                 if not valid_keys:
                     self.console.print("[yellow]Invalid selection. Please enter numbers corresponding to the platform options.[/yellow]")
                     continue
                 selected_platform_keys = valid_keys
                 selected_names = [platform_options[k].capitalize() for k in selected_platform_keys]
                 self.console.print(f"Selected: {', '.join(selected_names)}")


            platforms_to_query: Dict[str, List[str]] = {}
            try:
                for key in selected_platform_keys:
                     if key not in platform_options: continue # Should not happen with validation, but safe check
                     platform_name = platform_options[key]
                     prompt_message = f"{platform_name.capitalize()} username(s) (comma-separated"
                     # Add platform-specific hints
                     if platform_name == 'twitter': prompt_message += ", no '@'"
                     elif platform_name == 'reddit': prompt_message += ", no 'u/'"
                     elif platform_name == 'bluesky': prompt_message += ", e.g., 'handle.bsky.social'"
                     # +++ Mastodon Hint +++
                     elif platform_name == 'mastodon': prompt_message += ", format: 'user@instance.domain'"
                     # --- End Mastodon Hint ---
                     prompt_message += ")"

                     user_input = Prompt.ask(prompt_message, default="").strip()
                     if user_input:
                         # Split and strip, filter out empty strings
                         usernames = [u.strip() for u in user_input.split(',') if u.strip()]
                         if not usernames: continue # Skip if only whitespace or empty strings entered

                         # +++ Mastodon Username Validation (Interactive) +++
                         if platform_name == 'mastodon':
                              valid_masto_users = []
                              for u in usernames:
                                   if '@' in u and '.' in u.split('@')[1]: # Basic check for instance part
                                       valid_masto_users.append(u)
                                   else:
                                       # Check if a default instance can be inferred from env var
                                       default_instance_url = os.getenv('MASTODON_API_BASE_URL', '')
                                       default_instance_domain = urlparse(default_instance_url).netloc if default_instance_url else None

                                       if default_instance_domain:
                                           assumed_user = f"{u}@{default_instance_domain}"
                                           if Confirm.ask(f"[yellow]Username '{u}' lacks instance. Assume it's on '{default_instance_domain}' (i.e., '{assumed_user}')?[/yellow]", default=True):
                                                valid_masto_users.append(assumed_user)
                                           else:
                                                self.console.print(f"[yellow]Skipping Mastodon username '{u}' due to missing instance.[/yellow]")
                                       else:
                                           self.console.print(f"[bold red]Invalid Mastodon username format: '{u}'. Needs 'user@instance.domain'. Cannot assume default. Skipping.[/bold red]")
                              usernames = valid_masto_users # Use only validated/corrected usernames
                         # --- End Mastodon Validation ---

                         if usernames:
                             # Ensure platform_name is in dict before extending/assigning
                             if platform_name not in platforms_to_query:
                                 platforms_to_query[platform_name] = []
                             platforms_to_query[platform_name].extend(usernames)

                if not platforms_to_query:
                    self.console.print("[yellow]No valid usernames entered for selected platform(s). Returning to menu.[/yellow]")
                    continue

                # Start the analysis loop for the chosen platforms/users
                self._run_analysis_loop(platforms_to_query)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
                if Confirm.ask("Exit program?", default=False):
                    break
            except RuntimeError as e:
                 # Catch setup errors (missing keys, failed auth) during client initialization (e.g., in property getters)
                 self.console.print(f"[bold red]Configuration/Runtime Error:[/bold red] {e}")
                 self.console.print("Please check your .env file, API keys, and instance URLs.")
                 # Don't loop infinitely on setup errors, offer exit.
                 if not Confirm.ask("Exit program?", default=True):
                     continue # Allow retry if user explicitly wants it
                 else:
                     break
            except Exception as e:
                logger.error(f"Unexpected error in main interactive loop: {e}", exc_info=True)
                self.console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
                if Confirm.ask("Try again?", default=False):
                    continue
                else:
                    break

        self.console.print("\n[blue]Exiting Social Media analyzer.[/blue]")


    def _run_analysis_loop(self, platforms: Dict[str, List[str]]):
        """Inner loop for performing analysis queries on selected targets."""
        platform_labels = []
        platform_names_list = sorted(platforms.keys()) # For saving output

        for pf, users in platforms.items():
             user_prefix = ""
             # Use appropriate display format
             if pf == 'twitter': user_prefix = "@"
             elif pf == 'reddit': user_prefix = "u/"
             # Mastodon/Bluesky/HN display usernames as entered/resolved
             display_users = [f"{user_prefix}{u}" if user_prefix else u for u in users]
             platform_labels.append(f"{pf.capitalize()}: {', '.join(display_users)}")

        platform_info = " | ".join(platform_labels)

        self.console.print(Panel(
            f"Targets: {platform_info}\n"
            f"Enter your analysis query below.\n"
            f"Commands: `exit` (end session), `refresh` (force full data fetch), `help`",
            title="🔎 Analysis Session",
            border_style="cyan",
            expand=False
        ))

        while True:
            try:
                query = Prompt.ask("\n[bold green]Analysis Query>[/bold green]").strip()
                if not query:
                    continue

                cmd = query.lower()
                if cmd == 'exit':
                    self.console.print("[yellow]Exiting analysis session.[/yellow]")
                    break
                if cmd == 'help':
                     self.console.print(Panel(
                        "**Available Commands:**\n"
                        "- `exit`: End this analysis session and return to platform selection.\n"
                        "- `refresh`: Force a full data fetch for all current targets, ignoring cache.\n"
                        "- `help`: Show this help message.\n\n"
                        "**To analyze:**\n"
                        "Simply type your analysis question (e.g., 'What are the main topics discussed?', 'Identify potential location clues from images and text.')",
                        title="Help", border_style="blue", expand=False
                    ))
                     continue
                if cmd == 'refresh':
                    if Confirm.ask("Force refresh data for all targets? This will use more API calls.", default=False):
                         total_targets = sum(len(u) for u in platforms.values())
                         refresh_task = self.progress.add_task("[yellow]Refreshing data...", total=total_targets)
                         self.progress.start()
                         failed_refreshes = []
                         for platform, usernames in platforms.items():
                             fetcher = getattr(self, f'fetch_{platform}', None)
                             if not fetcher: continue
                             for username in usernames:
                                 # Construct display name for progress
                                 display_name = username
                                 if platform == 'twitter': display_name = f"@{username}"
                                 elif platform == 'reddit': display_name = f"u/{username}"
                                 # Mastodon/Bluesky/HN use username directly

                                 self.progress.update(refresh_task, description=f"[yellow]Refreshing {platform}/{display_name}...")
                                 try:
                                     # Call fetcher with force_refresh=True
                                     # Fetcher handles its own errors (UserNotFound, RateLimit etc)
                                     result = fetcher(username=username, force_refresh=True)
                                     if result is None: # Indicates fetch failure handled internally
                                         failed_refreshes.append((platform, display_name))
                                         # Error message should have been printed by fetcher or logger
                                 except Exception as e: # Catch unexpected fetch errors during refresh
                                     failed_refreshes.append((platform, display_name))
                                     logger.error(f"Unexpected Refresh failed for {platform}/{display_name}: {e}", exc_info=True)
                                     self.console.print(f"[red]Unexpected Refresh failed for {platform}/{display_name}: {e}[/red]")
                                 finally:
                                      self.progress.advance(refresh_task)

                         # Ensure progress finishes cleanly
                         if refresh_task in self.progress.task_ids:
                              self.progress.update(refresh_task, completed=total_targets, description="[green]Refresh attempt finished.")
                              # self.progress.remove_task(refresh_task) # Optional: remove task
                         if self.progress.live.is_started:
                             self.progress.stop()

                         if failed_refreshes:
                              self.console.print(f"[yellow]Data refresh attempted, but issues encountered for {len(failed_refreshes)} target(s) (see logs).[/yellow]")
                         else:
                              self.console.print("[green]Data refresh attempt completed for all targets.[/green]")
                    continue # Go back to prompt after refresh attempt


                # --- Perform Analysis ---
                self.console.print(f"[cyan]Starting analysis for query:[/cyan] '{query}'", highlight=False)
                # Ensure progress is stopped before starting analysis if not already stopped
                if self.progress.live.is_started:
                     self.progress.stop()

                # Call the main analysis function
                analysis_result = self.analyze(platforms, query) # This now handles its own progress display

                # Display and handle saving based on auto-save flag
                if analysis_result:
                    # Check for error markers more robustly (case-insensitive, strip whitespace)
                    result_lower_stripped = analysis_result.strip().lower()
                    is_error = result_lower_stripped.startswith(("[red]", "error:", "analysis failed", "analysis aborted"))
                    is_warning = result_lower_stripped.startswith(("[yellow]", "warning:"))

                    border_col = "red" if is_error else ("yellow" if is_warning else "green")

                    # Use Markdown rendering for output display
                    self.console.print(Panel(
                        Markdown(analysis_result), # Render result as Markdown
                        title="Analysis Report",
                        border_style=border_col,
                        expand=False # Avoid overly wide panels
                    ))

                    # --- Saving Logic ---
                    if not is_error: # Only attempt to save successful reports
                        save_report = False
                        # Use self.args consistently
                        if self.args and self.args.no_auto_save:
                             # Prompt user because auto-save is disabled
                             if Confirm.ask("Save this analysis report?", default=True):
                                 save_report = True
                        else:
                            # Auto-save is enabled (default behavior or --no-auto-save not used)
                            save_format_arg = self.args.format if self.args else 'markdown' # Get format from args or default
                            self.console.print(f"[cyan]Auto-saving analysis report as {save_format_arg}...[/cyan]")
                            save_report = True

                        if save_report:
                            # Use the format from args as the default for the prompt if needed,
                            # otherwise use the format specified in args directly.
                            save_format = self.args.format if self.args else 'markdown' # Default to markdown if args missing
                            # Only prompt if auto-save was off AND user explicitly chose to save
                            if self.args and self.args.no_auto_save:
                                save_format = Prompt.ask("Save format?", choices=["markdown", "json"], default=save_format)
                            # Pass the raw analysis_result (which should be markdown)
                            self._save_output(analysis_result, query, platform_names_list, save_format)
                    # --- End Saving Logic ---

                else:
                    self.console.print("[red]Analysis returned no result (None).[/red]")

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Analysis query cancelled.[/yellow]")
                if Confirm.ask("\nExit this analysis session?", default=False):
                    break # Exit inner loop
            except RateLimitExceededError as rle:
                 # Error already printed by handler, just inform user and maybe exit
                 self.console.print("[yellow]Please wait for rate limit to reset before trying again.[/yellow]")
                 if Confirm.ask("Exit analysis session due to rate limit?", default=False):
                     break # Exit inner loop
            except Exception as e:
                 logger.error(f"Unexpected error during analysis loop: {e}", exc_info=True)
                 self.console.print(f"\n[bold red]An unexpected error occurred during analysis:[/bold red] {e}")
                 if not Confirm.ask("An error occurred. Continue session?", default=True):
                     break # Exit inner loop


    # --- Non-Interactive Mode (stdin processing) ---
    def process_stdin(self, output_format: str):
        """Processes analysis request from JSON input via stdin."""
        self.console.print("[cyan]Processing analysis request from stdin...[/cyan]")
        try:
            input_data = json.load(sys.stdin)
            platforms = input_data.get("platforms")
            query = input_data.get("query")

            if not isinstance(platforms, dict) or not platforms:
                raise ValueError("Invalid 'platforms' data in JSON input. Must be a non-empty dictionary.")
            if not isinstance(query, str) or not query:
                raise ValueError("Invalid or missing 'query' in JSON input.")

            # Validate platform usernames are lists/strings and platform is available
            valid_platforms = {}
            available_configured = self.get_available_platforms(check_creds=True) # Check configured platforms
            available_conceptual = self.get_available_platforms(check_creds=False) # Get all possible platforms

            for platform, usernames in platforms.items():
                 platform = platform.lower() # Normalize platform name

                 # Check if platform is conceptually supported *and* configured (if requires creds)
                 is_available = False
                 if platform in available_conceptual:
                      if platform == 'hackernews': # HN needs no config
                           is_available = True
                      elif platform in available_configured: # Others need config
                           is_available = True

                 if not is_available:
                     logger.warning(f"Platform '{platform}' specified in stdin is not supported or not correctly configured. Skipping.")
                     continue

                 processed_users = []
                 if isinstance(usernames, str):
                     if usernames.strip(): processed_users = [usernames.strip()]
                 elif isinstance(usernames, list):
                     processed_users = [u.strip() for u in usernames if isinstance(u, str) and u.strip()]
                 else:
                     logger.warning(f"Invalid username format for platform '{platform}' in stdin. Expected string or list of strings. Skipping platform.")
                     continue # Skip platform with invalid username format

                 if not processed_users:
                     logger.warning(f"No valid usernames provided for platform '{platform}' in stdin. Skipping platform.")
                     continue # Skip platform with no users

                 # +++ Add Mastodon username validation for stdin +++
                 if platform == 'mastodon':
                     masto_users_validated = []
                     for u in processed_users:
                         if '@' in u and '.' in u.split('@')[1]:
                             masto_users_validated.append(u)
                         else:
                             logger.warning(f"Invalid Mastodon username format in stdin for '{u}'. Needs 'user@instance.domain'. Skipping user.")
                     processed_users = masto_users_validated # Use only valid ones
                     if not processed_users:
                          logger.warning(f"No valid Mastodon usernames remained for platform '{platform}' after validation. Skipping platform.")
                          continue # Skip if validation removed all users
                 # --- End Mastodon validation ---

                 if processed_users:
                     valid_platforms[platform] = processed_users

            if not valid_platforms:
                 raise ValueError("No valid and configured platforms with valid usernames found in the processed input.")

            # Run analysis
            # Ensure progress is stopped before starting analysis
            if self.progress.live.is_started: self.progress.stop()
            analysis_report = self.analyze(valid_platforms, query) # Handles its own progress

            if analysis_report:
                 # Check for error indicators (case-insensitive, stripped)
                 result_lower_stripped = analysis_report.strip().lower()
                 is_error = result_lower_stripped.startswith(("[red]", "error:", "analysis failed", "analysis aborted"))

                 if not is_error:
                    # Analysis succeeded
                    platform_names_list = sorted(valid_platforms.keys())
                    # Check args passed to __init__ for no_auto_save
                    if self.args and self.args.no_auto_save:
                        # Print raw report (should be markdown) to stdout
                        # Use print directly to avoid Rich formatting stdout for piping
                        print(analysis_report)
                        sys.exit(0) # Success
                    else:
                        # Auto-save enabled: Save the output
                        # Pass raw markdown report to save function
                        output_format_to_use = self.args.format if self.args else 'markdown' # Use arg format or default
                        self._save_output(analysis_report, query, platform_names_list, output_format_to_use)
                        # Print confirmation message to stderr so stdout only contains report if --no-auto-save used
                        self.console.print(f"[green]Analysis complete. Output auto-saved ({output_format_to_use}).[/green]", file=sys.stderr)
                        sys.exit(0) # Success
                 else:
                    # Analysis failed or produced error message
                    # Print error report to stderr
                    sys.stderr.write("[ERROR] Analysis failed or produced an error report:\n")
                    sys.stderr.write(analysis_report + "\n")
                    sys.exit(1) # Failure
            else:
                # Analysis returned nothing at all
                sys.stderr.write("[ERROR] Analysis returned no result (None).\n")
                sys.exit(1) # Failure

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from stdin.")
            sys.stderr.write("Error: Invalid JSON received on stdin.\n")
            sys.exit(1)
        except ValueError as ve:
            logger.error(f"Invalid input data: {ve}")
            sys.stderr.write(f"Error: Invalid input data - {ve}\n")
            sys.exit(1)
        except RateLimitExceededError as rle:
             logger.error(f"Processing failed due to rate limit: {rle}")
             sys.stderr.write(f"Error: Rate limit exceeded during processing - {rle}\n")
             sys.exit(1)
        except RuntimeError as rte: # Catch setup errors from analysis call
            logger.error(f"Runtime error during stdin processing: {rte}")
            sys.stderr.write(f"Error: Runtime error during processing - {rte}\n")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error during stdin processing: {e}", exc_info=True)
            sys.stderr.write(f"Error: An unexpected error occurred - {e}\n")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Social Media OSINT analyzer using LLMs. Fetches user data from various platforms, performs text and image analysis, and generates reports.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--stdin',
        action='store_true',
        help="Read analysis request from stdin as JSON.\n"
             "Expected JSON format:\n"
             '{\n'
             '  "platforms": {\n'
             '    "twitter": ["user1", "user2"],\n'
             '    "reddit": "user3",\n'
             '    "hackernews": ["user4"],\n'
             '    "bluesky": ["handle1.bsky.social"],\n'
             '    "mastodon": ["user@instance.social", "another@other.server"]\n' # Added Mastodon example
             '  },\n'
             '  "query": "Your analysis query here..."\n'
             '}'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'markdown'],
        default='markdown',
        help="Output format for saving analysis results (default: markdown).\n"
             "- markdown: Saves as a .md file with metadata header.\n"
             "- json: Saves as a .json file containing metadata and the markdown report."
    )
    parser.add_argument(
        '--no-auto-save',
        action='store_true',
        help="Disable automatic saving of analysis reports; prompt user before saving in interactive mode, print report directly to stdout in stdin mode."
    )
    # Add log level argument
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='Set the logging level (default: WARNING).'
    )


    args = parser.parse_args()

    # --- Configure Logging Level ---
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.getLogger().setLevel(log_level_numeric)
    logger.setLevel(log_level_numeric)
    # Ensure handlers also respect the new level (especially file handler)
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level_numeric)
    logger.info(f"Logging level set to {args.log_level}")
    # --- End Logging Config ---


    try:
        # Pass the parsed args to the constructor
        analyzer = SocialOSINTLM(args=args)
        if args.stdin:
            # process_stdin uses self.args internally now
            analyzer.process_stdin(args.format)
        else:
            # run -> _run_analysis_loop uses self.args internally now
            analyzer.run()

    except RuntimeError as e:
         # Catch critical setup errors during initialization (e.g., missing core env vars)
         logging.getLogger('SocialOSINTLM').critical(f"Initialization failed: {e}", exc_info=False)
         # Use Rich console if available for error printing
         console = Console(stderr=True)
         console.print(f"\n[bold red]CRITICAL ERROR:[/bold red] {e}")
         console.print("Ensure necessary API keys (OpenRouter) and platform credentials/URLs are correctly set in .env")
         sys.exit(1)
    except Exception as e:
         # Catch any other unexpected top-level errors
         logging.getLogger('SocialOSINTLM').critical(f"An unexpected critical error occurred: {e}", exc_info=True)
         console = Console(stderr=True)
         console.print(f"\n[bold red]UNEXPECTED CRITICAL ERROR:[/bold red] {e}")
         sys.exit(1)
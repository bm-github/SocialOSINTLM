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
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.gif'] # Consolidated list

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
            dt_obj = datetime.fromisoformat(dt_val)
            # Ensure timezone for comparison if needed (make it UTC if naive)
            return dt_obj if dt_obj.tzinfo else dt_obj.replace(tzinfo=timezone.utc)
        except ValueError: # Handle cases where conversion might fail
            logger.warning(f"Could not parse datetime string: {dt_val}")
            return datetime.min.replace(tzinfo=timezone.utc)
    elif isinstance(dt_val, datetime):
         # Ensure timezone for comparison if needed (make it UTC if naive)
         return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    # Fallback for missing/invalid keys or other types (like timestamps)
    elif isinstance(dt_val, (int, float)):
         try:
             # Attempt to treat as UNIX timestamp
             dt_obj = datetime.fromtimestamp(dt_val, tz=timezone.utc)
             return dt_obj
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
        available_no_creds = self.get_available_platforms(check_creds=False)
        if not platforms_configured and 'hackernews' not in available_no_creds:
             logger.warning("No platform API credentials found in environment variables. Only HackerNews might work.")
        elif not platforms_configured and 'hackernews' in available_no_creds:
            pass # HackerNews alone is okay

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
                # Attempt login, handle specific login failure
                try:
                    login_response = client.login(
                        os.environ['BLUESKY_IDENTIFIER'],
                        os.environ['BLUESKY_APP_SECRET']
                    )
                    # login_response doesn't explicitly signal failure other than exception AFAIK
                    logger.debug(f"Bluesky login successful for handle: {login_response.handle}")
                except atproto_exceptions.AtProtocolError as login_err:
                     logger.error(f"Bluesky login failed: {login_err}")
                     # Provide more specific error based on common login issues
                     if 'invalid identifier or password' in str(login_err).lower():
                         raise RuntimeError("Bluesky login failed: Invalid identifier or password.")
                     else:
                         raise RuntimeError(f"Bluesky login failed: {login_err}")

                self._bluesky_client = client

            except (KeyError, RuntimeError) as e: # Use alias
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
                try:
                     instance_info = client.instance()
                     logger.debug(f"Connected to Mastodon instance: {instance_info['title']} ({instance_info['uri']})")
                except MastodonError as instance_err:
                     logger.error(f"Failed to connect to Mastodon instance {base_url}: {instance_err}")
                     # Improve error message based on common connection issues
                     if 'unauthorized' in str(instance_err).lower() or '401' in str(instance_err):
                          raise RuntimeError(f"Mastodon connection failed: Invalid Access Token for {base_url}.")
                     elif 'not found' in str(instance_err).lower() or '404' in str(instance_err):
                          raise RuntimeError(f"Mastodon connection failed: API base URL {base_url} seems incorrect (Not Found).")
                     else:
                          raise RuntimeError(f"Mastodon connection failed for {base_url}: {instance_err}")

                self._mastodon_client = client
                logger.debug(f"Mastodon client initialized for {base_url}.")
            except (KeyError, RuntimeError) as e:
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
                # self.console.print(f"DEBUG: OpenRouter Key read from env: {'Yes' if api_key else 'No'}")
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
                # Optional: Add a test call here if needed, e.g., fetch models list
                # try:
                #     self._openrouter.get("/models").raise_for_status()
                #     logger.debug("OpenRouter client connectivity test successful.")
                # except httpx.HTTPStatusError as e:
                #     logger.error(f"OpenRouter client test failed (HTTP {e.response.status_code}): Check API Key and connectivity. Response: {e.response.text}")
                #     if e.response.status_code == 401:
                #         raise RuntimeError("OpenRouter authentication failed (401 Unauthorized). Check API Key.")
                #     else:
                #         raise RuntimeError(f"OpenRouter client test failed: {e}")

            except KeyError as e:
                raise RuntimeError(f"Missing OpenRouter API key (OPENROUTER_API_KEY): {e}")
            except Exception as e: # Catch other potential httpx client init errors
                raise RuntimeError(f"Failed to initialize OpenRouter client: {e}")
        return self._openrouter

    @property
    def reddit(self) -> praw.Reddit:
        if not hasattr(self, '_reddit'):
            try:
                if not all(os.getenv(k) for k in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']):
                    raise RuntimeError("Reddit credentials (ID, Secret, User-Agent) not fully set in environment.")
                self._reddit = praw.Reddit(
                    client_id=os.environ['REDDIT_CLIENT_ID'],
                    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
                    user_agent=os.environ['REDDIT_USER_AGENT'],
                    read_only=True # Explicitly set read-only mode
                )
                # Test connection/auth early by checking read-only status (forces auth)
                is_read_only = self._reddit.read_only
                logger.debug(f"Reddit client initialized (Read-Only: {is_read_only}).")
                if not is_read_only and self._reddit.user.me() is None: # Should be read-only, but extra check
                    logger.warning("Reddit client may not be properly authenticated even though no error was raised.")

            except (KeyError, prawcore.exceptions.OAuthException, prawcore.exceptions.ResponseException, RuntimeError) as e:
                 # Improve error message for common Reddit auth issues
                 err_msg = str(e)
                 if '401' in err_msg or 'invalid_client' in err_msg or 'unauthorized' in err_msg:
                     raise RuntimeError(f"Reddit authentication failed: Check Client ID/Secret. ({e})")
                 else:
                     logger.error(f"Reddit setup failed: {e}")
                     raise RuntimeError(f"Reddit setup failed: {e}")
        return self._reddit

    @property
    def twitter(self) -> tweepy.Client:
        if not hasattr(self, '_twitter'):
            try:
                token = os.getenv('TWITTER_BEARER_TOKEN')
                if not token:
                    raise RuntimeError("Twitter Bearer Token (TWITTER_BEARER_TOKEN) not set.")
                self._twitter = tweepy.Client(bearer_token=token, wait_on_rate_limit=False)
                # Test connection with a simple, low-impact call
                self._twitter.get_user(username="twitterdev", user_fields=["id"]) # Example known user, minimal fields
                logger.debug("Twitter client initialized and connection tested.")
            except (KeyError, RuntimeError) as e: # Catch setup errors first
                logger.error(f"Twitter setup failed: {e}")
                raise RuntimeError(f"Twitter setup failed: {e}")
            except tweepy.errors.Unauthorized as e:
                logger.error(f"Twitter authentication failed (Unauthorized): Check Bearer Token. {e}")
                raise RuntimeError(f"Twitter authentication failed: Invalid Bearer Token? ({e})")
            except tweepy.errors.TweepyException as e:
                logger.error(f"Twitter API error during initialization: {e}")
                raise RuntimeError(f"Twitter client initialization failed: {e}")
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
                    reset_timestamp = int(rate_limit_reset)
                    reset_time = datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)
                    current_time = datetime.now(timezone.utc)
                    # Calculate wait_seconds, ensuring it's at least 1 second + buffer
                    wait_seconds = max(int((reset_time - current_time).total_seconds()) + 5, 1)
                    reset_info = f"Try again after: {reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ({wait_seconds}s)"
                except (ValueError, TypeError):
                    logger.warning("Could not parse Twitter rate limit reset time header.")
                    reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
            else:
                reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying (reset header missing)."

        elif isinstance(exception, (prawcore.exceptions.RequestException, httpx.HTTPStatusError)) and \
             hasattr(exception, 'response') and exception.response is not None and exception.response.status_code == 429:
                 # Try to parse Reddit's rate limit headers if available
                 retry_after = exception.response.headers.get('x-ratelimit-reset') # Seconds until reset
                 if retry_after and retry_after.isdigit():
                     wait_seconds = int(retry_after) + 5 # Add buffer
                     reset_info = f"Try again in {wait_seconds} seconds."
                 else:
                     reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."

        # +++ Add Mastodon Rate Limit Handling +++
        elif isinstance(exception, MastodonRatelimitError):
            # Mastodon headers *might* contain rate limit info ('X-RateLimit-Reset')
            # The exception object itself doesn't easily expose the response in mastodon-py v1.8
            # Rely on default wait time for now.
            reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
            # Log header details if possible (might need modification if lib changes)
            # logger.debug(f"Mastodon rate limit triggered. Headers (if available): {getattr(exception, 'response', {}).get('headers', {})}")

        # --- End Mastodon Rate Limit Handling ---
        elif isinstance(exception, atproto_exceptions.AtProtocolError) and 'rate limit' in str(exception).lower(): # Use alias
             reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
             # Bluesky might return JSON with rate limit details, parse if possible
             try:
                 if hasattr(exception, 'response') and exception.response:
                      err_data = json.loads(exception.response.content)
                      if err_data.get('error') == 'RateLimitExceeded':
                           # Future: Parse details if the API provides them consistently
                           pass
             except (json.JSONDecodeError, AttributeError):
                 pass # Stick to default message if parsing fails

        else:
             # Check specifically for OpenRouter rate limit (often HTTP 429)
             if isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code == 429:
                 reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
                 error_message = f"Image Analysis ({platform}) API rate limit exceeded." # More specific message
             else:
                 # Default for unknown rate limit causes
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
        # Sanitize username for path component (redundant as hash is used, but safer if structure changes)
        safe_username = "".join(c if c.isalnum() or c in ['-', '_', '.', '@'] else '_' for c in username)
        # Consider adding platform/username to path for organization, though hash prevents collisions
        # return self.base_dir / 'media' / platform / safe_username[:50] / f"{url_hash}.media"
        return self.base_dir / 'media' / f"{url_hash}.media" # Keeping simple hash-based flat structure

    def _download_media(self, url: str, platform: str, username: str, headers: Optional[dict] = None) -> Optional[Path]:
        """Downloads media, saves with correct extension, returns path if successful."""
        media_path_stub = self._get_media_path(url, platform, username)
        # Check if any file with this hash exists (might have different extensions)
        # Use stem to match hash regardless of guessed/actual extension
        existing_files = list((self.base_dir / 'media').glob(f'{media_path_stub.stem}.*'))
        if existing_files:
            # Prefer common image types if multiple exist (e.g., jpg over media)
            preferred_exts = ['.jpg', '.png', '.webp', '.gif', '.mp4', '.webm']
            for ext in preferred_exts:
                 found_path = (self.base_dir / 'media' / f"{media_path_stub.stem}{ext}")
                 if found_path.exists() and found_path in existing_files:
                     logger.debug(f"Media cache hit (preferred ext): {found_path}")
                     return found_path
            # Otherwise, return the first one found (could be .media or other)
            logger.debug(f"Media cache hit (generic): {existing_files[0]}")
            return existing_files[0]

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
            # Twitter/Bluesky may need auth for certain media URLs.
            auth_headers = {}
            # Use try-except when accessing client properties to handle initialization errors gracefully
            try:
                if platform == 'twitter':
                    if not hasattr(self, '_twitter'): self.twitter # Init if needed
                    token = os.getenv('TWITTER_BEARER_TOKEN')
                    if token: auth_headers['Authorization'] = f'Bearer {token}'
                elif platform == 'bluesky':
                     if not hasattr(self, '_bluesky_client'): self.bluesky # Init if needed
                     # Accessing protected member, consider refactoring Bluesky client interaction if possible
                     access_token = getattr(self.bluesky._session, 'access_jwt', None)
                     if not access_token:
                         logger.warning(f"Bluesky access token not available for media download for URL: {url}")
                     else:
                         auth_headers['Authorization'] = f"Bearer {access_token}"
                     # CDN URL adjustments are handled in fetch_bluesky now
            except RuntimeError as client_init_err:
                 logger.warning(f"Cannot add auth headers for {platform} media download, client init failed: {client_init_err}")


            # Combine provided headers with auth headers
            request_headers = headers.copy() if headers else {}
            request_headers.update(auth_headers)
            # Add a generic user-agent
            request_headers.setdefault('User-Agent', 'SocialOSINTLM/1.0')


            with httpx.Client(follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
                resp = client.get(url, headers=request_headers) # Use combined headers
                resp.raise_for_status()

            content_type = resp.headers.get('content-type', '').lower().split(';')[0].strip()
            extension = valid_types.get(content_type)

            if not extension:
                # Log first few bytes to help identify unknown types
                content_preview = resp.content[:64]
                logger.warning(f"Unsupported or non-media type '{content_type}' for URL: {url}. Content preview (bytes): {content_preview}")
                # Might be HTML page etc.
                return None

            # Create directory if it doesn't exist (needed if using subdirs in _get_media_path)
            # media_path_stub.parent.mkdir(parents=True, exist_ok=True) # Uncomment if using subdirs

            final_media_path = media_path_stub.with_suffix(extension)
            final_media_path.write_bytes(resp.content)
            logger.debug(f"Downloaded media to: {final_media_path}")
            return final_media_path

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Guess platform based on URL structure if not directly passed? Or rely on caller context.
                # Assume platform passed in is correct context for rate limit message.
                self._handle_rate_limit(f"{platform} Media Download", e) # Use platform context
            elif e.response.status_code in [404, 410]: # 410 Gone
                 logger.warning(f"Media not found ({e.response.status_code}) for {url}. Skipping.")
            elif e.response.status_code in [403, 401]:
                 logger.warning(f"Media access forbidden/unauthorized ({e.response.status_code}) for {url}. Skipping.")
            else:
                logger.error(f"HTTP error {e.response.status_code} downloading {url}: {e}. Response: {e.response.text[:200]}") # Log response snippet
            return None
        except httpx.RequestError as e: # Catch network errors like timeouts, DNS issues
             logger.error(f"Network error downloading {url}: {e}")
             return None
        except Exception as e:
            logger.error(f"Media download failed unexpectedly for {url}: {str(e)}", exc_info=False)
            return None

    def _analyze_image(self, file_path: Path, context: str = "") -> Optional[str]:
        """Analyzes image using OpenRouter, handles resizing and errors."""
        if not file_path or not file_path.exists():
            logger.warning(f"Image analysis skipped: file path invalid or missing ({file_path})")
            return None

        # Check file extension - only analyze supported image types using the constant
        if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
             logger.debug(f"Skipping analysis for non-image file: {file_path}")
             return None

        temp_path = None # Define outside try block
        analysis_file_path = file_path # Default to original path
        original_format = None

        try:
            # Use context manager for image opening
            with Image.open(file_path) as img:
                original_format = img.format.lower() if img.format else None
                # Double check format against supported types (PIL might open things it can't fully process)
                if original_format not in ['jpeg', 'png', 'webp', 'gif']:
                    logger.warning(f"Unsupported image type detected by PIL: {original_format} at {file_path}")
                    return None

                # --- Resizing and Conversion Logic ---
                max_dimension = 1536 # Max dimension for analysis models
                target_format = 'JPEG' # Convert to JPEG for consistency with API expectations
                needs_resizing = max(img.size) > max_dimension
                needs_conversion = original_format != 'jpeg'
                is_animated = getattr(img, 'is_animated', False) and getattr(img, 'n_frames', 1) > 1

                img_to_process = img
                # Handle animated GIFs: Use the first frame
                if is_animated:
                    img.seek(0)
                    # If also needs converting (i.e., GIF -> JPEG), ensure we have the first frame loaded
                    # PIL can be tricky here; creating a copy might be safer
                    img_to_process = img.copy()
                    logger.debug(f"Using first frame of animated image: {file_path}")
                    needs_conversion = True # Force conversion if animated

                # Ensure RGB mode before resizing/saving as JPEG (handles RGBA, P, etc.)
                if img_to_process.mode != "RGB":
                     if img_to_process.mode == "P" and 'transparency' in img_to_process.info:
                          img_to_process = img_to_process.convert("RGBA") # Convert P with transparency via RGBA
                     if img_to_process.mode == "RGBA":
                          # Create a white background and paste RGBA image onto it
                          bg = Image.new("RGB", img_to_process.size, (255, 255, 255))
                          bg.paste(img_to_process, mask=img_to_process.split()[3]) # Use alpha channel as mask
                          img_to_process = bg
                          logger.debug(f"Converted RGBA image to RGB with white background: {file_path}")
                     else:
                          # General conversion for other modes (like L, P without transparency)
                          img_to_process = img_to_process.convert("RGB")
                          logger.debug(f"Converted image mode {img.mode} to RGB: {file_path}")
                     needs_conversion = True # Conversion happened

                # Apply resizing if needed
                if needs_resizing:
                    scale_factor = max_dimension / max(img_to_process.size)
                    new_size = (int(img_to_process.size[0] * scale_factor), int(img_to_process.size[1] * scale_factor))
                    img_to_process = img_to_process.resize(new_size, Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image to {new_size}: {file_path}")
                    needs_conversion = True # Treat resizing as requiring re-saving/conversion

                # Save to temporary file if resizing or conversion occurred
                if needs_conversion: # This flag is true if resized, format changed, mode changed, or animated
                    # Use a more descriptive temp name
                    temp_suffix = f".processed.{target_format.lower()}"
                    temp_path = file_path.with_suffix(temp_suffix)
                    img_to_process.save(temp_path, target_format, quality=85) # Save converted/resized
                    analysis_file_path = temp_path
                    logger.debug(f"Saved processed image for analysis: {analysis_file_path}")
                else:
                    # Original file is already a suitable JPEG within size limits
                    analysis_file_path = file_path
                    logger.debug(f"Using original image file for analysis: {analysis_file_path}")
            # --- End Resizing/Conversion ---


            base64_image = base64.b64encode(analysis_file_path.read_bytes()).decode('utf-8')
            image_data_url = f"data:image/jpeg;base64,{base64_image}" # Use JPEG as target format

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
                                "url": image_data_url,
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
                 logger.error(f"Image analysis API error (Code: {err_code}) for {file_path}: {err_msg}")
                 return None # Return None on API logic errors
            if not result.get('choices') or not result['choices'][0].get('message') or not result['choices'][0]['message'].get('content'):
                logger.error(f"Invalid image analysis API response structure for {file_path}: {result}")
                return None

            analysis_text = result['choices'][0]['message']['content']
            logger.debug(f"Image analysis successful for: {file_path}")
            return analysis_text.strip() # Strip whitespace from result

        except (IOError, Image.DecompressionBombError, SyntaxError, ValueError) as img_err: # Added SyntaxError/ValueError for corrupt images/bad data
             logger.error(f"Image processing error for {file_path}: {str(img_err)}")
             return None
        except httpx.RequestError as req_err:
             logger.error(f"Network error during image analysis API call for {file_path}: {str(req_err)}")
             return None # Network errors are often transient, don't raise fatal
        except httpx.HTTPStatusError as status_err:
            model_to_use = os.getenv('IMAGE_ANALYSIS_MODEL', 'google/gemini-pro-vision')
            response_text = status_err.response.text[:500] # Limit log size
            if status_err.response.status_code == 429:
                 # Pass the model name for clearer rate limit message
                 self._handle_rate_limit(model_to_use, status_err) # Should raise RateLimitExceededError
            # --- Explicitly check for 401 Unauthorized ---
            elif status_err.response.status_code == 401:
                 # This is where the original error likely occurred
                 logger.error(f"HTTP 401 Unauthorized during image analysis ({model_to_use}). Check your OPENROUTER_API_KEY.")
                 logger.error(f"API Response Snippet: {response_text}") # Log API response snippet
                 # Raise a runtime error to halt if auth fails critically? Or just return None? Return None for now.
                 return None
            elif status_err.response.status_code == 400:
                  logger.error(f"HTTP 400 Bad Request during image analysis ({model_to_use}). Often due to invalid input (e.g., unsupported image format by API, malformed request). Check image and API docs.")
                  logger.error(f"API Response Snippet: {response_text}")
                  return None
            else:
                 logger.error(f"HTTP error {status_err.response.status_code} during image analysis ({model_to_use}): {response_text}")
            return None # Return None for HTTP errors other than handled rate limits
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
        # Limit length to avoid filesystem issues
        safe_username = safe_username[:100]
        return self.base_dir / 'cache' / f"{platform}_{safe_username}.json"

    def _load_cache(self, platform: str, username: str) -> Optional[Dict[str, Any]]:
        """Loads cache data if it exists and is not expired."""
        cache_path = self._get_cache_path(platform, username)
        if not cache_path.exists():
            logger.debug(f"Cache miss (file not found): {cache_path}")
            return None

        try:
            logger.debug(f"Attempting to load cache: {cache_path}")
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            # Check for essential timestamp key first
            if 'timestamp' not in data:
                 logger.warning(f"Cache file for {platform}/{username} is missing timestamp. Discarding.")
                 cache_path.unlink(missing_ok=True)
                 return None

            # Safely parse timestamp
            try:
                timestamp_str = data['timestamp']
                # Handle both datetime objects saved by older versions and ISO strings
                if isinstance(timestamp_str, str):
                     timestamp = datetime.fromisoformat(timestamp_str)
                elif isinstance(timestamp_str, datetime): # Should not happen with DateTimeEncoder but handle defensively
                     timestamp = timestamp_str
                else:
                     raise ValueError("Invalid timestamp format in cache")
                # Ensure timestamp is timezone-aware (assume UTC if not)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)

            except (ValueError, TypeError) as ts_err:
                logger.warning(f"Failed to parse timestamp in cache for {platform}/{username}: {ts_err}. Discarding cache.")
                cache_path.unlink(missing_ok=True)
                return None


            # Check expiry using timezone-aware comparison
            if datetime.now(timezone.utc) - timestamp < timedelta(hours=CACHE_EXPIRY_HOURS):
                 required_keys = ['timestamp']
                 # Define required keys for each platform more carefully
                 if platform == 'mastodon': required_keys.extend(['posts', 'user_info', 'stats'])
                 elif platform == 'twitter': required_keys.extend(['tweets', 'user_info']) # media_analysis/paths are optional additions
                 elif platform == 'reddit': required_keys.extend(['submissions', 'comments', 'stats'])
                 elif platform == 'bluesky': required_keys.extend(['posts', 'stats']) # profile_info is optional but usually present
                 elif platform == 'hackernews': required_keys.extend(['submissions', 'stats'])

                 # Check if all required keys exist in the loaded data
                 missing_keys = [key for key in required_keys if key not in data]
                 if not missing_keys:
                      logger.info(f"Cache hit and valid for {platform}/{username}")
                      return data
                 else:
                     logger.warning(f"Cache file for {platform}/{username} is incomplete (missing: {missing_keys}). Discarding.")
                     cache_path.unlink(missing_ok=True) # Ensure deletion even if missing
                     return None
            else:
                logger.info(f"Cache expired for {platform}/{username}. Returning stale data for incremental baseline.")
                return data # Return expired data for incremental update baseline

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Failed to load or parse cache for {platform}/{username}: {e}. Discarding cache.")
            # Use missing_ok=True for unlink in case the file disappears between check and unlink
            cache_path.unlink(missing_ok=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading cache for {platform}/{username}: {e}", exc_info=True)
            cache_path.unlink(missing_ok=True)
            return None


    def _save_cache(self, platform: str, username: str, data: Dict[str, Any]):
        """Saves data to cache, ensuring timestamp is updated and data is sorted."""
        cache_path = self._get_cache_path(platform, username)
        try:
            # Define sort keys robustly
            # Format: {platform: [(list_key, datetime_key), ...]}
            sort_key_map = {
                'twitter': [('tweets', 'created_at')],
                'reddit': [('submissions', 'created_utc'), ('comments', 'created_utc')],
                'bluesky': [('posts', 'created_at')],
                'hackernews': [('submissions', 'created_at')],
                'mastodon': [('posts', 'created_at')],
            }

            # Sort relevant lists within the data dictionary
            if platform in sort_key_map:
                 for list_key, dt_key in sort_key_map[platform]:
                     if list_key in data and isinstance(data[list_key], list) and data[list_key]:
                         # Filter out items that might cause sorting errors (e.g., missing dt_key) although get_sort_key handles it
                         items_to_sort = [item for item in data[list_key] if dt_key in item]
                         # Sort using the globally defined get_sort_key function
                         items_to_sort.sort(key=lambda x: get_sort_key(x, dt_key), reverse=True)
                         # Handle potential missing items during filtering (though unlikely if dt_key check passes)
                         # It's generally safer to sort the original list if get_sort_key is robust
                         data[list_key].sort(key=lambda x: get_sort_key(x, dt_key), reverse=True)
                         logger.debug(f"Sorted '{list_key}' for {platform}/{username} by '{dt_key}'.")
                     # else: logger.debug(f"List '{list_key}' not found or empty for {platform}/{username}, skipping sort.")


            data['timestamp'] = datetime.now(timezone.utc) # Use object for encoder
            # Ensure parent directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(data, indent=2, cls=DateTimeEncoder),
                encoding='utf-8'
            )
            logger.info(f"Saved cache for {platform}/{username} to {cache_path}")
        except TypeError as e:
             logger.error(f"Failed to serialize data for {platform}/{username} cache (TypeError): {e}. Data snippet: {str(data)[:500]}...", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to save cache for {platform}/{username}: {e}", exc_info=True)


    # --- Platform Fetch Methods (with Incremental Logic) ---

    def fetch_twitter(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches Twitter tweets and user info."""
        # Use the constant for image extensions
        # supported_image_extensions = SUPPORTED_IMAGE_EXTENSIONS # Defined globally now

        cached_data = self._load_cache('twitter', username)

        # Condition 1: Cache is valid, recent, and not forced refresh
        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, 'timestamp')) < timedelta(hours=CACHE_EXPIRY_HOURS): # Use getter for cache ts
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
            if existing_tweets: # Sort only if list exists and is not empty
                existing_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
                since_id = existing_tweets[0].get('id') # Use .get for safety
                if since_id:
                    logger.debug(f"Using since_id: {since_id}")
                else:
                    logger.warning("Found existing tweets but couldn't get ID from the first one.")
            user_info = cached_data.get('user_info') # Keep existing user info
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])


        try:
            # --- Get User ID ---
            # User info might be missing on first fetch or corrupted cache
            if not user_info or force_refresh:
                try:
                     # Ensure client is ready (access property)
                     tw_client = self.twitter
                     user_response = tw_client.get_user(username=username, user_fields=['created_at', 'public_metrics', 'profile_image_url', 'verified', 'description', 'location'])
                     if not user_response or not user_response.data:
                         raise UserNotFoundError(f"Twitter user @{username} not found.")
                     user = user_response.data
                     # Convert datetime to string immediately for consistent JSON saving
                     created_at_iso = user.created_at.isoformat() if user.created_at else None
                     user_info = {
                         'id': str(user.id), # Ensure ID is string
                         'name': user.name,
                         'username': user.username,
                         'created_at': created_at_iso,
                         'public_metrics': user.public_metrics,
                         'profile_image_url': user.profile_image_url,
                         'verified': user.verified,
                         'description': user.description,
                         'location': user.location,
                     }
                     logger.debug(f"Fetched user info for @{username}")
                except tweepy.NotFound:
                     raise UserNotFoundError(f"Twitter user @{username} not found.")
                except tweepy.Forbidden as e:
                     # Check if it's a suspension message
                     if 'suspended' in str(e).lower():
                          raise AccessForbiddenError(f"Twitter user @{username} is suspended.")
                     else:
                          raise AccessForbiddenError(f"Access forbidden to Twitter user @{username}'s profile (protected/private?).")
                # Catch potential auth errors during client access/use
                except (tweepy.errors.Unauthorized, tweepy.errors.TweepyException) as auth_err:
                     logger.error(f"Twitter API authentication/request error getting user @{username}: {auth_err}")
                     raise RuntimeError(f"Twitter API error: {auth_err}") # Re-raise as runtime


            user_id = user_info['id'] # Use string ID

            # --- Fetch Tweets ---
            new_tweets_data = []
            new_media_includes = {} # Store includes from new fetches

            # Use pagination for potentially large number of new tweets since last check
            fetch_limit = INITIAL_FETCH_LIMIT if (force_refresh or not since_id) else INCREMENTAL_FETCH_LIMIT
            pagination_token = None
            tweets_fetch_count = 0 # Track actual tweets fetched

            while True: # Loop for pagination
                current_page_limit = min(fetch_limit - tweets_fetch_count, 100) # Max 100 per page for v2 tweets endpoint
                if current_page_limit <= 0: break # Stop if fetch limit reached

                try:
                    # Ensure client is ready
                    tw_client = self.twitter
                    tweets_response = tw_client.get_users_tweets(
                        id=user_id,
                        max_results=current_page_limit,
                        since_id=since_id if not force_refresh else None, # Only use since_id for incremental
                        pagination_token=pagination_token,
                        tweet_fields=['created_at', 'public_metrics', 'attachments', 'entities', 'conversation_id', 'in_reply_to_user_id', 'referenced_tweets'], # Added more context
                        expansions=['attachments.media_keys', 'author_id'], # Author_id redundant but good practice
                        media_fields=['url', 'preview_image_url', 'type', 'media_key', 'width', 'height', 'alt_text'] # Added alt_text
                    )
                except tweepy.TooManyRequests as e:
                    self._handle_rate_limit('Twitter', exception=e)
                    # Return None to indicate fetch aborted due to rate limit
                    # The calling function should handle this (e.g., skip analysis for this user)
                    return None
                except tweepy.NotFound:
                    # If user existed but tweets now 404, could be deleted/suspended/protected between calls
                    raise UserNotFoundError(f"Tweets not found for user ID {user_id} (@{username}). User might be protected, suspended or deleted after profile check.")
                except tweepy.Forbidden as e:
                     # Distinguish between protected and other forbidden if possible
                     if 'protected' in str(e).lower():
                         raise AccessForbiddenError(f"Access forbidden to @{username}'s tweets (account is protected).")
                     else:
                         raise AccessForbiddenError(f"Access forbidden to @{username}'s tweets (reason: {e}).")
                 # Catch potential auth errors during client access/use
                except (tweepy.errors.Unauthorized, tweepy.errors.TweepyException) as auth_err:
                     logger.error(f"Twitter API authentication/request error getting tweets for @{username}: {auth_err}")
                     raise RuntimeError(f"Twitter API error: {auth_err}") # Re-raise as runtime

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
                        # Avoid duplicates using media_key
                        existing_keys = {item['media_key'] for item in new_media_includes[key] if 'media_key' in item}
                        for item in items:
                            if 'media_key' in item and item['media_key'] not in existing_keys:
                                new_media_includes[key].append(item)
                                existing_keys.add(item['media_key'])

                pagination_token = tweets_response.meta.get('next_token')
                # Stop if no more pages OR fetch limit reached
                if not pagination_token or tweets_fetch_count >= fetch_limit:
                    if pagination_token and tweets_fetch_count >= fetch_limit:
                         logger.info(f"Reached fetch limit ({fetch_limit}) for Twitter @{username}.")
                    elif not pagination_token:
                         logger.debug("No more pages found for Twitter tweets.")
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
                             # Prefer media.url for photos/gifs, fallback to preview_image_url (videos often lack direct url)
                             url = media.url if media.type in ['photo', 'gif'] and media.url else media.preview_image_url
                             if url:
                                 media_path = self._download_media(url=url, platform='twitter', username=username)
                                 if media_path:
                                     # Analyze only if it's a supported image type
                                     analysis = None
                                     # Use the global constant here
                                     if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                         analysis = self._analyze_image(media_path, f"Twitter user @{username}'s tweet (ID: {tweet.id})")

                                     media_items_for_tweet.append({
                                         'type': media.type,
                                         'analysis': analysis, # Will be None if not analyzed
                                         'url': url, # Store the URL we tried to download
                                         'alt_text': media.alt_text, # Store alt text
                                         'local_path': str(media_path)
                                     })
                                     if analysis: newly_added_media_analysis.append(analysis)
                                     newly_added_media_paths.add(str(media_path))
                                 else:
                                     logger.warning(f"Failed to download media {media.type} from {url} for tweet {tweet.id}")


                 # Process referenced tweets (replies, quotes)
                 referenced_tweets_info = []
                 if tweet.referenced_tweets:
                     for ref in tweet.referenced_tweets:
                         referenced_tweets_info.append({'type': ref.type, 'id': str(ref.id)})

                 tweet_data = {
                     'id': str(tweet.id), # Ensure ID is string
                     'text': tweet.text,
                     'created_at': tweet.created_at.isoformat(), # Store as ISO string
                     'metrics': tweet.public_metrics,
                     'entities': tweet.entities, # Store entities (URLs, mentions, hashtags)
                     'conversation_id': str(tweet.conversation_id), # Ensure string
                     'in_reply_to_user_id': str(tweet.in_reply_to_user_id) if tweet.in_reply_to_user_id else None,
                     'referenced_tweets': referenced_tweets_info,
                     'media': media_items_for_tweet # Add processed media
                 }
                 processed_new_tweets.append(tweet_data)

            # --- Combine and Prune ---
            # Sort new tweets before combining (API generally returns newest first, but be sure)
            processed_new_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper

            # De-duplicate based on ID before combining and pruning
            existing_ids = {t['id'] for t in existing_tweets}
            unique_new_tweets = [t for t in processed_new_tweets if t['id'] not in existing_ids]

            combined_tweets = unique_new_tweets + existing_tweets
            # Sort again after combining unique new ones
            combined_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True)
            # Prune based on MAX_CACHE_ITEMS
            final_tweets = combined_tweets[:MAX_CACHE_ITEMS]

            # Combine media analysis and paths (only add new unique ones)
            # Filter out None/empty analyses before adding
            valid_new_analyses = [a for a in newly_added_media_analysis if a]
            # Simple de-duplication by converting to set and back
            final_media_analysis = sorted(list(set(valid_new_analyses + [m for m in existing_media_analysis if m])))
            final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))[:MAX_CACHE_ITEMS * 2] # Limit paths too

            # --- Prepare Final Data ---
            final_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(), # Use ISO format string
                'user_info': user_info,
                'tweets': final_tweets,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths
            }

            self._save_cache('twitter', username, final_data)
            logger.info(f"Successfully updated Twitter cache for @{username}. Total tweets cached: {len(final_tweets)}")
            return final_data

        except RateLimitExceededError:
             logger.warning(f"Twitter fetch for @{username} aborted due to rate limit.")
             return None # Indicate fetch failed due to rate limit
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Twitter fetch failed for @{username}: {user_err}")
             return None # Return None, error already logged
        except RuntimeError as e: # Catch re-raised API/Auth errors
             logger.error(f"Runtime error during Twitter fetch for @{username}: {e}")
             # Potentially re-raise critical errors? For now, return None.
             return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Twitter data for @{username}: {str(e)}", exc_info=True)
            return None


    def fetch_reddit(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches Reddit submissions and comments."""
        # Use the constant for image extensions
        # supported_image_extensions = SUPPORTED_IMAGE_EXTENSIONS # Defined globally now

        cached_data = self._load_cache('reddit', username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, 'timestamp')) < timedelta(hours=CACHE_EXPIRY_HOURS): # Use getter
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
            if existing_submissions:
                 existing_submissions.sort(key=lambda x: get_sort_key(x, 'created_utc'), reverse=True) # Use global helper
                 latest_submission_fullname = existing_submissions[0].get('fullname') # Use get
                 if latest_submission_fullname: logger.debug(f"Using latest submission fullname: {latest_submission_fullname}")
            if existing_comments:
                 existing_comments.sort(key=lambda x: get_sort_key(x, 'created_utc'), reverse=True) # Use global helper
                 latest_comment_fullname = existing_comments[0].get('fullname') # Use get
                 if latest_comment_fullname: logger.debug(f"Using latest comment fullname: {latest_comment_fullname}")

            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])

        try:
            # Ensure client is ready
            reddit_client = self.reddit
            redditor = reddit_client.redditor(username)
            try:
                # Accessing redditor properties forces a check if the user exists and is accessible
                redditor_id = redditor.id
                redditor_created_utc = redditor.created_utc
                logger.debug(f"Reddit user u/{username} found (ID: {redditor_id}). Created: {datetime.fromtimestamp(redditor_created_utc, tz=timezone.utc)}")
                # Fetch basic profile info once
                redditor_info = {
                     'id': redditor_id,
                     'name': redditor.name,
                     'created_utc': datetime.fromtimestamp(redditor_created_utc, tz=timezone.utc).isoformat(),
                     'link_karma': getattr(redditor, 'link_karma', 0), # Use getattr for safety
                     'comment_karma': getattr(redditor, 'comment_karma', 0),
                     'icon_img': getattr(redditor, 'icon_img', None),
                     'is_suspended': getattr(redditor, 'is_suspended', False) # Check suspension status
                }
                if redditor_info['is_suspended']:
                     logger.warning(f"Reddit user u/{username} is suspended.")
                     # Optionally raise AccessForbiddenError here if suspended users should be skipped entirely
                     # raise AccessForbiddenError(f"Reddit user u/{username} is suspended.")

            except prawcore.exceptions.NotFound:
                raise UserNotFoundError(f"Reddit user u/{username} not found.")
            except prawcore.exceptions.Forbidden:
                 # This could be shadowban, suspension, PII block etc.
                 raise AccessForbiddenError(f"Access forbidden to Reddit user u/{username} (possibly suspended, shadowbanned, or blocked).")
            except (prawcore.exceptions.PrawcoreException, RuntimeError) as client_err: # Catch other PRAW or setup errors
                 logger.error(f"Reddit API/client error accessing user u/{username}: {client_err}")
                 raise RuntimeError(f"Reddit API error: {client_err}")


            # --- Fetch New Submissions ---
            new_submissions_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            fetch_limit = INCREMENTAL_FETCH_LIMIT # Limit incremental fetch
            count = 0
            processed_ids = {s['id'] for s in existing_submissions} # Track existing submission IDs

            logger.debug("Fetching new submissions...")
            try:
                # Fetch using 'before' parameter with latest fullname for more efficient incremental fetching
                params = {'limit': fetch_limit} # PRAW handles limit internally, but params needed for 'before'
                if not force_refresh and latest_submission_fullname:
                    params['before'] = latest_submission_fullname
                    logger.debug(f"Fetching submissions before {latest_submission_fullname}")

                # Use limit parameter directly in the call
                for submission in redditor.submissions.new(limit=fetch_limit, params=params):
                    count += 1
                    submission_fullname = submission.fullname
                    submission_id = submission.id

                    # Avoid reprocessing already cached submissions
                    if submission_id in processed_ids:
                        logger.debug(f"Skipping already processed submission ID: {submission_id}")
                        continue

                    media_items_for_submission = [] # Process media first
                    media_processed_inline = False # Track if direct URL was handled

                    # Direct Image/GIF/Video URL (check common extensions and Reddit domains)
                    submission_url = getattr(submission, 'url', None)
                    if submission_url:
                         # Check common media extensions
                         is_direct_media_link = any(submission_url.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS + ['.mp4', '.webm', '.mov'])
                         # Check common Reddit media hosts
                         is_reddit_media = any(host in urlparse(submission_url).netloc for host in ['i.redd.it', 'v.redd.it', 'preview.redd.it'])

                         if is_direct_media_link or is_reddit_media:
                              media_path = self._download_media(url=submission_url, platform='reddit', username=username)
                              if media_path:
                                  analysis = None
                                  # Analyze images only
                                  if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                      analysis = self._analyze_image(media_path, f"Reddit user u/{username}'s post in r/{submission.subreddit.display_name} (ID: {submission_id})")
                                  media_items_for_submission.append({
                                      'type': 'image' if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else 'video', # Simple type guess
                                      'analysis': analysis,
                                      'url': submission_url,
                                      'local_path': str(media_path)
                                  })
                                  if analysis: newly_added_media_analysis.append(analysis)
                                  newly_added_media_paths.add(str(media_path))
                                  media_processed_inline = True

                    # Reddit Gallery (only if direct URL wasn't processed and it's a gallery)
                    is_gallery = getattr(submission, 'is_gallery', False)
                    media_metadata = getattr(submission, 'media_metadata', None)
                    if not media_processed_inline and is_gallery and media_metadata:
                        for media_id, media_item in media_metadata.items():
                             # Extract image URL from gallery metadata (prefer highest res 'u', fallback to 'gif' or 'p')
                             source = media_item.get('s') # Source dictionary
                             preview_data = media_item.get('p', []) # Preview list (lower res)
                             image_url = None
                             if source: image_url = source.get('u') or source.get('gif')
                             if not image_url and preview_data: image_url = preview_data[-1].get('u') # Get highest res preview URL

                             if image_url:
                                 # Reddit URLs often have escaped ampersands
                                 image_url = image_url.replace('&amp;', '&')
                                 media_path = self._download_media(url=image_url, platform='reddit', username=username)
                                 if media_path:
                                     analysis = None
                                     if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                         analysis = self._analyze_image(media_path, f"Reddit user u/{username}'s gallery post in r/{submission.subreddit.display_name} (ID: {submission_id}, Media: {media_id})")
                                     media_items_for_submission.append({
                                         'type': 'gallery_image',
                                         'analysis': analysis,
                                         'url': image_url,
                                         'alt_text': media_item.get('caption') or media_item.get('title'), # Use caption/title as alt-text proxy
                                         'local_path': str(media_path)
                                     })
                                     if analysis: newly_added_media_analysis.append(analysis)
                                     newly_added_media_paths.add(str(media_path))

                    submission_data = {
                        'id': submission_id,
                        'fullname': submission_fullname,
                        'title': submission.title,
                        'text': submission.selftext[:2000] if hasattr(submission, 'selftext') else '', # Increased snippet
                        'score': submission.score,
                        'upvote_ratio': getattr(submission, 'upvote_ratio', None),
                        'subreddit': submission.subreddit.display_name,
                        'permalink': f"https://www.reddit.com{submission.permalink}",
                        'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(), # Store ISO string
                        'url': submission.url if submission.is_self else None, # URL if link post
                        'link_url': submission.url if not submission.is_self else None, # URL if link post (clearer name)
                        'is_self': submission.is_self,
                        'is_gallery': is_gallery,
                        'num_comments': submission.num_comments,
                        'stickied': submission.stickied,
                        'over_18': submission.over_18,
                        'spoiler': submission.spoiler,
                        'media': media_items_for_submission # Add processed media
                    }
                    new_submissions_data.append(submission_data)
                    processed_ids.add(submission_id) # Track processed ID

            except prawcore.exceptions.Forbidden:
                logger.warning(f"Access forbidden while fetching submissions for u/{username} (possibly subreddit restriction).")
            except prawcore.exceptions.RequestException as req_err:
                 # Handle potential 429 specifically
                 if hasattr(req_err, 'response') and req_err.response is not None and req_err.response.status_code == 429:
                     self._handle_rate_limit('Reddit', exception=req_err)
                     return None # Abort fetch
                 else:
                     logger.error(f"Reddit request failed fetching submissions for u/{username}: {req_err}")
                     # Decide whether to continue or abort; maybe continue to comments? For now, log and continue.
            logger.info(f"Fetched {len(new_submissions_data)} new submissions for u/{username} (scanned approx {count}).")


            # --- Fetch New Comments ---
            new_comments_data = []
            count = 0
            processed_comment_ids = {c['id'] for c in existing_comments} # Track existing comment IDs
            logger.debug("Fetching new comments...")
            try:
                # Fetch a batch and filter locally, use 'before' param
                params = {'limit': fetch_limit}
                if not force_refresh and latest_comment_fullname:
                    params['before'] = latest_comment_fullname
                    logger.debug(f"Fetching comments before {latest_comment_fullname}")

                for comment in redditor.comments.new(limit=fetch_limit, params=params):
                     count += 1
                     comment_fullname = comment.fullname
                     comment_id = comment.id

                     if comment_id in processed_comment_ids:
                         logger.debug(f"Skipping already processed comment ID: {comment_id}")
                         continue

                     new_comments_data.append({
                         'id': comment_id,
                         'fullname': comment_fullname,
                         'text': comment.body[:2000], # Increased snippet
                         'score': comment.score,
                         'subreddit': comment.subreddit.display_name,
                         'permalink': f"https://www.reddit.com{comment.permalink}", # Full URL
                         'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(), # Store ISO string
                         'is_submitter': comment.is_submitter,
                         'stickied': comment.stickied,
                         'parent_id': comment.parent_id, # ID of parent comment/submission
                         'submission_id': comment.submission.id # ID of the submission thread
                     })
                     processed_comment_ids.add(comment_id)

            except prawcore.exceptions.Forbidden:
                 logger.warning(f"Access forbidden while fetching comments for u/{username}.")
            except prawcore.exceptions.RequestException as req_err:
                 # Handle potential 429 specifically
                 if hasattr(req_err, 'response') and req_err.response is not None and req_err.response.status_code == 429:
                     self._handle_rate_limit('Reddit', exception=req_err)
                     return None # Abort fetch
                 else:
                     logger.error(f"Reddit request failed fetching comments for u/{username}: {req_err}")
                     # Log and continue, maybe submissions were successful
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
            valid_new_analyses = [a for a in newly_added_media_analysis if a]
            final_media_analysis = sorted(list(set(valid_new_analyses + [m for m in existing_media_analysis if m])))
            final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))[:MAX_CACHE_ITEMS * 2]

            # --- Calculate Stats ---
            total_submissions = len(final_submissions)
            total_comments = len(final_comments)
            submissions_with_media = len([s for s in final_submissions if s.get('media')])
            # Use Decimal for potentially more accurate averages? Or float is fine.
            avg_sub_score = sum(s.get('score', 0) for s in final_submissions) / max(total_submissions, 1)
            avg_comment_score = sum(c.get('score', 0) for c in final_comments) / max(total_comments, 1)
            avg_sub_upvote_ratio = sum(s.get('upvote_ratio', 0.0) or 0.0 for s in final_submissions if s.get('upvote_ratio') is not None) / max(total_submissions, 1)

            stats = {
                'total_submissions_cached': total_submissions,
                'total_comments_cached': total_comments,
                'submissions_with_media': submissions_with_media,
                'total_media_items_processed': len(final_media_paths), # Count unique paths
                'avg_submission_score': round(avg_sub_score, 2),
                'avg_comment_score': round(avg_comment_score, 2),
                'avg_submission_upvote_ratio': round(avg_sub_upvote_ratio, 3)
            }

            # --- Prepare Final Data ---
            final_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'user_profile': redditor_info, # Add fetched profile info
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
            logger.warning(f"Reddit fetch for u/{username} aborted due to rate limit.")
            return None # Handled
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Reddit fetch failed for u/{username}: {user_err}")
             return None
        except RuntimeError as e: # Catch re-raised API/Auth errors
             logger.error(f"Runtime error during Reddit fetch for u/{username}: {e}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Reddit data for u/{username}: {str(e)}", exc_info=True)
            return None


    def fetch_bluesky(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches Bluesky posts and profile info."""
        # Use the constant for image extensions
        # supported_image_extensions = SUPPORTED_IMAGE_EXTENSIONS # Defined globally now

        cached_data = self._load_cache('bluesky', username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, 'timestamp')) < timedelta(hours=CACHE_EXPIRY_HOURS): # Use getter
            logger.info(f"Using recent cache for Bluesky user {username}")
            return cached_data

        logger.info(f"Fetching Bluesky data for {username} (Force Refresh: {force_refresh})")
        latest_post_datetime = None # Use latest datetime for comparison, as cursor/CID isn't purely sequential for incremental
        existing_posts = []
        existing_media_analysis = []
        existing_media_paths = []
        profile_info = None # Store profile info separately

        if not force_refresh and cached_data:
            logger.info(f"Attempting incremental fetch for Bluesky {username}")
            existing_posts = cached_data.get('posts', [])
            if existing_posts:
                existing_posts.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
                # Use the datetime of the latest post for comparison during fetch
                latest_post_datetime = get_sort_key(existing_posts[0], 'created_at')
                logger.debug(f"Latest known post datetime: {latest_post_datetime.isoformat() if latest_post_datetime else 'N/A'}")

            profile_info = cached_data.get('profile_info') # Keep existing profile info
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])

        try:
            # Ensure client is ready
            bsky_client = self.bluesky

            # --- Get Profile Info (only if missing or forced) ---
            if not profile_info or force_refresh:
                try:
                     profile = bsky_client.get_profile(actor=username)
                     # Extract labels if present
                     labels_list = []
                     if profile.labels:
                         labels_list = [{'value': lbl.val, 'timestamp': lbl.cts} for lbl in profile.labels]

                     profile_info = { # Store basic profile info
                          'did': profile.did,
                          'handle': profile.handle,
                          'display_name': profile.display_name,
                          'description': profile.description,
                          'avatar': profile.avatar,
                          'banner': profile.banner,
                          'followers_count': profile.followers_count,
                          'follows_count': profile.follows_count,
                          'posts_count': profile.posts_count,
                          'labels': labels_list # Store extracted labels
                      }
                     logger.debug(f"Fetched Bluesky profile info for {username}")
                except atproto_exceptions.AtProtocolError as e:
                     err_str = str(e).lower()
                     # More specific error checking based on common API responses
                     if isinstance(e, atproto_exceptions.BadRequestError) and ('profile not found' in err_str or 'could not resolve handle' in err_str):
                          raise UserNotFoundError(f"Bluesky user {username} not found or handle invalid.")
                     elif isinstance(e, atproto_exceptions.NetworkError) and ('blocked by actor' in err_str or 'blocking actor' in err_str):
                          raise AccessForbiddenError(f"Blocked from accessing Bluesky profile for {username}.")
                     # Catch auth errors specifically
                     elif isinstance(e, atproto_exceptions.UnauthorizedError):
                          logger.error(f"Bluesky authentication error fetching profile for {username}: {e}")
                          raise RuntimeError(f"Bluesky authentication failed: {e}")
                     else: # Re-raise other profile lookup errors
                          logger.error(f"Unexpected error fetching Bluesky profile for {username}: {e}")
                          raise AccessForbiddenError(f"Error fetching Bluesky profile for {username}: {e}")


            # --- Fetch New Posts ---
            new_posts_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            cursor = None
            processed_uris = set(p['uri'] for p in existing_posts) # Track existing URIs
            fetch_limit_per_page = min(INCREMENTAL_FETCH_LIMIT, 100) # Bluesky limit is 100
            total_fetched_this_run = 0
            max_fetches = INITIAL_FETCH_LIMIT if (force_refresh or not latest_post_datetime) else INCREMENTAL_FETCH_LIMIT
            reached_old_post = False

            logger.debug(f"Fetching new Bluesky posts for {username}...")
            while total_fetched_this_run < max_fetches and not reached_old_post:
                try:
                    response = bsky_client.get_author_feed(
                        actor=username, # Use handle/DID provided
                        cursor=cursor,
                        limit=fetch_limit_per_page
                    )
                except atproto_exceptions.RateLimitExceededError as rle:
                     self._handle_rate_limit('Bluesky', exception=rle)
                     return None # Rate limit handled, abort fetch
                except atproto_exceptions.AtProtocolError as e:
                    err_str = str(e).lower()
                    # More robust user not found / forbidden checks during feed fetch
                    if isinstance(e, atproto_exceptions.BadRequestError) and ('could not resolve handle' in err_str or 'profile not found' in err_str):
                         raise UserNotFoundError(f"Bluesky user {username} not found or handle cannot be resolved during feed fetch.")
                    if isinstance(e, atproto_exceptions.NetworkError) and ('blocked by actor' in err_str or 'blocking actor' in err_str):
                         raise AccessForbiddenError(f"Access to Bluesky user {username}'s feed is blocked.")
                    # Catch auth errors specifically
                    elif isinstance(e, atproto_exceptions.UnauthorizedError):
                         logger.error(f"Bluesky authentication error fetching feed for {username}: {e}")
                         raise RuntimeError(f"Bluesky authentication failed: {e}")

                    logger.error(f"Bluesky API error fetching feed for {username}: {e}")
                    return None # Stop fetching on unexpected errors

                if not response or not response.feed:
                    logger.debug("No more posts found in feed.")
                    break # No more posts

                logger.debug(f"Processing feed page with {len(response.feed)} items. Cursor: {response.cursor}")
                for feed_item in response.feed:
                    # Check for 'post' attribute, skip if missing (e.g., deleted posts might appear differently)
                    if not hasattr(feed_item, 'post'):
                        logger.debug(f"Skipping feed item without 'post' attribute: {feed_item}")
                        continue
                    post = feed_item.post
                    post_uri = post.uri # Use URI for uniqueness

                    # Avoid reprocessing the same post (API might overlap slightly)
                    if post_uri in processed_uris:
                        continue

                    # Get post creation time for incremental check
                    record = getattr(post, 'record', None)
                    if not record: continue # Skip if post has no record data (shouldn't happen often)
                    created_at_dt = get_sort_key({'created_at': getattr(record, 'created_at', None)}, 'created_at')

                    # Stop if we hit a post older than or same age as the latest known post during incremental update
                    if not force_refresh and latest_post_datetime and created_at_dt <= latest_post_datetime:
                        logger.info(f"Reached post ({post_uri} at {created_at_dt.isoformat()}) older than or same as latest known post ({latest_post_datetime.isoformat()}). Stopping incremental fetch.")
                        reached_old_post = True
                        break # Stop processing this page

                    # --- Process Media ---
                    media_items_for_post = []
                    embed = getattr(record, 'embed', None)
                    image_embeds_to_process = []
                    # Determine embed type string safely
                    embed_type_str = getattr(embed, '$type', 'unknown') if embed else None

                    if embed:
                         # Case 1: Direct images (app.bsky.embed.images)
                         if hasattr(embed, 'images'): image_embeds_to_process.extend(embed.images)
                         # Case 2: Record with media (app.bsky.embed.recordWithMedia)
                         # Check both 'media' and potential nested 'record' within the embed
                         media_embed = getattr(embed, 'media', None)
                         record_embed = getattr(embed, 'record', None) # Embed can contain a record (e.g., quote post)

                         if media_embed and hasattr(media_embed, 'images'):
                              image_embeds_to_process.extend(media_embed.images)

                         # Check inside nested record (e.g., quote post's media)
                         if record_embed:
                              nested_record_value = getattr(record_embed, 'record', None) # Check 'record' within the embedded 'record' object
                              if nested_record_value:
                                   nested_embed = getattr(nested_record_value, 'embed', None)
                                   if nested_embed and hasattr(nested_embed, 'images'):
                                        image_embeds_to_process.extend(nested_embed.images)


                    # Process collected image embeds
                    for image_info in image_embeds_to_process:
                        # Check for 'image' attribute which contains the blob
                        img_blob = getattr(image_info, 'image', None)
                        alt_text = getattr(image_info, 'alt', '') # Get alt text
                        if img_blob:
                            # Get CID (link attribute of the blob)
                            cid = getattr(img_blob, 'cid', None) or getattr(getattr(img_blob, 'ref', None), 'link', None) # Handle older 'ref' structure too

                            if cid:
                                author_did = post.author.did
                                # Construct the CDN URL (ensure DID and CID are properly quoted)
                                # Using default jpeg, could check mimeType if needed ('image/png' etc.)
                                img_mime_type = getattr(img_blob, 'mime_type', 'image/jpeg').split('/')[-1] # Get format like 'jpeg'
                                safe_author_did = quote_plus(str(author_did))
                                safe_cid = quote_plus(str(cid))
                                cdn_url = f"https://cdn.bsky.app/img/feed_fullsize/plain/{safe_author_did}/{safe_cid}@{img_mime_type}"
                                media_path = self._download_media(url=cdn_url, platform='bluesky', username=username)
                                if media_path:
                                    analysis = None
                                    if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                         analysis = self._analyze_image(media_path, f"Bluesky user {username}'s post ({post.uri})")
                                    media_items_for_post.append({
                                        'type': 'image', # Assume image for now based on processing logic
                                        'analysis': analysis,
                                        'url': cdn_url,
                                        'alt_text': alt_text,
                                        'local_path': str(media_path)
                                    })
                                    if analysis: newly_added_media_analysis.append(analysis)
                                    newly_added_media_paths.add(str(media_path))
                                else: logger.warning(f"Failed to download Bluesky image from {cdn_url} for post {post_uri}")
                            else: logger.warning(f"Could not find image CID/link in image blob for post {post_uri}")
                        else: logger.warning(f"Image embed structure missing 'image' blob for post {post_uri}")

                    # --- Extract other post details ---
                    reply_ref = getattr(record, 'reply', None)
                    reply_parent_uri = None
                    reply_root_uri = None

                    if reply_ref:
                        parent_ref = getattr(reply_ref, 'parent', None)
                        # Check if parent_ref exists and directly access its uri attribute
                        if parent_ref and hasattr(parent_ref, 'uri'):
                            reply_parent_uri = parent_ref.uri

                        root_ref = getattr(reply_ref, 'root', None)
                        # Check if root_ref exists and directly access its uri attribute
                        if root_ref and hasattr(root_ref, 'uri'):
                             reply_root_uri = root_ref.uri

                    post_langs = getattr(record, 'langs', []) # Languages detected/set

                    post_data = {
                        'uri': post_uri,
                        'cid': post.cid,
                        'author_did': post.author.did,
                        'text': getattr(record, 'text', '')[:3000], # Bluesky limit is 3000 chars
                        'created_at': created_at_dt.isoformat(), # Store ISO string
                        'langs': post_langs,
                        'reply_parent': reply_parent_uri, # Use the correctly extracted URI
                        'reply_root': reply_root_uri,     # Use the correctly extracted URI
                        'likes': getattr(post, 'like_count', 0),
                        'reposts': getattr(post, 'repost_count', 0),
                        'reply_count': getattr(post, 'reply_count', 0),
                        'embed_type': embed_type_str, # Store the $type string
                        'media': media_items_for_post
                    }

                    new_posts_data.append(post_data)
                    processed_uris.add(post_uri)
                    total_fetched_this_run += 1
                    if total_fetched_this_run >= max_fetches:
                        logger.info(f"Reached fetch limit ({max_fetches}) for Bluesky {username}.")
                        # Don't set reached_old_post here, just break the inner loop
                        break # Stop processing this page after reaching limit

                # Exit outer loop if we finished processing due to reaching old posts
                if reached_old_post:
                    break

                cursor = response.cursor
                if not cursor:
                    logger.debug("Reached end of feed (no cursor).")
                    break # No more pages

                # Check if we hit the fetch limit after processing the page
                if total_fetched_this_run >= max_fetches:
                     break # Exit outer loop


            logger.info(f"Fetched {len(new_posts_data)} new posts for Bluesky user {username}.")

            # --- Combine and Prune ---
            # De-duplicate based on URI before combining
            existing_uris_set = {p['uri'] for p in existing_posts}
            unique_new_posts = [p for p in new_posts_data if p['uri'] not in existing_uris_set]

            combined_posts = unique_new_posts + existing_posts
            # Sort again after combining
            combined_posts.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            final_posts = combined_posts[:MAX_CACHE_ITEMS]

            # Combine media analysis and paths
            valid_new_analyses = [a for a in newly_added_media_analysis if a]
            final_media_analysis = sorted(list(set(valid_new_analyses + [m for m in existing_media_analysis if m])))
            final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))[:MAX_CACHE_ITEMS * 2]

            # --- Calculate Stats ---
            total_posts = len(final_posts)
            posts_with_media = len([p for p in final_posts if p.get('media')])
            reply_posts = len([p for p in final_posts if p.get('reply_parent')])
            repost_count_sum = sum(p.get('reposts', 0) for p in final_posts) # Sum of reposts *of* these posts
            like_count_sum = sum(p.get('likes', 0) for p in final_posts)

            stats = {
                'total_posts_cached': total_posts,
                'posts_with_media': posts_with_media,
                'reply_posts_cached': reply_posts,
                'total_media_items_processed': len(final_media_paths),
                'avg_likes': round(like_count_sum / max(total_posts, 1), 2),
                'avg_reposts': round(repost_count_sum / max(total_posts, 1), 2),
                'avg_replies': round(sum(p.get('reply_count', 0) for p in final_posts) / max(total_posts, 1), 2)
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
            logger.warning(f"Bluesky fetch for {username} aborted due to rate limit.")
            return None # Handled
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Bluesky fetch failed for {username}: {user_err}")
             return None
        except RuntimeError as e: # Catch re-raised API/Auth errors
             logger.error(f"Runtime error during Bluesky fetch for {username}: {e}")
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
        # Use the constant for image extensions
        # supported_image_extensions = SUPPORTED_IMAGE_EXTENSIONS # Defined globally now

        # Basic validation for Mastodon username format
        if '@' not in cache_key_username or '.' not in cache_key_username.split('@')[1]:
             logger.error(f"Invalid Mastodon username format for fetch: '{cache_key_username}'. Needs 'user@instance.domain'.")
             # Attempt to fix if a default instance is set? No, fail explicitly here.
             # Caller (interactive/stdin) should handle correction/confirmation.
             raise ValueError(f"Invalid Mastodon username format: '{cache_key_username}'. Must be 'user@instance.domain'.")


        cached_data = self._load_cache('mastodon', cache_key_username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, 'timestamp')) < timedelta(hours=CACHE_EXPIRY_HOURS): # Use getter
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
            if existing_posts:
                 existing_posts.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
                 since_id = existing_posts[0].get('id') # Use get for safety
                 if since_id: logger.debug(f"Using since_id: {since_id}")

            user_info = cached_data.get('user_info') # Keep existing user info if available
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])

        try:
            # Ensure client is ready
            masto_client = self.mastodon

            # --- Get User Account Info ---
            # Only refetch user info if forced or missing
            if not user_info or force_refresh:
                try:
                    # account_lookup handles 'user@instance.domain'
                    # This call might fail if the instance in 'username' is different from the client's base_url
                    # and federation is blocked or the target instance doesn't allow lookups.
                    logger.debug(f"Looking up Mastodon account: {username} using client for {masto_client.api_base_url}")
                    # Use timeout specified in client
                    account = masto_client.account_lookup(acct=username)

                    # Extract fields, converting datetime to ISO string
                    created_at_dt = account.get('created_at')
                    created_at_iso = created_at_dt.isoformat() if isinstance(created_at_dt, datetime) else str(created_at_dt)

                    # Extract custom fields (profile metadata)
                    custom_fields = []
                    if account.get('fields'):
                         custom_fields = [{'name': f.get('name'), 'value': f.get('value')} for f in account['fields']]

                    user_info = {
                        'id': str(account['id']), # Ensure ID is string
                        'username': account['username'], # Local username part
                        'acct': account['acct'], # Full handle (user@instance or just user)
                        'display_name': account['display_name'],
                        'note_html': account.get('note', ''), # Bio (HTML)
                        'note_text': BeautifulSoup(account.get('note', ''), 'html.parser').get_text(separator=' ', strip=True), # Cleaned bio
                        'url': account['url'], # Link to profile page
                        'avatar': account['avatar'],
                        'header': account['header'],
                        'locked': account.get('locked', False), # Check if account is locked/private
                        'bot': account.get('bot', False),
                        'discoverable': account.get('discoverable'),
                        'group': account.get('group', False),
                        'followers_count': account['followers_count'],
                        'following_count': account['following_count'],
                        'statuses_count': account['statuses_count'],
                        'last_status_at': account.get('last_status_at'), # Can be None or datetime
                        'created_at': created_at_iso,
                        'custom_fields': custom_fields
                    }
                    logger.info(f"Fetched Mastodon user info for {cache_key_username}")
                    if user_info['locked']:
                         logger.warning(f"Mastodon user {cache_key_username} has a locked/private account. Status fetch might be limited or fail.")
                         # raise AccessForbiddenError(f"Mastodon user {cache_key_username} account is locked/private.") # Optionally block here

                except MastodonNotFoundError:
                    raise UserNotFoundError(f"Mastodon user {username} not found via {masto_client.api_base_url}.")
                except MastodonUnauthorizedError:
                    # Might happen for locked accounts you don't follow or instance restrictions
                    raise AccessForbiddenError(f"Unauthorized access to Mastodon user {username}'s info (locked account / instance policy?).")
                # MastodonVersionError can indicate various issues (block, federation)
                except (MastodonVersionError, MastodonError) as e:
                    logger.error(f"Mastodon API error looking up {username}: {e}")
                    # Check for common messages indicating blocking or federation issues
                    err_str = str(e).lower()
                    if 'blocked' in err_str or 'forbidden' in err_str or 'federation' in err_str:
                          raise AccessForbiddenError(f"Forbidden/Blocked from accessing Mastodon user {username}'s info (possibly blocked by user/instance or federation issue).")
                    else:
                         raise RuntimeError(f"Mastodon API error during user lookup: {e}")

            user_id = user_info['id'] # Use string ID

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
                # Paginate if needed (Mastodon.py handles pagination within the limit)
                # Fetching slightly more than api_limit might trigger pagination if available
                new_statuses = masto_client.account_statuses(
                    id=user_id,
                    limit=api_limit, # Fetch up to the API limit per request
                    since_id=since_id if not force_refresh else None,
                    exclude_replies=False, # Include replies
                    exclude_reblogs=False   # Include boosts
                )
                # Note: This fetches only one page up to api_limit.
                # For very active users and long gaps, multiple pages might be needed.
                # Implementing full pagination would require looping with max_id.
                # For now, assume INCREMENTAL_FETCH_LIMIT <= MASTODON_FETCH_LIMIT handles most cases.
                if len(new_statuses) == api_limit:
                     logger.warning(f"Reached Mastodon API limit ({api_limit}) in a single fetch for {cache_key_username}. Some newer posts might be missed if >{api_limit} were posted since last check.")

            except MastodonRatelimitError as e:
                 self._handle_rate_limit('Mastodon', exception=e)
                 return None # Abort fetch
            except MastodonNotFoundError: # Should not happen if account lookup succeeded, but maybe account deleted between calls
                 raise UserNotFoundError(f"Mastodon user ID {user_id} (handle: {username}) not found during status fetch.")
            except (MastodonUnauthorizedError, MastodonVersionError, MastodonError) as e:
                 # Check if due to locked account
                 err_str = str(e).lower()
                 if user_info and user_info.get('locked') and ('unauthorized' in err_str or 'forbidden' in err_str):
                      logger.warning(f"Cannot fetch statuses for locked Mastodon account {username}.")
                      # Return current data (profile info only) or None? Let's return profile info.
                      # Skip status fetching part, proceed to save cache with just profile.
                      new_statuses = [] # Ensure status list is empty
                 elif 'blocked' in err_str or 'forbidden' in err_str or 'federation' in err_str:
                      raise AccessForbiddenError(f"Access forbidden/blocked fetching Mastodon statuses for {username}.")
                 else:
                      logger.error(f"Error fetching statuses for {username}: {e}")
                      raise RuntimeError(f"Mastodon API error during status fetch: {e}")

            logger.info(f"Fetched {len(new_statuses)} new raw statuses for Mastodon user {cache_key_username}.")

            # --- Process New Statuses ---
            count_added = 0
            for status in new_statuses:
                status_id = str(status['id']) # Ensure ID is string
                # Skip if already processed (safeguard against API overlap or since_id issues)
                if status_id in processed_status_ids:
                    logger.debug(f"Skipping already processed Mastodon status ID: {status_id}")
                    continue

                # Clean HTML content using BeautifulSoup
                # Handle content warning - store cleaned CW text and indicate content is hidden
                cleaned_text = ''
                is_content_warning = bool(status.get('spoiler_text'))
                if is_content_warning:
                     # Clean the spoiler text itself
                     try:
                         cw_soup = BeautifulSoup(status['spoiler_text'], 'html.parser')
                         cleaned_cw_text = cw_soup.get_text(separator=' ', strip=True)
                     except Exception as parse_err:
                         logger.warning(f"HTML parsing failed for CW text of status {status_id}: {parse_err}.")
                         cleaned_cw_text = status['spoiler_text'] # Fallback
                     cleaned_text = f"[CW: {cleaned_cw_text}] [Content Hidden]"
                else:
                    # Parse main content if no CW
                    try:
                         soup = BeautifulSoup(status['content'], 'html.parser')
                         # Improve text extraction (preserve paragraphs, links)
                         # Remove <style> and <script> tags first
                         for script_or_style in soup(["script", "style"]):
                             script_or_style.decompose()
                         # Get text, preserving line breaks from <p> and <br> somewhat
                         # Replace <br> with newline
                         for br in soup.find_all("br"): br.replace_with("\n")
                         # Add newline after paragraphs
                         for p in soup.find_all("p"): p.append("\n")
                         # Extract text, strip leading/trailing whitespace from lines, join with space
                         lines = (line.strip() for line in soup.get_text().splitlines())
                         cleaned_text = "\n".join(line for line in lines if line) # Reconstruct with single newlines
                    except Exception as parse_err:
                         logger.warning(f"HTML parsing failed for status {status_id}: {parse_err}. Using raw content snippet.")
                         cleaned_text = status['content'][:500] + "..." # Fallback snippet


                # --- Process Media ---
                media_items_for_post = []
                for attachment in status.get('media_attachments', []):
                     media_url = attachment.get('url') # Full resolution URL
                     preview_url = attachment.get('preview_url') # Lower res preview
                     media_type = attachment.get('type', 'unknown') # image, video, gifv, audio
                     description = attachment.get('description') # Alt text
                     remote_url = attachment.get('remote_url') # URL on original instance (if remote)

                     # Prefer full URL for download if available
                     url_to_download = media_url or preview_url
                     if url_to_download:
                         media_path = self._download_media(url=url_to_download, platform='mastodon', username=cache_key_username)
                         if media_path:
                             analysis = None
                             # Only analyze supported image types
                             if media_type == 'image' and media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                 # Pass context including the status URL for reference
                                 image_context = f"Mastodon user {cache_key_username}'s post ({status.get('url', status_id)})"
                                 analysis = self._analyze_image(media_path, image_context)

                             media_items_for_post.append({
                                 'id': str(attachment.get('id')), # Ensure string ID
                                 'type': media_type,
                                 'analysis': analysis,
                                 'url': media_url, # Store original full URL if available
                                 'preview_url': preview_url,
                                 'remote_url': remote_url, # Store remote URL if different
                                 'description': description, # Alt text
                                 'local_path': str(media_path)
                             })
                             if analysis: newly_added_media_analysis.append(analysis)
                             newly_added_media_paths.add(str(media_path))
                         else:
                              logger.warning(f"Failed to download Mastodon media {media_type} from {url_to_download} for status {status_id}")

                # --- Process Other Fields ---
                is_reblog = status.get('reblog') is not None
                reblog_info = status.get('reblog') if is_reblog else None
                reblog_original_author_acct = None
                reblog_original_url = None
                if reblog_info:
                     # Ensure reblog account info is present
                     reblog_acct_info = reblog_info.get('account')
                     if reblog_acct_info:
                         reblog_original_author_acct = reblog_acct_info.get('acct')
                     reblog_original_url = reblog_info.get('url')


                # Extract tags, mentions, emojis
                tags = [{'name': tag['name'], 'url': tag['url']} for tag in status.get('tags', [])]
                mentions = [{'acct': mention['acct'], 'url': mention['url']} for mention in status.get('mentions', [])]
                emojis = [{'shortcode': emoji['shortcode'], 'url': emoji['url']} for emoji in status.get('emojis', [])]

                # Poll info
                poll_data = None
                if status.get('poll'):
                     poll = status['poll']
                     poll_options = [{'title': opt['title'], 'votes_count': opt.get('votes_count')} for opt in poll.get('options', [])]
                     poll_data = {
                         'id': str(poll.get('id')),
                         'expires_at': poll.get('expires_at'), # Datetime or None
                         'expired': poll.get('expired'),
                         'multiple': poll.get('multiple'),
                         'votes_count': poll.get('votes_count'),
                         'voters_count': poll.get('voters_count'), # May be None
                         'options': poll_options
                     }


                post_data = {
                    'id': status_id,
                    'created_at': status['created_at'].isoformat(), # Store ISO string
                    'url': status['url'], # Link to the status on its home instance
                    'in_reply_to_id': str(status.get('in_reply_to_id')) if status.get('in_reply_to_id') else None,
                    'in_reply_to_account_id': str(status.get('in_reply_to_account_id')) if status.get('in_reply_to_account_id') else None,
                    'text_html': status['content'], # Raw HTML content
                    'text_cleaned': cleaned_text[:3000], # Store cleaned, truncated text
                    'spoiler_text': status.get('spoiler_text', ''),
                    'visibility': status.get('visibility'), # public, unlisted, private, direct
                    'sensitive': status.get('sensitive', False),
                    'language': status.get('language'),
                    'reblogs_count': status.get('reblogs_count', 0),
                    'favourites_count': status.get('favourites_count', 0),
                    'replies_count': status.get('replies_count', 0),
                    'is_reblog': is_reblog,
                    'reblog_original_author': reblog_original_author_acct,
                    'reblog_original_url': reblog_original_url,
                    'tags': tags,
                    'mentions': mentions,
                    'emojis': emojis,
                    'poll': poll_data,
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
            valid_new_analyses = [a for a in newly_added_media_analysis if a]
            final_media_analysis = sorted(list(set(valid_new_analyses + [m for m in existing_media_analysis if m])))
            final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))[:MAX_CACHE_ITEMS * 2]

            # --- Calculate Stats ---
            total_posts = len(final_posts)
            original_posts = [p for p in final_posts if not p.get('is_reblog')]
            total_original_posts = len(original_posts)
            total_reblogs = total_posts - total_original_posts
            posts_with_media = len([p for p in final_posts if p.get('media')])
            reply_posts_count = len([p for p in final_posts if p.get('in_reply_to_id')])

            # Calculate stats only on original posts to avoid inflating interaction counts from boosts
            avg_favs = sum(p['favourites_count'] for p in original_posts) / max(total_original_posts, 1)
            avg_reblogs = sum(p['reblogs_count'] for p in original_posts) / max(total_original_posts, 1)
            avg_replies = sum(p['replies_count'] for p in original_posts) / max(total_original_posts, 1)

            stats = {
                'total_posts_cached': total_posts,
                'total_original_posts_cached': total_original_posts,
                'total_reblogs_cached': total_reblogs,
                'total_replies_cached': reply_posts_count, # Replies made *by* the user in cache
                'posts_with_media': posts_with_media,
                'total_media_items_processed': len(final_media_paths),
                'avg_favourites_on_originals': round(avg_favs, 2),
                'avg_reblogs_on_originals': round(avg_reblogs, 2),
                'avg_replies_on_originals': round(avg_replies, 2) # Replies *received* on user's original posts
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

        except ValueError as ve: # Catch the format validation error from the top
             logger.error(f"Mastodon fetch failed for {username}: {ve}")
             return None
        except RateLimitExceededError:
            logger.warning(f"Mastodon fetch for {username} aborted due to rate limit.")
            return None # Handled
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Mastodon fetch failed for {username}: {user_err}")
             return None
        except RuntimeError as e: # Catch re-raised API/Auth errors
             logger.error(f"Runtime error during Mastodon fetch for {username}: {e}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Mastodon data for {username}: {str(e)}", exc_info=True)
            return None
    # --- End Mastodon Fetcher ---


    def fetch_hackernews(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches Hacker News items (stories, comments) via Algolia API."""
        cached_data = self._load_cache('hackernews', username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, 'timestamp')) < timedelta(hours=CACHE_EXPIRY_HOURS): # Use getter
            logger.info(f"Using recent cache for HackerNews user {username}")
            return cached_data

        logger.info(f"Fetching HackerNews data for {username} (Force Refresh: {force_refresh})")
        latest_timestamp_i = 0 # Algolia uses integer timestamps
        existing_submissions = [] # Renaming to 'items' might be clearer
        existing_items = []

        if not force_refresh and cached_data:
            logger.info(f"Attempting incremental fetch for HackerNews {username}")
            existing_items = cached_data.get('items', []) # Use 'items' key
            if existing_items:
                 existing_items.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
                 # Find the max timestamp_i from the existing data
                 try:
                     latest_timestamp_i = max(item.get('created_at_i', 0) for item in existing_items)
                     logger.debug(f"Using latest timestamp_i: {latest_timestamp_i}")
                 except ValueError: # Handle empty list case after filtering Nones
                     logger.debug("No valid existing items found to determine latest timestamp_i.")
                     latest_timestamp_i = 0


        try:
            # Algolia API endpoint
            base_url = "https://hn.algolia.com/api/v1/search"
            # Fetch more on initial/forced, use incremental limit otherwise
            hits_per_page = INITIAL_FETCH_LIMIT if (force_refresh or not latest_timestamp_i) else INCREMENTAL_FETCH_LIMIT
            # Ensure username is URL encoded for the tag query
            safe_username = quote_plus(username)
            params = {
                "tags": f"author_{safe_username}",
                "hitsPerPage": hits_per_page,
                "typoTolerance": False # Be strict with username matching
            }

            # Add numeric filter for incremental fetch - fetch items created *after* the latest known timestamp
            if not force_refresh and latest_timestamp_i > 0:
                params["numericFilters"] = f"created_at_i>{latest_timestamp_i}"
                logger.debug(f"Applying numeric filter: created_at_i > {latest_timestamp_i}")

            new_items_data = []
            processed_ids = set(item['objectID'] for item in existing_items) # Track existing IDs

            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                 response = client.get(base_url, params=params)
                 # Check for specific Algolia errors before raising generic status error
                 if response.status_code == 400:
                      try:
                           err_data = response.json()
                           if 'message' in err_data:
                                err_msg = err_data['message']
                                # Check for common Algolia 400 reasons like invalid tag format
                                if 'invalid tag name' in err_msg.lower():
                                     logger.error(f"HN Algolia API Bad Request (400): Invalid tag name likely due to username format '{username}'. Error: {err_msg}")
                                     raise UserNotFoundError(f"HackerNews username '{username}' resulted in an invalid API tag (check format).")
                                else:
                                     logger.error(f"HN Algolia API Bad Request (400) for {username}: {err_msg}")
                                     raise httpx.HTTPStatusError(err_msg, request=response.request, response=response) # Re-raise generic if not tag error
                           else:
                                response.raise_for_status() # Raise generic if no message found
                      except json.JSONDecodeError:
                           response.raise_for_status() # Raise generic if response isn't JSON
                 else:
                      response.raise_for_status() # Check for other HTTP errors (e.g., 429, 5xx)

                 data = response.json()

            if 'hits' not in data:
                 logger.warning(f"No 'hits' found in HN Algolia response for {username}")
                 data['hits'] = []

            # Check if Algolia thinks the user exists but returned 0 hits
            # nbHits = 0 does not guarantee user non-existence, they might just have 0 items matching query
            if not data['hits']:
                 # To confirm user existence, we'd need a separate call to the official HN API (if one exists for users)
                 # or rely on the fact that a query for a non-existent user *usually* returns 0 hits quickly.
                 logger.info(f"HN Algolia query for '{username}' returned 0 new items. User might have no recent activity or might not exist.")


            logger.info(f"Fetched {len(data.get('hits', []))} potential new items for HN user {username}.")

            for hit in data.get('hits', []):
                object_id = hit.get('objectID')
                # Basic validation of hit data
                if not object_id or not hit.get('created_at_i'):
                     logger.warning(f"Skipping invalid HN hit (missing ID or timestamp): {hit.get('title', object_id)}")
                     continue
                if object_id in processed_ids:
                     logger.debug(f"Skipping already processed HN item ID: {object_id}")
                     continue # Skip duplicates

                created_at_ts = hit['created_at_i']

                tags = hit.get('_tags', [])
                item_type = 'unknown'
                # Determine type more reliably
                if 'story' in tags and 'comment' not in tags: item_type = 'story'
                elif 'comment' in tags: item_type = 'comment'
                elif 'poll' in tags and 'pollopt' not in tags: item_type = 'poll'
                elif 'pollopt' in tags: item_type = 'pollopt'
                else:
                     # Infer based on fields present
                     if hit.get('title') and hit.get('url') and 'comment' not in tags: item_type = 'story' # Link post
                     elif hit.get('title') and not hit.get('url') and 'comment' not in tags: item_type = 'ask_hn_or_job' # Ask HN / Job etc.
                     elif hit.get('comment_text') and 'comment' not in tags: item_type = 'comment' # Should have tag, but fallback

                # Clean up text (remove HTML tags) if present
                raw_text = hit.get('story_text') or hit.get('comment_text') or ''
                cleaned_text = ''
                if raw_text:
                     try:
                          soup = BeautifulSoup(raw_text, 'html.parser')
                          # Simple text extraction, might want more sophisticated cleaning
                          cleaned_text = soup.get_text(separator=' ', strip=True)
                     except Exception as parse_err:
                          logger.warning(f"HTML parsing failed for HN item {object_id}: {parse_err}. Using raw snippet.")
                          cleaned_text = raw_text[:500] + "..."

                # Get parent author if it's a comment
                parent_author = None
                if item_type == 'comment':
                    parent_id = hit.get('parent_id')
                    # We can't easily get the parent author from Algolia, would need another API call
                    # For now, just store parent ID.

                item_data = {
                    'objectID': object_id,
                    'type': item_type,
                    'title': hit.get('title'), # Present for stories, polls, ask_hn
                    'url': hit.get('url'), # Present for link stories
                    'points': hit.get('points'),
                    'num_comments': hit.get('num_comments'), # Relevant for stories/polls
                    'story_id': hit.get('story_id'), # ID of the parent story for comments/pollopts
                    'parent_id': hit.get('parent_id'), # ID of the parent item (story or comment) for comments
                    'created_at_i': created_at_ts,
                    'created_at': datetime.fromtimestamp(created_at_ts, tz=timezone.utc).isoformat(), # Store ISO string
                    'text': cleaned_text # Use cleaned text
                }
                new_items_data.append(item_data)
                processed_ids.add(object_id)

            # --- Combine and Prune ---
            # New data is already unique due to ID check
            combined_items = new_items_data + existing_items
            # Sort combined list before pruning
            combined_items.sort(key=lambda x: get_sort_key(x, 'created_at'), reverse=True) # Use global helper
            final_items = combined_items[:MAX_CACHE_ITEMS]

            # --- Calculate Stats ---
            story_items = [s for s in final_items if s['type'] == 'story' or s['type'] == 'ask_hn_or_job']
            comment_items = [c for c in final_items if c['type'] == 'comment']
            poll_items = [p for p in final_items if p['type'] == 'poll']
            total_items = len(final_items)
            total_stories = len(story_items)
            total_comments = len(comment_items)
            total_polls = len(poll_items)

            # Calculate averages safely, check for points being None
            avg_story_pts = sum(s.get('points', 0) or 0 for s in story_items) / max(total_stories, 1)
            avg_story_num_comments = sum(s.get('num_comments', 0) or 0 for s in story_items) / max(total_stories, 1)
            avg_comment_pts = sum(c.get('points', 0) or 0 for c in comment_items) / max(total_comments, 1)

            stats = {
                'total_items_cached': total_items,
                'total_stories_cached': total_stories, # Includes Ask/Job
                'total_comments_cached': total_comments,
                'total_polls_cached': total_polls,
                'average_story_points': round(avg_story_pts, 2),
                'average_story_num_comments': round(avg_story_num_comments, 2),
                'average_comment_points': round(avg_comment_pts, 2)
            }

            # --- Prepare Final Data ---
            final_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'items': final_items, # Use 'items' key consistently
                'stats': stats
            }

            self._save_cache('hackernews', username, final_data)
            logger.info(f"Successfully updated HackerNews cache for {username}. Total items cached: {total_items}")
            return final_data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._handle_rate_limit('HackerNews (Algolia)', e)
            # 400 Bad Request handled above for specific tag errors
            elif e.response.status_code >= 500:
                 logger.error(f"HN Algolia API server error ({e.response.status_code}) for {username}: {e.response.text[:200]}")
            else:
                 logger.error(f"HN Algolia API HTTP error for {username}: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except httpx.RequestError as e:
             logger.error(f"HN Algolia API network error for {username}: {str(e)}")
             return None
        except UserNotFoundError as e: # Re-raise specific UserNotFound if detected above
             logger.error(f"HN fetch failed for {username}: {e}")
             return None # Return None, error logged
        except Exception as e:
            logger.error(f"Unexpected error fetching HackerNews data for {username}: {str(e)}", exc_info=True)
            return None


    # --- Analysis Core ---

    def analyze(self, platforms: Dict[str, Union[str, List[str]]], query: str) -> str:
        """Collects data (using fetch methods) and performs LLM analysis."""
        collected_text_summaries = []
        all_media_analyzes = []
        failed_fetches = []
        analysis_start_time = datetime.now(timezone.utc)

        platform_targets = {} # Store successfully fetched targets per platform
        targets_to_process = [] # List of (platform, username, display_name) tuples

        for platform, usernames in platforms.items():
            if isinstance(usernames, str): usernames = [usernames]
            if not usernames: continue
            for username in usernames:
                 # Construct display name based on platform convention
                 display_name = username # Default
                 if platform == 'twitter': display_name = f"@{username}"
                 elif platform == 'reddit': display_name = f"u/{username}"
                 elif platform == 'mastodon': display_name = username # user@instance.domain
                 elif platform == 'bluesky': display_name = username # handle.bsky.social or DID
                 elif platform == 'hackernews': display_name = username

                 targets_to_process.append((platform, username, display_name))

        if not targets_to_process:
             return "[yellow]No valid platforms or users specified for analysis.[/yellow]"

        total_targets = len(targets_to_process)
        collect_task = None # Initialize task ID

        try:
             # Start progress bar for collection phase
            with self.progress: # Use Progress as a context manager
                collect_task = self.progress.add_task(
                    f"[cyan]Collecting data for {total_targets} target(s)...",
                    total=total_targets
                )

                for platform, username, display_name in targets_to_process:
                    fetcher = getattr(self, f'fetch_{platform}', None)
                    if not fetcher:
                         logger.warning(f"No fetcher method found for platform: {platform}")
                         failed_fetches.append((platform, display_name, "Fetcher not implemented"))
                         self.progress.advance(collect_task) # Advance progress for skipped user
                         continue

                    task_desc = f"[cyan]Fetching {platform} for {display_name}..."
                    self.progress.update(collect_task, description=task_desc)
                    data = None # Reset data for each target
                    try:
                        # Use force_refresh=False for analysis calls unless explicitly needed later
                        # Individual fetchers handle their specific exceptions (UserNotFound, Forbidden, RateLimit)
                        data = fetcher(username=username, force_refresh=False)

                        if data:
                            # Format data immediately after fetching
                            summary = self._format_text_data(platform, username, data)
                            if summary: # Check if summary formatting was successful
                                 collected_text_summaries.append(summary)
                                 # Store successful target
                                 if platform not in platform_targets: platform_targets[platform] = []
                                 platform_targets[platform].append(username) # Store original username used for fetch
                            else:
                                 logger.warning(f"Failed to format data summary for {platform}/{display_name}. Skipping.")
                                 failed_fetches.append((platform, display_name, "Data formatting failed"))

                            # Collect media analysis only from successful fetches with data
                            # Filter out None/empty strings immediately
                            media_analyses = [ma for ma in data.get('media_analysis', []) if isinstance(ma, str) and ma.strip()]
                            if media_analyses:
                                 all_media_analyzes.extend(media_analyses)
                            logger.info(f"Successfully collected and formatted data for {platform}/{display_name}")
                        else:
                            # Fetcher returned None, implying failure handled internally (rate limit, not found etc)
                             # Add to failed_fetches ONLY if it wasn't already handled by specific exceptions below
                             # Assume fetcher logged the specific reason if it returned None without raising UserNotFound/AccessForbidden/RateLimitExceeded
                             # We'll capture the specific errors below.
                             # If data is None and no exception caught, assume it was a handled failure (like rate limit)
                             # The specific reason is logged by the fetcher or rate limit handler.
                             if not any(f[0] == platform and f[1] == display_name for f in failed_fetches):
                                  failed_fetches.append((platform, display_name, "Data fetch failed (check logs)"))
                             logger.warning(f"Data fetch returned None for {platform}/{display_name}")

                    except RateLimitExceededError as rle:
                        # Error already logged and printed by _handle_rate_limit
                        failed_fetches.append((platform, display_name, f"Rate Limited")) # Simpler message
                        # No further action needed here, loop continues
                    except (UserNotFoundError, AccessForbiddenError) as afe:
                         # These errors are logged by the fetchers
                         failed_fetches.append((platform, display_name, f"Access Error ({type(afe).__name__})"))
                         self.console.print(f"[yellow]Skipping {platform}/{display_name}: {afe}[/yellow]", highlight=False)
                         # No further action needed here, loop continues
                    except RuntimeError as rte:
                         # Catch setup/auth errors raised during fetcher execution
                         failed_fetches.append((platform, display_name, f"Runtime Error ({rte})"))
                         logger.error(f"Runtime error during fetch for {platform}/{display_name}: {rte}", exc_info=False) # Log without full trace maybe
                         self.console.print(f"[red]Runtime Error fetching {platform}/{display_name}: {rte}[/red]", highlight=False)
                    except Exception as e:
                        # Catch unexpected errors during fetch call
                        fetch_error_msg = f"Unexpected error during fetch for {platform}/{display_name}: {e}"
                        logger.error(fetch_error_msg, exc_info=True)
                        failed_fetches.append((platform, display_name, "Unexpected fetch error"))
                        self.console.print(f"[red]Error fetching {platform}/{display_name}: {e}[/red]", highlight=False)
                    finally:
                         # Ensure progress advances even if username processing fails inside loop
                         if collect_task is not None and collect_task in self.progress.task_ids:
                              self.progress.advance(collect_task)

            # --- Post-Collection Reporting ---
            # Progress bar context manager handles stopping

            # Report Failed Fetches clearly
            if failed_fetches:
                self.console.print("\n[bold yellow]Data Collection Issues:[/bold yellow]")
                # Group failures by reason for clarity
                failures_by_reason = {}
                for pf, user, reason in failed_fetches:
                    if reason not in failures_by_reason: failures_by_reason[reason] = []
                    failures_by_reason[reason].append(f"{pf}/{user}")

                for reason, targets in failures_by_reason.items():
                    self.console.print(f"- {reason}: {', '.join(targets)}")
                self.console.print("[yellow]Analysis will proceed with available data.[/yellow]\n")

            # --- Prepare for LLM Analysis ---
            if not collected_text_summaries and not all_media_analyzes:
                 # Check if *any* data was fetched, even if formatting failed or media analysis was empty
                 if not platform_targets and failed_fetches:
                     return "[red]Data collection failed for all targets. Analysis cannot proceed.[/red]"
                 elif not platform_targets:
                     # This case should be rare if targets_to_process was not empty
                     return "[red]No data successfully collected or formatted. Analysis cannot proceed.[/red]"
                 else:
                      # Data was collected but text summaries or media analyzes are empty
                      logger.warning("Proceeding to analysis with potentially limited data (no text summaries or no media analyzes).")
                      # Fall through to analysis, prompt will be constructed with available data

            # De-duplicate media analysis strings using a set
            unique_media_analyzes = sorted(list(set(all_media_analyzes))) # Already filtered for non-empty strings

            analysis_components = []
            image_model = os.getenv('IMAGE_ANALYSIS_MODEL', 'google/gemini-pro-vision') # Get model used
            text_model = os.getenv('ANALYSIS_MODEL', 'mistralai/mixtral-8x7b-instruct') # Get text model

            # Add Media Analysis Section (if any)
            if unique_media_analyzes:
                 media_summary = f"## Consolidated Media Analysis (using {image_model}):\n\n"
                 media_summary += "*Note: The following are objective descriptions based on visual content analysis.*\n\n"
                 # Ensure analysis strings are stripped of extra whitespace before joining
                 media_summary += "\n\n".join(f"### Image Analysis {i+1}\n{analysis.strip()}" for i, analysis in enumerate(unique_media_analyzes)) # Use H3 for each image
                 analysis_components.append(media_summary)
                 logger.debug(f"Added {len(unique_media_analyzes)} unique media analyzes to prompt.")
            else:
                 logger.debug("No media analysis results to add to prompt.")


            # Add Text Data Section (if any)
            if collected_text_summaries:
                 text_summary = f"## Collected Textual & Activity Data Summary:\n\n"
                 # Ensure summaries are stripped before joining
                 text_summary += "\n\n---\n\n".join([s.strip() for s in collected_text_summaries]) # Separate platforms clearly
                 analysis_components.append(text_summary)
                 logger.debug(f"Added {len(collected_text_summaries)} platform text summaries to prompt.")
            else:
                 logger.debug("No text summaries to add to prompt.")

            # Construct the final prompt if there's anything to analyze
            if not analysis_components:
                 return "[yellow]No text or media data available to send for analysis after collection phase.[/yellow]"


            system_prompt = """**Objective:** Generate a comprehensive behavioral and linguistic profile based on the provided social media data, employing structured analytic techniques focused on objectivity, evidence-based reasoning, and clear articulation.

**Input:** You will receive summaries of user activity (text posts, engagement metrics, descriptive analyzes of images shared) from platforms like Twitter, Reddit, Bluesky, Mastodon, and Hacker News for one or more specified users. The user will also provide a specific analysis query. You may also receive consolidated analyzes of images shared by the user(s).

**Primary Task:** Address the user's specific analysis query using ALL the data provided (text summaries AND image analyzes if available) and the analytical framework below.

**Analysis Domains (Use these to structure your thinking and response where relevant to the query):**
1.  **Behavioral Patterns:** Analyze interaction frequency, platform-specific activity (e.g., retweets vs. posts, submissions vs. comments, boosts vs. original posts), potential engagement triggers, and temporal communication rhythms apparent *in the provided data*. Note differences across platforms if multiple are present. Note visibility settings (e.g., Mastodon).
2.  **Semantic Content & Themes:** Identify recurring topics, keywords, and concepts. Analyze linguistic indicators such as expressed sentiment/tone (positive, negative, neutral, specific emotions if clear), potential ideological leanings *if explicitly stated or strongly implied by language/topics*, and cognitive framing (how subjects are discussed). Assess information source credibility *only if* the user shares external links/content within the provided data AND you can evaluate the source based on common knowledge. Note use of content warnings/spoilers. Identify languages used (e.g., from Bluesky/Mastodon data).
3.  **Interests & Network Context:** Deduce primary interests, hobbies, or professional domains suggested by post content and image analysis. Note any interaction patterns visible *within the provided posts* (e.g., frequent replies to specific user types, retweets/boosts of particular accounts, participation in specific communities like subreddits or Mastodon hashtags/local timelines if mentioned). Look for profile metadata clues (e.g., Mastodon custom fields). Avoid inferring broad influence or definitive group membership without strong evidence.
4.  **Communication Style:** Assess linguistic complexity (simple/complex vocabulary, sentence structure), use of jargon/slang, rhetorical strategies (e.g., humor, sarcasm, argumentation), markers of emotional expression (e.g., emoji use, exclamation points, emotionally charged words), and narrative consistency across platforms. Note use of HTML/rich text formatting (e.g., in Mastodon) or markdown (Reddit/HN).
5.  **Visual Data Integration:** Explicitly incorporate insights derived from the provided image analyzes, *if available*. How do the visual elements (settings, objects, activities depicted) complement, contradict, or add context to the textual data? Note any patterns in the *types* of images shared (photos, screenshots, art) or use of alt text. If no image analysis is provided, state that visual context is missing.

**Analytical Constraints & Guidelines:**
*   **Evidence-Based:** Ground ALL conclusions *strictly and exclusively* on the provided source materials (text summaries AND image analyzes). Reference specific examples or patterns from the data (e.g., "Frequent posts about [topic] on Reddit," "Image analysis of setting suggests [environment]," "Consistent use of technical jargon on HackerNews", "Use of spoiler tags on Mastodon for [topic]").
*   **Objectivity & Neutrality:** Maintain analytical neutrality. Avoid speculation, moral judgments, personal opinions, or projecting external knowledge not present in the data. Focus on describing *what the data shows*.
*   **Synthesize, Don't Just List:** Integrate findings from different platforms and data types (text/image) into a coherent narrative that addresses the query. Highlight correlations or discrepancies.
*   **Address the Query Directly:** Structure your response primarily around answering the user's specific question(s). Use the analysis domains as tools to build your answer.
*   **Acknowledge Limitations:** If the data is sparse, lacks specific details needed for the query, only covers a short time period, or if certain data types (e.g., images) were unavailable/unprocessed, explicitly state these limitations (e.g., "Based on the limited posts available...", "Image analysis was not performed or available", "Mastodon data includes boosts and replies...", "Data only reflects recent activity up to N items"). Do not invent information. Mention if data collection failed for any targets.
*   **Clarity & Structure:** Use clear language. Employ formatting (markdown headings, bullet points) to organize the response logically, often starting with a direct answer to the query followed by supporting evidence/analysis.

**Output:** A structured analytical report that directly addresses the user's query, rigorously supported by evidence from the provided text and image data, adhering to all constraints. Start with a summary answer, then elaborate with details structured using relevant analysis domains. If data collection failed for some targets, mention this early on.
"""
            user_prompt = f"**Analysis Query:** {query}\n\n" \
                          f"**Provided Data:**\n\n" + \
                          "\n\n===\n\n".join(analysis_components) # Use a very clear separator

            # --- Call OpenRouter LLM ---
            analysis_task = None
            # Use progress context manager for the analysis call spinner
            with self.progress:
                 analysis_task = self.progress.add_task(f"[magenta]Analyzing with {text_model}...", total=None)

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
                                "temperature": 0.5, # Keep relatively low for factual analysis
                                # "stream": False # Streaming disabled for now
                            }
                        }
                    )
                    api_thread.start()

                    # Wait for the thread to finish while keeping the progress spinner active
                    while api_thread.is_alive():
                        api_thread.join(0.1) # Short join timeout to allow spinner update
                        # No need to update description constantly, spinner indicates activity

                    # Thread finished, check results

                    if self._analysis_exception:
                        err_details = str(self._analysis_exception)
                        # Improve error reporting for HTTPStatusError
                        if isinstance(self._analysis_exception, httpx.HTTPStatusError):
                             err_code = self._analysis_exception.response.status_code
                             err_details = f"API HTTP {err_code}"
                             response_text = self._analysis_exception.response.text[:500] # Limit log size
                             logger.error(f"Analysis API Error Response ({err_code}): {response_text}")
                             err_details += " (See analyzer.log for more)"
                             try: # Try to get cleaner message from JSON body
                                 error_data = self._analysis_exception.response.json()
                                 if 'error' in error_data and 'message' in error_data['error']:
                                     err_details = f"API Error: {error_data['error']['message']}"
                             except (json.JSONDecodeError, KeyError, AttributeError, TypeError): pass # Use status code if not JSON or format unknown

                        # Re-raise as a runtime error to be caught by the outer handler
                        raise RuntimeError(f"Analysis API request failed: {err_details}") from self._analysis_exception

                    if not self._analysis_response:
                         # This case should ideally not happen if the exception wasn't raised, but good failsafe
                         raise RuntimeError("Analysis API call completed but no response object was captured.")

                    # Process successful response (outside thread)
                    response = self._analysis_response
                    response.raise_for_status() # Should have been checked in thread, but double check

                    response_data = response.json()
                    # Check response structure more carefully
                    choice = response_data.get('choices', [{}])[0]
                    message = choice.get('message', {})
                    analysis_content = message.get('content')

                    if not analysis_content:
                         logger.error(f"Invalid analysis API response format or empty content: {response_data}")
                         return "[red]Analysis failed: Invalid response format or empty content from API.[/red]"

                    # Add a header to the final output
                    targets_str = ", ".join(sorted(platform_targets.keys()))
                    report_timestamp = analysis_start_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                    final_report = f"# OSINT Analysis Report\n\n" \
                                   f"**Query:** {query}\n" \
                                   f"**Targets Queried:** {targets_str}\n" \
                                   f"**Report Generated:** {report_timestamp}\n" \
                                   f"**Models Used:**\n- Text Analysis: `{text_model}`\n- Image Analysis: `{image_model}`\n\n"
                    if failed_fetches:
                        final_report += f"**Data Collection Issues:** Data collection failed or was limited for some targets (see logs or previous messages).\n\n"
                    final_report += "---\n\n" + analysis_content.strip() # Add stripped content

                    return final_report

                 finally:
                     # Stop the analysis task spinner once the thread is done or error occurred
                     if analysis_task is not None and analysis_task in self.progress.task_ids:
                          self.progress.update(analysis_task, visible=False) # Hide task instead of removing
                          self.progress.remove_task(analysis_task)
                     # Reset state variables used by the thread
                     self._analysis_response = None
                     self._analysis_exception = None
            # --- End Analysis Call Block ---


        except RateLimitExceededError as rle:
             # Stop progress if it's running from collection phase
             # Context manager should handle this, but check just in case
             # if self.progress.live.is_started: self.progress.stop()
             # Error message already printed by handler
             return f"[red]Analysis aborted due to rate limiting during data collection: {rle}[/red]"
        except RuntimeError as run_err: # Catch the re-raised API error or other runtime issues
             # if self.progress.live.is_started: self.progress.stop()
             self.console.print(f"[bold red]Analysis Failed:[/bold red] {run_err}")
             return f"[red]Analysis failed: {run_err}[/red]"
        except Exception as e:
             # if self.progress.live.is_started: self.progress.stop()
             logger.error(f"Unexpected error during analysis phase: {str(e)}", exc_info=True)
             return f"[red]Analysis failed due to unexpected error: {str(e)}[/red]"


    def _format_text_data(self, platform: str, username: str, data: dict) -> str:
        """Formats fetched data into a detailed text block for the analysis LLM."""
        # Limit items shown per type and text length for LLM context window
        MAX_ITEMS_PER_TYPE = 25
        TEXT_SNIPPET_LENGTH = 750

        if not data: return "" # Return empty if no data provided

        output_lines = []
        platform_display_name = platform.capitalize()
        user_prefix = ""
        display_username = username # Default

        # Get platform-specific display name and potentially override username display
        user_info = data.get('user_info') or data.get('profile_info') or data.get('user_profile') # Consolidate user info lookup
        if platform == 'twitter':
             user_prefix = "@"
             display_username = user_info.get('username', username) if user_info else username
        elif platform == 'reddit':
             user_prefix = "u/"
             display_username = user_info.get('name', username) if user_info else username
        elif platform == 'mastodon':
             # Use the full 'acct' from user_info if available, otherwise keep original input
             display_username = user_info.get('acct', username) if user_info else username
        elif platform == 'hackernews': user_prefix = "" # HN has no standard prefix
        elif platform == 'bluesky':
             # Use handle from profile if available
             display_username = user_info.get('handle', username) if user_info else username


        output_lines.append(f"### {platform_display_name} Data Summary for: {user_prefix}{display_username}")

        # --- Add Profile Info ---
        if user_info:
            output_lines.append("\n**User Profile:**")
            created_at_dt = get_sort_key(user_info, 'created_at') or get_sort_key(user_info, 'created_utc')
            created_at_str = created_at_dt.strftime('%Y-%m-%d') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
            output_lines.append(f"- Handle/Username: `{user_prefix}{display_username}`")
            if 'id' in user_info: output_lines.append(f"- ID: `{user_info['id']}`")
            if 'display_name' in user_info and user_info['display_name'] != display_username: output_lines.append(f"- Display Name: '{user_info['display_name']}'")
            output_lines.append(f"- Account Created: {created_at_str}")

            # Platform-specific profile details
            if platform == 'twitter':
                output_lines.append(f"- Verified: {user_info.get('verified', 'N/A')}")
                if user_info.get('location'): output_lines.append(f"- Location: {user_info['location']}")
                if user_info.get('description'): output_lines.append(f"- Description: {user_info['description'][:200] + ('...' if len(user_info['description']) > 200 else '')}") # Snippet
                pm = user_info.get('public_metrics', {})
                output_lines.append(f"- Stats: Followers={pm.get('followers_count', 'N/A')}, Following={pm.get('following_count', 'N/A')}, Tweets={pm.get('tweet_count', 'N/A')}")
            elif platform == 'reddit':
                 output_lines.append(f"- Karma: Link={user_info.get('link_karma', 'N/A')}, Comment={user_info.get('comment_karma', 'N/A')}")
                 output_lines.append(f"- Suspended: {user_info.get('is_suspended', 'N/A')}")
                 if user_info.get('icon_img'): output_lines.append(f"- Icon URL: {user_info['icon_img']}")
            elif platform == 'bluesky':
                 if user_info.get('description'): output_lines.append(f"- Bio: {user_info['description'][:200] + ('...' if len(user_info['description']) > 200 else '')}") # Snippet
                 pm = user_info # Bluesky profile info has counts directly
                 output_lines.append(f"- Stats: Posts={pm.get('posts_count', 'N/A')}, Following={pm.get('follows_count', 'N/A')}, Followers={pm.get('followers_count', 'N/A')}")
                 if user_info.get('labels'): output_lines.append(f"- Labels: {', '.join(l['value'] for l in user_info['labels'])}")
            elif platform == 'mastodon':
                 output_lines.append(f"- Locked Account: {user_info.get('locked', 'N/A')}")
                 output_lines.append(f"- Bot Account: {user_info.get('bot', 'N/A')}")
                 if user_info.get('note_text'): output_lines.append(f"- Bio: {user_info['note_text'][:200] + ('...' if len(user_info['note_text']) > 200 else '')}") # Use cleaned text snippet
                 pm = user_info
                 output_lines.append(f"- Stats: Followers={pm.get('followers_count', 'N/A')}, Following={pm.get('following_count', 'N/A')}, Posts={pm.get('statuses_count', 'N/A')}")
                 if user_info.get('custom_fields'):
                      fields_str = ", ".join([f"{f['name']}: {f['value'][:50]}" for f in user_info['custom_fields']])
                      output_lines.append(f"- Profile Metadata: {fields_str}")

        # --- Add Activity Stats Summary ---
        stats = data.get('stats', {})
        if stats:
            output_lines.append("\n**Cached Activity Overview:**")
            stat_items = []
            if platform == 'reddit':
                 stat_items.extend([f"Subs={stats.get('total_submissions_cached', 0)}", f"Comments={stats.get('total_comments_cached', 0)}", f"Media Posts={stats.get('submissions_with_media', 0)}", f"Avg Sub Score={stats.get('avg_submission_score', 0):.1f}", f"Avg Comment Score={stats.get('avg_comment_score', 0):.1f}"])
            elif platform == 'twitter': # Twitter doesn't have pre-calculated stats in cache, could add if needed
                pass # No summary stats in current Twitter cache structure
            elif platform == 'bluesky':
                 stat_items.extend([f"Posts={stats.get('total_posts_cached', 0)}", f"Media Posts={stats.get('posts_with_media', 0)}", f"Replies={stats.get('reply_posts_cached', 0)}", f"Avg Likes={stats.get('avg_likes', 0):.1f}", f"Avg Reposts={stats.get('avg_reposts', 0):.1f}"])
            elif platform == 'mastodon':
                 stat_items.extend([f"Posts={stats.get('total_posts_cached',0)}", f"Originals={stats.get('total_original_posts_cached',0)}", f"Boosts={stats.get('total_reblogs_cached',0)}", f"Replies={stats.get('total_replies_cached',0)}", f"Media Posts={stats.get('posts_with_media', 0)}", f"Avg Favs (Orig)={stats.get('avg_favourites_on_originals', 0):.1f}", f"Avg Boosts (Orig)={stats.get('avg_reblogs_on_originals', 0):.1f}"])
            elif platform == 'hackernews':
                 stat_items.extend([f"Items={stats.get('total_items_cached', 0)}", f"Stories={stats.get('total_stories_cached', 0)}", f"Comments={stats.get('total_comments_cached', 0)}", f"Avg Story Pts={stats.get('average_story_points', 0):.1f}", f"Avg Comment Pts={stats.get('average_comment_points', 0):.1f}"])

            if stat_items: output_lines.append(f"- {', '.join(stat_items)}")
            if stats.get('total_media_items_processed') is not None:
                 output_lines.append(f"- Total Media Items Processed (in cache): {stats.get('total_media_items_processed')}")


        # --- Add Recent Activity Items ---

        if platform == 'twitter':
            tweets = data.get('tweets', [])
            output_lines.append(f"\n**Recent Tweets (up to {MAX_ITEMS_PER_TYPE}):**")
            if not tweets:
                output_lines.append("- No tweets found in cached data.")
            else:
                for i, t in enumerate(tweets[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = get_sort_key(t, 'created_at') # Use global helper
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                    media_count = len(t.get('media', []))
                    media_info = f" (Media: {media_count})" if media_count > 0 else ""
                    # Indicate type (reply, quote)
                    ref_info = ""
                    if t.get('in_reply_to_user_id'): ref_info += " (Reply)"
                    if any(ref['type'] == 'quoted' for ref in t.get('referenced_tweets', [])): ref_info += " (Quote Tweet)"

                    text = t.get('text', '[No Text]')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    metrics = t.get('metrics', {})
                    output_lines.append(
                        f"- Tweet {i+1} ({created_at_str}){ref_info}{media_info}:\n"
                        f"  Content: {text_snippet}\n"
                        f"  Metrics: Likes={metrics.get('like_count', 0)}, RTs={metrics.get('retweet_count', 0)}, Replies={metrics.get('reply_count', 0)}, Quotes={metrics.get('quote_count', 0)}"
                    )

        elif platform == 'reddit':
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
                    nsfw_info = " (NSFW)" if s.get('over_18') else ""
                    spoiler_info = " (Spoiler)" if s.get('spoiler') else ""
                    text = s.get('text', '')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    text_info = f"\n  Self-Text: {text_snippet}" if text_snippet else ""
                    link_info = f"\n  Link URL: {s.get('link_url')}" if s.get('link_url') else ""
                    output_lines.append(
                        f"- Submission {i+1} in r/{s.get('subreddit', 'N/A')} ({created_at_str}):{media_info}{nsfw_info}{spoiler_info}\n"
                        f"  Title: {s.get('title', '[No Title]')}\n"
                        f"  Score: {s.get('score', 0)} (Ratio: {s.get('upvote_ratio', 'N/A')}), Comments: {s.get('num_comments', 'N/A')}"
                        f"{link_info}{text_info}"
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
                    submitter_info = " (OP)" if c.get('is_submitter') else ""
                    output_lines.append(
                        f"- Comment {i+1} in r/{c.get('subreddit', 'N/A')} ({created_at_str}){submitter_info}:\n"
                        f"  Content: {text_snippet}\n"
                        f"  Score: {c.get('score', 0)}, Link: {c.get('permalink', 'N/A')}"
                    )

        elif platform == 'hackernews':
            items = data.get('items', []) # Use 'items' key
            output_lines.append(f"\n**Recent Activity (Stories & Comments, up to {MAX_ITEMS_PER_TYPE}):**")
            if not items:
                output_lines.append("- No activity found.")
            else:
                for i, item in enumerate(items[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = get_sort_key(item, 'created_at') # Use global helper
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else 'N/A'
                    item_type = item.get('type', 'unknown').capitalize()
                    title = item.get('title')
                    text = item.get('text', '') # Already cleaned in fetcher
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    hn_link = f"https://news.ycombinator.com/item?id={item.get('objectID')}"

                    output_lines.append(f"- Item {i+1} ({item_type}, {created_at_str}):")
                    if title: output_lines.append(f"  Title: {title}")
                    if item.get('url'): output_lines.append(f"  URL: {item.get('url')}")
                    if text_snippet: output_lines.append(f"  Text: {text_snippet}")

                    points = item.get('points')
                    num_comments = item.get('num_comments')
                    stats_parts = []
                    if points is not None: stats_parts.append(f"Pts={points}")
                    if item_type == 'Story' and num_comments is not None: # Only show num_comments for stories
                        stats_parts.append(f"Comments={num_comments}")
                    if stats_parts: output_lines.append(f"  Stats: {', '.join(stats_parts)}")

                    output_lines.append(f"  HN Link: {hn_link}")
                    if item_type == 'Comment' and item.get('story_id'):
                         output_lines.append(f"  Parent Story: https://news.ycombinator.com/item?id={item['story_id']}")


        elif platform == 'bluesky':
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
                    embed_type = p.get('embed_type')
                    embed_desc = f" (Embed: {embed_type.split('.')[-1]})" if embed_type else "" # Show simplified type
                    reply_info = " (Reply)" if p.get('reply_parent') else ""
                    langs_info = f" (Lang: {','.join(p.get('langs',[]))})" if p.get('langs') else ""

                    output_lines.append(
                         f"- Post {i+1} ({created_at_str}):{reply_info}{media_info}{embed_desc}{langs_info}\n"
                         f"  Content: {text_snippet}\n"
                         f"  Stats: Likes={p.get('likes', 0)}, Reposts={p.get('reposts', 0)}, Replies={p.get('reply_count', 0)}\n"
                         f"  URI: {p.get('uri', 'N/A')}"
                     )

        # +++ Add Mastodon Formatting +++
        elif platform == 'mastodon':
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
                    cw_text = p.get('spoiler_text', '')
                    cw_info = f" (CW: {cw_text[:50]}{'...' if len(cw_text)>50 else ''})" if cw_text else ""
                    is_boost = p.get('is_reblog', False)
                    boost_info = f" (Boost of {p.get('reblog_original_author', 'unknown')})" if is_boost else ""
                    reply_info = " (Reply)" if p.get('in_reply_to_id') else ""
                    visibility = p.get('visibility', 'public')
                    vis_info = f" ({visibility.capitalize()})" if visibility != 'public' else ""
                    lang_info = f" (Lang: {p.get('language')})" if p.get('language') else ""


                    # Use cleaned text, provide context for boosts/CWs
                    text_snippet = p.get('text_cleaned', '')
                    if cw_text and '[Content Hidden]' in text_snippet: # Check if content was explicitly hidden
                         text_display = text_snippet # Show "[CW:...] [Content Hidden]"
                    elif is_boost and not text_snippet:
                         text_display = "[Boost Content Only - See Original]"
                    else: # Show snippet if available
                         text_display = text_snippet[:TEXT_SNIPPET_LENGTH] + ('...' if len(text_snippet) > TEXT_SNIPPET_LENGTH else '')
                         if not text_display: text_display = "[No Text Content]" # Handle empty strings


                    output_lines.append(
                        f"- Post {i+1} ({created_at_str}):{boost_info}{reply_info}{vis_info}{cw_info}{media_info}{lang_info}\n"
                        f"  Content: {text_display}\n"
                        f"  Stats: Favs={p.get('favourites_count', 0)}, Boosts={p.get('reblogs_count', 0)}, Replies={p.get('replies_count', 0)}\n"
                        f"  Link: {p.get('url', 'N/A')}"
                    )
                    # Add tags and poll info if present
                    if p.get('tags'): output_lines.append(f"  Tags: {', '.join(['#' + t['name'] for t in p['tags']])}")
                    if p.get('poll'):
                         poll = p['poll']
                         options_str = ", ".join([f"'{opt['title']}' ({opt.get('votes_count','?')})" for opt in poll.get('options',[])])
                         output_lines.append(f"  Poll ({poll.get('votes_count', '?')} votes): {options_str}")
                    # Optionally add link to boosted post
                    if is_boost and p.get('reblog_original_url'):
                         output_lines.append(f"  Original Post: {p['reblog_original_url']}")

        # --- End Mastodon Formatting ---

        else:
            # Fallback for any other platform - display raw data snippet
            output_lines.append(f"\n**Raw Data Preview (Unknown Platform Type):**")
            output_lines.append(f"- {str(data)[:TEXT_SNIPPET_LENGTH]}...")

        return "\n".join(output_lines)


    def _call_openrouter(self, json_data: dict):
        """Worker function for making the OpenRouter API call in a thread."""
        # Reset state variables specific to this call
        thread_response = None
        thread_exception = None
        try:
            # Ensure client is ready (accessing property initializes if needed)
            client = self.openrouter
            logger.debug(f"Sending analysis request to OpenRouter model: {json_data.get('model')}")
            thread_response = client.post("/chat/completions", json=json_data)
            # Raise HTTP errors immediately *within the thread* so they get caught
            thread_response.raise_for_status()
            logger.debug(f"Received successful ({thread_response.status_code}) response from OpenRouter.")
        except Exception as e:
            # Store the exception to be checked by the main thread
            logger.error(f"OpenRouter API call error in thread: {str(e)}")
            # Log response details if it's an HTTP error
            if isinstance(e, httpx.HTTPStatusError) and e.response:
                response_text = e.response.text[:500] # Limit log size
                logger.error(f"Response status: {e.response.status_code}. Response body snippet: {response_text}")
            thread_exception = e
        finally:
            # Set the shared instance variables *after* the call completes or fails
            self._analysis_response = thread_response
            self._analysis_exception = thread_exception


    def _save_output(self, content: str, query: str, platforms_analyzed: List[str], format_type: str = "markdown"):
        """Saves the analysis report to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_dir / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

        # Create a safe filename base from query and platforms
        safe_query = "".join(c if c.isalnum() else '_' for c in query[:30]).strip('_') or "query"
        safe_platforms = "_".join(sorted(platforms_analyzed))[:30].strip('_') or "platforms"
        filename_base = f"analysis_{timestamp}_{safe_platforms}_{safe_query}"

        # Extract metadata from the report header if possible
        report_lines = content.splitlines()
        metadata = { # Defaults
             "query": query,
             "platforms_analyzed": platforms_analyzed,
             "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "text_model": os.getenv('ANALYSIS_MODEL', 'unknown'),
             "image_model": os.getenv('IMAGE_ANALYSIS_MODEL', 'unknown'),
             "output_format": format_type
        }
        report_content_md = content # Default to full content

        # Try to parse metadata from the standard header format added in analyze()
        try:
            if report_lines[0].strip() == "# OSINT Analysis Report":
                header_lines = []
                content_start_index = 1
                for i, line in enumerate(report_lines[1:], 1):
                    if line.strip() == '---':
                         content_start_index = i + 1
                         break
                    header_lines.append(line)

                # Parse key-value pairs from header
                for line in header_lines:
                     if ':' in line:
                          key, val = line.split(':', 1)
                          key = key.strip().lower().replace(' ', '_')
                          val = val.strip()
                          if key == 'query': metadata['query'] = val
                          elif key == 'targets_queried': metadata['platforms_analyzed'] = sorted([p.strip() for p in val.split(',')])
                          elif key == 'report_generated': metadata['timestamp_utc'] = val # Keep as string from report
                          elif key == 'text_analysis': metadata['text_model'] = val.strip('`')
                          elif key == 'image_analysis': metadata['image_model'] = val.strip('`')

                # Get the main content after the header and separator
                report_content_md = "\n".join(report_lines[content_start_index:]).strip()
        except IndexError:
             logger.warning("Could not parse metadata from report header, using defaults.")
             report_content_md = content # Use original content if parsing fails


        try:
            if format_type == "json":
                filename = output_dir / f"{filename_base}.json"
                # Store structured metadata and raw markdown content
                data_to_save = {
                    "analysis_metadata": metadata,
                    "analysis_report_markdown": report_content_md
                }
                filename.write_text(json.dumps(data_to_save, indent=2, cls=DateTimeEncoder), encoding='utf-8')
            else: # Default to markdown
                filename = output_dir / f"{filename_base}.md"
                # Reconstruct markdown with YAML frontmatter
                md_metadata = "---\n"
                md_metadata += f"Query: {metadata['query']}\n"
                md_metadata += f"Platforms: {', '.join(metadata['platforms_analyzed'])}\n"
                md_metadata += f"Timestamp: {metadata['timestamp_utc']}\n" # Use timestamp from metadata
                md_metadata += f"Text Model: {metadata['text_model']}\n"
                md_metadata += f"Image Model: {metadata['image_model']}\n"
                md_metadata += "---\n\n"

                # Use the parsed report content
                full_content = md_metadata + report_content_md
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
             parsed_url = urlparse(base_url) if base_url else None
             if parsed_url and parsed_url.scheme and parsed_url.netloc:
                 available.append('mastodon')
             elif check_creds: # Only warn if checking creds and URL is bad/missing
                 if not base_url: logger.warning("Mastodon credentials found, but MASTODON_API_BASE_URL is missing.")
                 else: logger.warning(f"Mastodon credentials found, but MASTODON_API_BASE_URL ('{base_url}') is invalid.")

        # --- End Mastodon Check ---
        # HackerNews is always conceptually available (no creds needed)
        if not check_creds or True: # Always add HN if not checking creds, or just always add it
            available.append('hackernews')

        return sorted(list(set(available))) # Ensure unique and sorted

    # --- Interactive Mode ---
    def run(self):
        """Runs the interactive command-line interface."""
        self.console.print(Panel(
            "[bold blue]Social Media OSINT LM[/bold blue]\n" # Renamed slightly
            "Collects and analyzes user activity across multiple platforms using LLMs.\n"
            "Ensure API keys and identifiers are set in your `.env` file.",
            title="Welcome",
            border_style="blue"
        ))

        # Check essential core config first (OpenRouter)
        try:
            _ = self.openrouter # Access property to trigger init check
        except RuntimeError as core_err:
            self.console.print(f"[bold red]Critical Configuration Error:[/bold red] {core_err}")
            self.console.print("Cannot proceed without OpenRouter configuration.")
            return # Exit immediately

        while True:
            self.console.print("\n[bold cyan]Select Platform(s) for Analysis:[/bold cyan]")

            # Get currently configured platforms for the menu
            current_available = self.get_available_platforms(check_creds=True)

            if not current_available:
                 # Check if HN is the *only* thing available conceptually
                 all_conceptual = self.get_available_platforms(check_creds=False)
                 if 'hackernews' in all_conceptual and len(all_conceptual) == 1:
                      self.console.print("[yellow]Only HackerNews seems to be available (no other platform credentials found).[/yellow]")
                      current_available = ['hackernews'] # Allow selection of HN alone
                 else:
                      self.console.print("[bold red]Error: No platforms seem to be configured correctly.[/bold red]")
                      self.console.print("Please set credentials in a `.env` file (e.g., TWITTER_BEARER_TOKEN, MASTODON_ACCESS_TOKEN etc.) and ensure URLs/Identifiers are valid.")
                      self.console.print("Check `analyzer.log` for detailed errors during startup.")
                      break # Exit loop if nothing is configured


            platform_priority = {
                'twitter': 1,
                'bluesky': 2,
                'mastodon': 3, # Add Mastodon priority
                'reddit': 4,
                'hackernews': 5,
            }

            # Sort available platforms by priority, then alphabetically
            current_available.sort(key=lambda x: (platform_priority.get(x, 999), x))

            platform_options = {str(i+1): p for i, p in enumerate(current_available)}
            num_platforms = len(current_available)

            # Add cross-platform and exit options dynamically
            next_key = num_platforms + 1
            cross_platform_key = None
            if num_platforms > 1: # Only offer cross-platform if more than one platform is available
                cross_platform_key = str(next_key)
                platform_options[cross_platform_key] = "cross-platform"
                next_key += 1
            exit_key = str(next_key)
            platform_options[exit_key] = "exit"

            # Display options
            for key, name in platform_options.items():
                 self.console.print(f" {key}. {name.capitalize()}")

            choice = Prompt.ask("Enter number(s) (e.g., 1 or 1,3 or cross-platform key)", default=exit_key).strip().lower()

            if choice == exit_key or choice == 'exit':
                break

            selected_platform_keys = []
            selected_names = []
            if cross_platform_key and (choice == cross_platform_key or choice == 'cross-platform'):
                 # Select all available platforms *except* 'cross-platform' and 'exit' themselves
                 selected_platform_keys = [k for k, v in platform_options.items() if v not in ["cross-platform", "exit"]]
                 selected_names = [platform_options[k] for k in selected_platform_keys]
                 self.console.print(f"Selected: Cross-Platform Analysis ({', '.join(name.capitalize() for name in selected_names)})")
            else:
                 # Handle comma-separated input or single number
                 raw_keys = [k.strip() for k in choice.split(',')]
                 valid_keys_found = []
                 invalid_inputs = []
                 for k in raw_keys:
                     if k in platform_options and k not in [cross_platform_key, exit_key]:
                         valid_keys_found.append(k)
                     else:
                         invalid_inputs.append(k)

                 if not valid_keys_found:
                     self.console.print(f"[yellow]Invalid selection: '{choice}'. Please enter numbers corresponding to the platform options.[/yellow]")
                     continue
                 if invalid_inputs:
                      self.console.print(f"[yellow]Ignoring invalid input(s): {', '.join(invalid_inputs)}[/yellow]")

                 selected_platform_keys = sorted(list(set(valid_keys_found))) # Ensure unique and sorted
                 selected_names = [platform_options[k].capitalize() for k in selected_platform_keys]
                 self.console.print(f"Selected: {', '.join(selected_names)}")


            platforms_to_query: Dict[str, List[str]] = {}
            abort_selection = False
            try:
                for key in selected_platform_keys:
                     if abort_selection: break # Stop asking if an error occurred
                     if key not in platform_options: continue # Should not happen with validation, but safe check

                     platform_name = platform_options[key]
                     prompt_message = f"Enter {platform_name.capitalize()} username(s) (comma-separated"
                     # Add platform-specific hints
                     if platform_name == 'twitter': prompt_message += ", no '@')"
                     elif platform_name == 'reddit': prompt_message += ", no 'u/')"
                     elif platform_name == 'bluesky': prompt_message += ", e.g., 'handle.bsky.social')"
                     elif platform_name == 'mastodon': prompt_message += ", format: 'user@instance.domain')"
                     elif platform_name == 'hackernews': prompt_message += ")"
                     else: prompt_message += ")" # Default closing parenthesis

                     user_input = Prompt.ask(prompt_message, default="").strip()
                     if not user_input:
                          self.console.print(f"[yellow]No usernames entered for {platform_name.capitalize()}. Skipping.[/yellow]")
                          continue # Skip this platform if no input

                     # Split and strip, filter out empty strings
                     usernames = [u.strip() for u in user_input.split(',') if u.strip()]
                     if not usernames:
                          self.console.print(f"[yellow]No valid usernames provided for {platform_name.capitalize()} after stripping. Skipping.[/yellow]")
                          continue # Skip if only whitespace or empty strings entered

                     validated_users = []
                     # --- Platform-Specific Validation ---
                     if platform_name == 'mastodon':
                          default_instance_url = os.getenv('MASTODON_API_BASE_URL', '')
                          default_instance_domain = urlparse(default_instance_url).netloc if default_instance_url else None
                          for u in usernames:
                               if '@' in u and '.' in u.split('@')[1]: # Basic check for instance part
                                   validated_users.append(u)
                               else:
                                   # Check if a default instance can be inferred
                                   if default_instance_domain:
                                       assumed_user = f"{u}@{default_instance_domain}"
                                       # Ask for confirmation with default True
                                       if Confirm.ask(f"[yellow]Username '{u}' lacks instance. Assume '{assumed_user}' (from .env)?", default=True):
                                            validated_users.append(assumed_user)
                                       else:
                                            self.console.print(f"[yellow]Skipping Mastodon username '{u}' due to missing instance.[/yellow]")
                                   else:
                                       # No default instance, invalid format
                                       self.console.print(f"[bold red]Invalid Mastodon username format: '{u}'. Needs 'user@instance.domain'. Cannot assume default. Skipping.[/bold red]")
                          usernames = validated_users # Use only validated/corrected usernames

                     elif platform_name == 'twitter': # Strip leading @ if present
                           usernames = [u.lstrip('@') for u in usernames]
                           validated_users = usernames # Assume valid after stripping @
                     elif platform_name == 'reddit': # Strip leading u/ or /u/
                           usernames = [u.replace('u/', '').replace('/u/', '') for u in usernames]
                           validated_users = usernames
                     else: # For Bluesky, HN, assume input is okay for now
                          validated_users = usernames
                     # --- End Validation ---


                     if validated_users:
                         # Ensure platform_name is in dict before extending/assigning
                         if platform_name not in platforms_to_query:
                             platforms_to_query[platform_name] = []
                         # Add only unique usernames per platform
                         current_users = set(platforms_to_query[platform_name])
                         added_count = 0
                         for user in validated_users:
                              if user not in current_users:
                                   platforms_to_query[platform_name].append(user)
                                   current_users.add(user)
                                   added_count += 1
                         if added_count < len(validated_users):
                              logger.debug(f"Excluded {len(validated_users) - added_count} duplicate username(s) for {platform_name}.")
                     else:
                          # Log if validation removed all users for a platform
                          logger.warning(f"No valid usernames remained for {platform_name} after validation/confirmation.")


                if not platforms_to_query:
                    self.console.print("[yellow]No valid usernames entered or confirmed for any selected platform. Returning to menu.[/yellow]")
                    continue

                # --- Initialize Clients Here (Before Analysis Loop) ---
                # Attempt to initialize clients for selected platforms to catch setup errors early
                self.console.print("[cyan]Initializing API clients...")
                init_ok = True
                with self.progress: # Use progress for client init
                    init_task = self.progress.add_task("Initializing...", total=len(platforms_to_query))
                    for platform_name in platforms_to_query.keys():
                        self.progress.update(init_task, description=f"Initializing {platform_name.capitalize()}...")
                        try:
                            _ = getattr(self, platform_name) # Access property to trigger initialization
                            logger.info(f"{platform_name.capitalize()} client initialized successfully.")
                        except (RuntimeError, ValueError, MastodonError, tweepy.errors.TweepyException, prawcore.exceptions.PrawcoreException, atproto_exceptions.AtProtocolError) as client_err:
                             self.console.print(f"[bold red]Error initializing {platform_name.capitalize()} client:[/bold red] {client_err}")
                             self.console.print(f"[yellow]Cannot analyze {platform_name.capitalize()}. Check credentials/config and logs.[/yellow]")
                             # Remove platform from analysis list
                             del platforms_to_query[platform_name]
                             init_ok = False # Mark that at least one client failed
                        finally:
                             self.progress.advance(init_task)

                if not platforms_to_query:
                     self.console.print("[bold red]No clients could be initialized successfully. Returning to menu.[/bold red]")
                     continue # Go back to main menu if all selected platforms failed init


                # Start the analysis loop for the successfully initialized platforms/users
                self._run_analysis_loop(platforms_to_query)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
                if Confirm.ask("Exit program?", default=False):
                    abort_selection = True # Signal to break outer loop
                    break # Exit while loop
                else:
                     continue # Go back to platform selection
            except Exception as e:
                logger.error(f"Unexpected error in main interactive loop: {e}", exc_info=True)
                self.console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
                if Confirm.ask("Try again?", default=False):
                    continue
                else:
                    abort_selection = True # Signal to break outer loop
                    break # Exit while loop

            if abort_selection: break # Exit outer loop if signalled


        self.console.print("\n[blue]Exiting Social Media analyzer.[/blue]")


    def _run_analysis_loop(self, platforms: Dict[str, List[str]]):
        """Inner loop for performing analysis queries on selected targets."""
        platform_labels = []
        platform_names_list = sorted(platforms.keys()) # For saving output

        # Generate display labels using validated/formatted usernames if possible
        for pf, users in platforms.items():
             display_users_list = []
             user_prefix = ""
             if pf == 'twitter': user_prefix = "@"
             elif pf == 'reddit': user_prefix = "u/"
             # Mastodon/Bluesky/HN display usernames as entered/validated
             for u in users:
                 display_users_list.append(f"{user_prefix}{u}" if user_prefix else u)
             platform_labels.append(f"{pf.capitalize()}: {', '.join(display_users_list)}")

        platform_info = " | ".join(platform_labels)
        current_targets_str = f"Targets: {platform_info}"

        self.console.print(Panel(
            f"{current_targets_str}\n"
            f"Enter your analysis query below.\n"
            f"Commands: `exit` (return to menu), `refresh` (force full data fetch), `help`",
            title="🔎 Analysis Session",
            border_style="cyan",
            expand=False
        ))

        last_query = "" # Remember last query

        while True:
            try:
                query = Prompt.ask("\n[bold green]Analysis Query>[/bold green]", default=last_query).strip()
                if not query:
                    continue

                # Store query for next default prompt
                last_query = query

                cmd = query.lower()
                if cmd == 'exit':
                    self.console.print("[yellow]Exiting analysis session, returning to platform selection.[/yellow]")
                    break # Exit inner loop -> back to platform selection
                if cmd == 'help':
                     self.console.print(Panel(
                        "**Available Commands:**\n"
                        "- `exit`: Return to the platform selection menu.\n"
                        "- `refresh`: Force a full data fetch for all current targets, ignoring cache.\n"
                        "- `help`: Show this help message.\n\n"
                        "**To analyze:**\n"
                        "Simply type your analysis question (e.g., 'What are the main topics discussed?', 'Identify potential location clues from images and text.')\n"
                        "Pressing Enter with no input repeats the last query.",
                        title="Help", border_style="blue", expand=False
                    ))
                     continue
                if cmd == 'refresh':
                    if Confirm.ask("Force refresh data for all current targets? This ignores cache and uses more API calls.", default=False):
                         total_targets_refresh = sum(len(u) for u in platforms.values())
                         failed_refreshes = []
                         # Use progress context manager for refresh
                         with self.progress:
                             refresh_task = self.progress.add_task("[yellow]Refreshing data...", total=total_targets_refresh)
                             for platform, usernames in platforms.items():
                                 fetcher = getattr(self, f'fetch_{platform}', None)
                                 if not fetcher: continue # Should not happen if client init worked, but safety check
                                 for username in usernames:
                                     # Construct display name for progress
                                     display_name = username # Default
                                     if platform == 'twitter': display_name = f"@{username}"
                                     elif platform == 'reddit': display_name = f"u/{username}"
                                     # Mastodon/Bluesky/HN use username directly

                                     self.progress.update(refresh_task, description=f"[yellow]Refreshing {platform}/{display_name}...")
                                     try:
                                         # Call fetcher with force_refresh=True
                                         # Fetcher handles its own errors (UserNotFound, RateLimit etc) and returns None on failure
                                         result = fetcher(username=username, force_refresh=True)
                                         if result is None: # Indicates fetch failure handled internally
                                             # Avoid adding duplicate failure messages if error was already logged by fetcher
                                             if not any(f[0]==platform and f[1]==display_name for f in failed_refreshes):
                                                  failed_refreshes.append((platform, display_name))
                                     except Exception as e: # Catch unexpected fetch errors during refresh
                                         # Log unexpected errors here
                                         logger.error(f"Unexpected error during refresh for {platform}/{display_name}: {e}", exc_info=True)
                                         self.console.print(f"[red]Unexpected Refresh failed for {platform}/{display_name}: {e}[/red]")
                                         if not any(f[0]==platform and f[1]==display_name for f in failed_refreshes):
                                             failed_refreshes.append((platform, display_name))
                                     finally:
                                          if refresh_task in self.progress.task_ids:
                                               self.progress.advance(refresh_task)

                         # Progress context manager handles stopping

                         if failed_refreshes:
                              self.console.print(f"[yellow]Data refresh attempted, but issues encountered for {len(failed_refreshes)} target(s) (see logs/previous messages).[/yellow]")
                         else:
                              self.console.print("[green]Data refresh attempt completed for all targets.[/green]")
                    continue # Go back to prompt after refresh attempt


                # --- Perform Analysis ---
                self.console.print(f"\n[cyan]Starting analysis for query:[/cyan] '{query}'", highlight=False)

                # Call the main analysis function (handles its own progress display)
                analysis_result = self.analyze(platforms, query)

                # Display and handle saving based on auto-save flag
                if analysis_result:
                    # Check for error markers more robustly (case-insensitive, strip whitespace)
                    result_lower_stripped = analysis_result.strip().lower()
                    # Use specific error prefixes for better matching
                    is_error = any(result_lower_stripped.startswith(prefix) for prefix in ["[red]", "error:", "analysis failed", "analysis aborted"])
                    is_warning = result_lower_stripped.startswith("[yellow]") or result_lower_stripped.startswith("warning:")

                    border_col = "red" if is_error else ("yellow" if is_warning else "green")

                    # Use Markdown rendering for output display
                    self.console.print(Panel(
                        Markdown(analysis_result), # Render result as Markdown
                        title="Analysis Report",
                        border_style=border_col,
                        expand=False, # Avoid overly wide panels
                        title_align="left"
                    ))

                    # --- Saving Logic ---
                    if not is_error: # Only attempt to save successful reports
                        save_report = False
                        save_format = 'markdown' # Default save format

                        # Check args passed to __init__ for --no-auto-save flag
                        # Note: self.args might be None if class was instantiated without args
                        no_auto_save_flag = getattr(self.args, 'no_auto_save', False)
                        specified_format = getattr(self.args, 'format', 'markdown') # Get format from args or default

                        if no_auto_save_flag:
                             # Prompt user because auto-save is disabled
                             if Confirm.ask("Save this analysis report?", default=True):
                                 save_report = True
                                 # Ask for format only if prompting for save
                                 save_format = Prompt.ask("Save format?", choices=["markdown", "json"], default=specified_format)
                        else:
                            # Auto-save is enabled (default behavior or --no-auto-save not used)
                            save_format = specified_format # Use the format specified in args (or default)
                            self.console.print(f"[cyan]Auto-saving analysis report as {save_format}...[/cyan]")
                            save_report = True

                        if save_report:
                            # Pass the raw analysis_result (which should be markdown)
                            self._save_output(analysis_result, query, platform_names_list, save_format)
                    # --- End Saving Logic ---

                else:
                    # This case should be rare if analyze() returns error strings
                    self.console.print("[red]Analysis returned no result (None). Check logs.[/red]")

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Analysis query cancelled.[/yellow]")
                if Confirm.ask("\nExit this analysis session (return to menu)?", default=False):
                    break # Exit inner loop
                else:
                    last_query = "" # Clear last query after cancellation
            except RateLimitExceededError as rle:
                 # Error already printed by handler, just inform user
                 self.console.print("[yellow]A rate limit was hit during analysis. Please wait before trying again.[/yellow]")
                 # Don't exit automatically, let user decide
            except RuntimeError as e: # Catch API errors from analyze() or other runtime issues
                 logger.error(f"Runtime error during analysis query processing: {e}", exc_info=True)
                 self.console.print(f"\n[bold red]An error occurred during analysis:[/bold red] {e}")
                 self.console.print("[yellow]Check logs for details. You can try again or exit.[/yellow]")
            except Exception as e:
                 logger.error(f"Unexpected error during analysis loop: {e}", exc_info=True)
                 self.console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")
                 if not Confirm.ask("An error occurred. Continue session?", default=True):
                     break # Exit inner loop


    # --- Non-Interactive Mode (stdin processing) ---
    def process_stdin(self):
        """
        Processes analysis request from JSON input via stdin.
        Uses args set during initialization (self.args) for format/auto-save.
        """
        # Initialize a separate console for stderr output ---
        stderr_console = Console(stderr=True)
        stderr_console.print("[cyan]Processing analysis request from stdin...[/cyan]") # Log to stderr
        try:
            try:
                 input_data = json.load(sys.stdin)
            except json.JSONDecodeError as json_err:
                 raise ValueError(f"Invalid JSON received on stdin: {json_err}")

            platforms_in = input_data.get("platforms")
            query = input_data.get("query")

            if not isinstance(platforms_in, dict) or not platforms_in:
                raise ValueError("Invalid or missing 'platforms' data in JSON input. Must be a non-empty dictionary.")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Invalid or missing 'query' in JSON input.")
            query = query.strip()

            # Validate platform usernames are lists/strings and platform is available/configured
            valid_platforms_to_analyze: Dict[str, List[str]] = {}
            available_configured = self.get_available_platforms(check_creds=True) # Check configured platforms
            available_conceptual = self.get_available_platforms(check_creds=False) # Get all possible platforms

            for platform, usernames in platforms_in.items():
                 platform = platform.lower() # Normalize platform name

                 # Check if platform is conceptually supported
                 if platform not in available_conceptual:
                      logger.warning(f"Platform '{platform}' specified in stdin is not supported by this tool. Skipping.")
                      continue

                 # Check if platform requires config and is configured
                 requires_config = platform != 'hackernews'
                 if requires_config and platform not in available_configured:
                     logger.warning(f"Platform '{platform}' specified in stdin requires configuration, but credentials/setup seem missing or invalid. Skipping.")
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

                 # --- Platform-Specific Validation (Stdin) ---
                 validated_users_for_platform = []
                 if platform == 'mastodon':
                     for u in processed_users:
                         if '@' in u and '.' in u.split('@')[1]:
                             validated_users_for_platform.append(u)
                         else:
                             logger.warning(f"Invalid Mastodon username format in stdin for '{u}'. Needs 'user@instance.domain'. Skipping user.")
                 elif platform == 'twitter':
                      validated_users_for_platform = [u.lstrip('@') for u in processed_users]
                 elif platform == 'reddit':
                      validated_users_for_platform = [u.replace('u/', '').replace('/u/', '') for u in processed_users]
                 else: # Bluesky, HN
                      validated_users_for_platform = processed_users
                 # --- End Validation ---

                 if not validated_users_for_platform:
                      logger.warning(f"No valid usernames remained for platform '{platform}' after validation. Skipping platform.")
                      continue # Skip if validation removed all users

                 if validated_users_for_platform:
                     valid_platforms_to_analyze[platform] = sorted(list(set(validated_users_for_platform))) # Store unique sorted list

            if not valid_platforms_to_analyze:
                 raise ValueError("No valid and configured platforms with valid usernames found in the processed input.")

            # --- Initialize Clients (Stdin) ---
            # Attempt to initialize clients for validated platforms to catch setup errors early
            logger.info("Initializing API clients for stdin request...")
            platforms_failed_init = []
            for platform_name in list(valid_platforms_to_analyze.keys()): # Iterate over keys copy
                 if platform_name == 'hackernews':
                     logger.info("HackerNews requires no specific client initialization.")
                     continue
                 try:
                     _ = getattr(self, platform_name) # Access property to trigger initialization
                     logger.info(f"{platform_name.capitalize()} client initialized successfully for stdin.")
                 except (RuntimeError, ValueError, MastodonError, tweepy.errors.TweepyException, prawcore.exceptions.PrawcoreException, atproto_exceptions.AtProtocolError) as client_err:
                     logger.error(f"Error initializing {platform_name.capitalize()} client for stdin: {client_err}")
                     # Remove platform from analysis list if init fails
                     del valid_platforms_to_analyze[platform_name]
                     platforms_failed_init.append(platform_name)

            if not valid_platforms_to_analyze and not platforms_failed_init:
                 # This case means only hackernews was requested and it doesn't need init
                 pass # It's okay to proceed if only hackernews was requested
            elif not valid_platforms_to_analyze:
                 # This means other platforms were requested but all failed init
                 raise RuntimeError("No clients could be initialized successfully for the requested platforms.")
            if platforms_failed_init:
                 logger.warning(f"Analysis will proceed without platforms that failed initialization: {', '.join(platforms_failed_init)}")

            # --- Run analysis ---
            logger.info(f"Starting stdin analysis for query: '{query}' on platforms: {list(valid_platforms_to_analyze.keys())}")
            analysis_report = self.analyze(valid_platforms_to_analyze, query) # Handles its own progress

            if not analysis_report:
                 # This case indicates a major failure within analyze() even before LLM call perhaps
                 raise RuntimeError("Analysis function returned no result (None). Check logs for errors during data collection or formatting.")

            # Check for error indicators in the report itself
            result_lower_stripped = analysis_report.strip().lower()
            is_error_report = any(result_lower_stripped.startswith(prefix) for prefix in ["[red]", "error:", "analysis failed", "analysis aborted"])

            if not is_error_report:
                # Analysis succeeded, output based on --no-auto-save
                platform_names_list = sorted(valid_platforms_to_analyze.keys())
                no_auto_save_flag = getattr(self.args, 'no_auto_save', False)
                output_format = getattr(self.args, 'format', 'markdown')

                if no_auto_save_flag:
                    # Print raw report (should be markdown) directly to stdout
                    print(analysis_report)
                    logger.info("Analysis complete. Report printed to stdout (--no-auto-save).")
                    sys.exit(0) # Success
                else:
                    # Auto-save enabled: Save the output file
                    # Pass raw markdown report to save function
                    self._save_output(analysis_report, query, platform_names_list, output_format)
                    # Print confirmation message to stderr using the stderr_console
                    stderr_console.print(f"[green]Analysis complete. Output auto-saved ({output_format}).[/green]")
                    sys.exit(0) # Success
            else:
                # Analysis itself produced an error message (e.g., LLM failure, partial data failure)
                # Print the error report content to stderr
                sys.stderr.write("Analysis completed with errors:\n")
                sys.stderr.write(analysis_report + "\n")
                logger.error(f"Analysis via stdin completed but generated an error report. Query: '{query}'")
                sys.exit(2) # Use different exit code for analysis errors vs setup errors

        except (json.JSONDecodeError, ValueError, RuntimeError) as e: # Catch input, setup, or client init errors
            logger.error(f"Error processing stdin request: {e}", exc_info=False) # Log error concisely
            sys.stderr.write(f"Error: {e}\n") # Print error to stderr
            sys.exit(1) # Failure code 1 for input/setup errors
        except Exception as e: # Catch unexpected errors
            logger.critical(f"Unexpected critical error during stdin processing: {e}", exc_info=True)
            sys.stderr.write(f"Critical Error: An unexpected error occurred - {e}\n")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Social Media OSINT analyzer using LLMs. Fetches user data from various platforms, performs text and image analysis, and generates reports.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Environment Variables Required:
  OPENROUTER_API_KEY      : Your OpenRouter.ai API key.
  IMAGE_ANALYSIS_MODEL    : Vision model on OpenRouter (e.g., google/gemini-pro-vision).
  ANALYSIS_MODEL          : Text model on OpenRouter (e.g., mistralai/mixtral-8x7b-instruct).

Platform Credentials (at least one set required, or just use HackerNews):
  TWITTER_BEARER_TOKEN    : Twitter API v2 Bearer Token.
  REDDIT_CLIENT_ID        : Reddit App Client ID.
  REDDIT_CLIENT_SECRET    : Reddit App Client Secret.
  REDDIT_USER_AGENT       : Reddit App User Agent string.
  BLUESKY_IDENTIFIER      : Bluesky handle or DID.
  BLUESKY_APP_SECRET      : Bluesky App Password.
  MASTODON_ACCESS_TOKEN   : Mastodon App Access Token.
  MASTODON_API_BASE_URL   : Mastodon instance base URL (e.g., https://mastodon.social).

Place these in a `.env` file in the same directory or set them in your environment.
"""
    )
    parser.add_argument(
        '--stdin',
        action='store_true',
        help="Read analysis request from stdin as JSON.\n"
             "Expected JSON format example:\n"
             '{\n'
             '  "platforms": {\n'
             '    "twitter": ["user1", "user2"],\n'
             '    "reddit": "user3",\n'
             '    "hackernews": ["user4"],\n'
             '    "bluesky": ["handle1.bsky.social"],\n'
             '    "mastodon": ["user@instance.social", "another@other.server"]\n'
             '  },\n'
             '  "query": "Analyze communication style and main topics."\n'
             '}'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'markdown'],
        default='markdown',
        help="Output format for saving analysis reports (default: markdown).\n"
             "- markdown: Saves as a .md file with YAML frontmatter.\n"
             "- json: Saves as a .json file containing metadata and the markdown report."
    )
    parser.add_argument(
        '--no-auto-save',
        action='store_true',
        help="Disable automatic saving of reports.\n"
             "- Interactive mode: Prompt user before saving.\n"
             "- Stdin mode: Print the report directly to stdout instead of saving."
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='Set the logging level (default: WARNING).'
    )


    args = parser.parse_args()

    # --- Configure Logging Level ---
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.WARNING)
    # Set level on the root logger first
    logging.getLogger().setLevel(log_level_numeric)
    # Set level on the specific logger for this module
    logger.setLevel(log_level_numeric)
    # Ensure handlers also respect the new level
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level_numeric)

    logger.info(f"Logging level set to {args.log_level}")
    # --- End Logging Config ---

    analyzer_instance = None # Define outside try block
    try:
        # Pass the parsed args to the constructor
        analyzer_instance = SocialOSINTLM(args=args)
        if args.stdin:
            analyzer_instance.process_stdin() # Uses self.args internally
        else:
            analyzer_instance.run() # Uses self.args internally

    except RuntimeError as e:
         # Catch critical setup errors during initialization (e.g., missing core env vars, client init failures)
         logging.getLogger('SocialOSINTLM').critical(f"Initialization failed: {e}", exc_info=False)
         # Use a dedicated stderr console for critical errors ---
         error_console = Console(stderr=True, style="bold red")
         error_console.print(f"\nCRITICAL ERROR: {e}")
         error_console.print("Ensure necessary API keys (OpenRouter) and platform credentials/URLs are correctly set in .env or environment.")
         error_console.print("Check analyzer.log for more details.")
         sys.exit(1)
    except Exception as e:
         # Catch any other unexpected top-level errors
         logging.getLogger('SocialOSINTLM').critical(f"An unexpected critical error occurred: {e}", exc_info=True)
         # Use a dedicated stderr console for critical errors ---
         error_console = Console(stderr=True, style="bold red")
         error_console.print(f"\nUNEXPECTED CRITICAL ERROR: {e}")
         error_console.print("Check analyzer.log for the full traceback.")
         sys.exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Social Media OSINT analyzer using LLMs. Fetches user data from various platforms, performs text and image analysis, and generates reports.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Environment Variables Required:
  OPENROUTER_API_KEY      : Your OpenRouter.ai API key.
  IMAGE_ANALYSIS_MODEL    : Vision model on OpenRouter (e.g., google/gemini-pro-vision).
  ANALYSIS_MODEL          : Text model on OpenRouter (e.g., mistralai/mixtral-8x7b-instruct).

Platform Credentials (at least one set required, or just use HackerNews):
  TWITTER_BEARER_TOKEN    : Twitter API v2 Bearer Token.
  REDDIT_CLIENT_ID        : Reddit App Client ID.
  REDDIT_CLIENT_SECRET    : Reddit App Client Secret.
  REDDIT_USER_AGENT       : Reddit App User Agent string.
  BLUESKY_IDENTIFIER      : Bluesky handle or DID.
  BLUESKY_APP_SECRET      : Bluesky App Password.
  MASTODON_ACCESS_TOKEN   : Mastodon App Access Token.
  MASTODON_API_BASE_URL   : Mastodon instance base URL (e.g., https://mastodon.social).

Place these in a `.env` file in the same directory or set them in your environment.
"""
    )
    parser.add_argument(
        '--stdin',
        action='store_true',
        help="Read analysis request from stdin as JSON.\n"
             "Expected JSON format example:\n"
             '{\n'
             '  "platforms": {\n'
             '    "twitter": ["user1", "user2"],\n'
             '    "reddit": "user3",\n'
             '    "hackernews": ["user4"],\n'
             '    "bluesky": ["handle1.bsky.social"],\n'
             '    "mastodon": ["user@instance.social", "another@other.server"]\n'
             '  },\n'
             '  "query": "Analyze communication style and main topics."\n'
             '}'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'markdown'],
        default='markdown',
        help="Output format for saving analysis reports (default: markdown).\n"
             "- markdown: Saves as a .md file with YAML frontmatter.\n"
             "- json: Saves as a .json file containing metadata and the markdown report."
    )
    parser.add_argument(
        '--no-auto-save',
        action='store_true',
        help="Disable automatic saving of reports.\n"
             "- Interactive mode: Prompt user before saving.\n"
             "- Stdin mode: Print the report directly to stdout instead of saving."
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help='Set the logging level (default: WARNING).'
    )


    args = parser.parse_args()

    # --- Configure Logging Level ---
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.WARNING)
    # Set level on the root logger first
    logging.getLogger().setLevel(log_level_numeric)
    # Set level on the specific logger for this module
    logger.setLevel(log_level_numeric)
    # Ensure handlers also respect the new level
    # Create console handler with Rich support if not already present
    # (Avoids double printing if basicConfig added StreamHandler)
    # Check if a StreamHandler already exists
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers)

    #if not has_stream_handler: # Add RichHandler if no stream handler exists
    #    from rich.logging import RichHandler
    #    rich_handler = RichHandler(rich_tracebacks=True, level=log_level_numeric, console=Console(stderr=True))
    #    logging.getLogger().addHandler(rich_handler)
    #else: # Ensure existing handlers have the right level
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level_numeric)

    logger.info(f"Logging level set to {args.log_level}")
    # --- End Logging Config ---

    analyzer_instance = None # Define outside try block
    try:
        # Pass the parsed args to the constructor
        analyzer_instance = SocialOSINTLM(args=args)
        if args.stdin:
            analyzer_instance.process_stdin() # Uses self.args internally
        else:
            analyzer_instance.run() # Uses self.args internally

    except RuntimeError as e:
         # Catch critical setup errors during initialization (e.g., missing core env vars, client init failures)
         logging.getLogger('SocialOSINTLM').critical(f"Initialization failed: {e}", exc_info=False)
         console = Console(stderr=True, style="bold red")
         console.print(f"\nCRITICAL ERROR: {e}")
         console.print("Ensure necessary API keys (OpenRouter) and platform credentials/URLs are correctly set in .env or environment.")
         console.print("Check analyzer.log for more details.")
         sys.exit(1)
    except Exception as e:
         # Catch any other unexpected top-level errors
         logging.getLogger('SocialOSINTLM').critical(f"An unexpected critical error occurred: {e}", exc_info=True)
         console = Console(stderr=True, style="bold red")
         console.print(f"\nUNEXPECTED CRITICAL ERROR: {e}")
         console.print("Check analyzer.log for the full traceback.")
         sys.exit(1)
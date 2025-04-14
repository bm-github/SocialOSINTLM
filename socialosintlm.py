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
import prawcore # Added for specific exception handling
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
import base64
from urllib.parse import quote_plus
from PIL import Image
from atproto import Client, exceptions
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

# --- Custom Exceptions ---
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
        ])
        if not platforms_configured and 'hackernews' not in self.get_available_platforms(): # HN needs no keys
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
            except (KeyError, exceptions.AtProtocolError, RuntimeError) as e:
                logger.error(f"Bluesky setup failed: {e}")
                raise RuntimeError(f"Bluesky setup failed: {e}") # Re-raise after logging
        return self._bluesky_client

    @property
    def openrouter(self) -> httpx.Client:
        if not hasattr(self, '_openrouter'):
            try:
                self._openrouter = httpx.Client(
                    base_url="https://openrouter.ai/api/v1",
                    headers={
                        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
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
             # Reddit (429) or other HTTP 429s
             if hasattr(exception, 'response') and exception.response.status_code == 429:
                 # PRAW/HTTPX don't reliably provide reset time in headers accessible here
                 reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
             else: # Not a rate limit error we specifically handle
                 logger.error(f"Unhandled HTTP Error for {platform}: {exception}")
                 raise exception # Re-raise if not rate limit related
        elif isinstance(exception, exceptions.AtProtocolError) and 'rate limit' in str(exception).lower():
             reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
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
            'image/gif': '.gif', 'image/webp': '.webp'
        }
        final_media_path = None

        try:
            # Platform-specific adjustments
            if platform == 'twitter':
                # Ensure we have the token
                if not hasattr(self, '_twitter'): self.twitter # Initialize if needed
                auth_header = {'Authorization': f'Bearer {os.environ["TWITTER_BEARER_TOKEN"]}'}
                if headers: headers.update(auth_header)
                else: headers = auth_header
            elif platform == 'bluesky':
                 # Ensure logged in
                 if not hasattr(self, '_bluesky_client'): self.bluesky
                 access_token = self.bluesky._session.access_jwt # Access protected member
                 if not access_token: raise RuntimeError("Bluesky access token not available for download.")
                 auth_header = {'Authorization': f"Bearer {access_token}"}
                 if headers: headers.update(auth_header)
                 else: headers = auth_header
                 # Clean URL
                 url = url.replace('http://', 'https://')
                 if '@' in url: # Remove potential suffixes like @jpeg
                     url = url.split('@')[0] + '@jpeg'


            with httpx.Client(follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()

            content_type = resp.headers.get('content-type', '').lower().split(';')[0] # Get 'image/jpeg' from 'image/jpeg; charset=utf-8'
            extension = valid_types.get(content_type)

            if not extension:
                logger.warning(f"Unsupported media type '{content_type}' for URL: {url}")
                return None

            final_media_path = media_path_stub.with_suffix(extension)
            final_media_path.write_bytes(resp.content)
            logger.debug(f"Downloaded media to: {final_media_path}")
            return final_media_path

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._handle_rate_limit(platform, e)
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

        try:
            temp_path = None
            with Image.open(file_path) as img:
                if img.format.lower() not in ['jpeg', 'png', 'webp', 'gif']: # Added GIF
                    logger.warning(f"Unsupported image type for analysis: {img.format} at {file_path}")
                    return None

                # Resize large images - use a larger max dim for high detail
                max_dimension = 1536 # Larger max dimension
                scale_factor = min(max_dimension / img.size[0], max_dimension / img.size[1], 1.0)

                if scale_factor < 1.0:
                    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    # Save resized as JPEG for broad compatibility
                    temp_path = file_path.with_suffix('.resized.jpg')
                    # Handle potential transparency in PNG/GIF
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(temp_path, 'JPEG', quality=85)
                    analysis_file_path = temp_path
                    logger.debug(f"Resized image for analysis: {file_path} -> {temp_path}")
                else:
                    # If not resized but not JPEG, convert for consistency
                    if img.format.lower() != 'jpeg':
                         temp_path = file_path.with_suffix('.converted.jpg')
                         if img.mode in ("RGBA", "P"):
                             img = img.convert("RGB")
                         img.save(temp_path, 'JPEG', quality=90)
                         analysis_file_path = temp_path
                         logger.debug(f"Converted image to JPEG for analysis: {file_path} -> {temp_path}")
                    else:
                        analysis_file_path = file_path

            base64_image = base64.b64encode(analysis_file_path.read_bytes()).decode('utf-8')

            # Clean up temporary file if created
            if temp_path:
                temp_path.unlink()

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

            response = self.openrouter.post(
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
            response.raise_for_status() # Check for HTTP errors first
            result = response.json()

            # Check for API-level errors sometimes returned in a 200 OK response
            if 'error' in result:
                 logger.error(f"Image analysis API error: {result['error'].get('message', 'Unknown error')}")
                 return None
            if 'choices' not in result or not result['choices'] or 'message' not in result['choices'][0] or 'content' not in result['choices'][0]['message']:
                logger.error(f"Invalid image analysis API response structure: {result}")
                return None

            analysis_text = result['choices'][0]['message']['content']
            logger.debug(f"Image analysis successful for: {file_path}")
            return analysis_text

        except (IOError, Image.DecompressionBombError) as img_err:
             logger.error(f"Image processing error for {file_path}: {str(img_err)}")
             return None
        except httpx.RequestError as req_err:
             logger.error(f"Network error during image analysis API call: {str(req_err)}")
             return None # Network errors are often transient, don't raise fatal
        except httpx.HTTPStatusError as status_err:
            # Handle rate limits specifically if applicable to the vision model endpoint
            if status_err.response.status_code == 429:
                 self._handle_rate_limit(f"Image Analysis ({model_to_use})", status_err) # Should raise RateLimitExceededError
            else:
                 logger.error(f"HTTP error {status_err.response.status_code} during image analysis: {status_err.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during image analysis for {file_path}: {str(e)}", exc_info=True) # Log full traceback for unexpected errors
            return None

    # --- Cache Management ---
    @lru_cache(maxsize=128) # Cache path generation
    def _get_cache_path(self, platform: str, username: str) -> Path:
        # Sanitize username slightly for filesystem safety, although unlikely needed
        safe_username = "".join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in username)
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
                 # Basic validation - check for expected top-level keys based on platform
                 required_keys = ['timestamp']
                 if platform == 'twitter': required_keys.extend(['tweets', 'user_info'])
                 elif platform == 'reddit': required_keys.extend(['submissions', 'comments', 'stats'])
                 elif platform == 'bluesky': required_keys.extend(['posts', 'stats'])
                 elif platform == 'hackernews': required_keys.extend(['submissions', 'stats'])

                 if all(key in data for key in required_keys):
                      logger.debug(f"Cache hit for {platform}/{username}")
                      # Ensure nested datetime strings are converted back if needed (though often handled during use)
                      # Example: Convert tweet created_at back to datetime if needed immediately
                      # if platform == 'twitter':
                      #     for tweet in data.get('tweets', []):
                      #         if isinstance(tweet.get('created_at'), str):
                      #            tweet['created_at'] = datetime.fromisoformat(tweet['created_at'])
                      return data
                 else:
                     logger.warning(f"Cache file for {platform}/{username} seems incomplete. Discarding.")
                     cache_path.unlink()
                     return None
            else:
                logger.info(f"Cache expired for {platform}/{username}")
                return data # Return expired data for incremental update baseline

        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Failed to load or parse cache for {platform}/{username}: {e}. Discarding cache.")
            cache_path.unlink(missing_ok=True) # Delete corrupted/invalid cache
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
                'reddit': [('submissions', 'created_utc'), ('comments', 'created_utc')], # Handle multiple lists
                'bluesky': ('posts', 'created_at'),
                'hackernews': ('submissions', 'created_at'),
            }

            if platform in sort_key_map:
                 items_to_sort = sort_key_map[platform]
                 if isinstance(items_to_sort, list): # Like Reddit with submissions/comments
                     for list_key, dt_key in items_to_sort:
                         if list_key in data and data[list_key]:
                            # Handle potential string datetimes during sort
                             data[list_key].sort(key=lambda x: datetime.fromisoformat(x[dt_key]) if isinstance(x.get(dt_key), str) else x.get(dt_key, datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
                 else: # Single list platforms
                    list_key, dt_key = items_to_sort
                    if list_key in data and data[list_key]:
                       data[list_key].sort(key=lambda x: datetime.fromisoformat(x[dt_key]) if isinstance(x.get(dt_key), str) else x.get(dt_key, datetime.min.replace(tzinfo=timezone.utc)), reverse=True)


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
            existing_tweets.sort(key=lambda x: datetime.fromisoformat(x['created_at']) if isinstance(x['created_at'], str) else x['created_at'], reverse=True)
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

            while True: # Loop for pagination
                try:
                    tweets_response = self.twitter.get_users_tweets(
                        id=user_id,
                        max_results=min(fetch_limit, 100), # Twitter API max is 100 per page
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
                    raise UserNotFoundError(f"Tweets not found for user ID {user_id} (@{username}). User might be protected or deleted.")
                except tweepy.Forbidden as e:
                     raise AccessForbiddenError(f"Access forbidden to @{username}'s tweets (possibly protected). Details: {e}")

                if tweets_response.data:
                    new_tweets_data.extend(tweets_response.data)
                    logger.debug(f"Fetched {len(tweets_response.data)} new tweets page.")
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
                if not pagination_token or len(new_tweets_data) >= fetch_limit : # Stop if no more pages or we hit our intended limit
                    break

            logger.info(f"Fetched {len(new_tweets_data)} total new tweets for @{username}.")

            # --- Process New Tweets and Media ---
            processed_new_tweets = []
            newly_added_media_analysis = []
            newly_added_media_paths = set() # Use set for efficient check later

            all_media_objects = {m.media_key: m for m in new_media_includes.get('media', [])}

            for tweet in new_tweets_data:
                 tweet_data = {
                     'id': tweet.id,
                     'text': tweet.text,
                     'created_at': tweet.created_at, # Already datetime object
                     'metrics': tweet.public_metrics,
                     'entities': tweet.entities, # Store entities
                     'media': []
                 }

                 if tweet.attachments and 'media_keys' in tweet.attachments:
                     for media_key in tweet.attachments['media_keys']:
                         media = all_media_objects.get(media_key)
                         if media:
                             url = media.url if media.type == 'photo' else media.preview_image_url
                             if url:
                                 media_path = self._download_media(url=url, platform='twitter', username=username)
                                 if media_path:
                                     analysis = self._analyze_image(media_path, f"Twitter user @{username}'s tweet")
                                     tweet_data['media'].append({
                                         'type': media.type,
                                         'analysis': analysis,
                                         'url': url,
                                         'local_path': str(media_path)
                                     })
                                     if analysis: newly_added_media_analysis.append(analysis)
                                     newly_added_media_paths.add(str(media_path))


                 processed_new_tweets.append(tweet_data)

            # --- Combine and Prune ---
            # Ensure lists are sorted correctly before combining
            processed_new_tweets.sort(key=lambda x: x['created_at'], reverse=True)
            existing_tweets.sort(key=lambda x: datetime.fromisoformat(x['created_at']) if isinstance(x['created_at'], str) else x['created_at'], reverse=True)

            combined_tweets = processed_new_tweets + existing_tweets
            # Prune based on MAX_CACHE_ITEMS
            final_tweets = combined_tweets[:MAX_CACHE_ITEMS]

            # Combine media analysis and paths (only add new ones)
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
             # Already handled by _handle_rate_limit, just pass
             return None # Indicate fetch failed due to rate limit
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Twitter fetch failed for @{username}: {user_err}")
             # Optionally cache the failure? No, probably better to retry next time.
             return None
        except tweepy.errors.TweepyException as e:
            logger.error(f"Twitter API error for @{username}: {str(e)}", exc_info=False)
            # Check if it's a different kind of auth error
            if "Authentication credentials" in str(e) or "bearer token" in str(e).lower():
                 raise RuntimeError(f"Twitter authentication failed. Check Bearer Token. ({e})")
            return None # Don't save cache on general API errors
        except Exception as e:
            logger.error(f"Unexpected error fetching Twitter data for @{username}: {str(e)}", exc_info=True)
            return None # Don't save cache on unexpected errors


    def fetch_reddit(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
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
            existing_submissions.sort(key=lambda x: datetime.fromisoformat(x['created_utc']) if isinstance(x['created_utc'], str) else x['created_utc'], reverse=True)
            existing_comments.sort(key=lambda x: datetime.fromisoformat(x['created_utc']) if isinstance(x['created_utc'], str) else x['created_utc'], reverse=True)

            if existing_submissions:
                latest_submission_fullname = f"t3_{existing_submissions[0]['id']}"
                logger.debug(f"Using latest submission fullname: {latest_submission_fullname}")
            if existing_comments:
                latest_comment_fullname = f"t1_{existing_comments[0]['id']}"
                logger.debug(f"Using latest comment fullname: {latest_comment_fullname}")

            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])

        try:
            redditor = self.reddit.redditor(username)
            # Check if user exists - accessing .id will raise NotFound if they don't
            try:
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

            logger.debug("Fetching new submissions...")
            try:
                # Fetch a batch and filter locally
                for submission in redditor.submissions.new(limit=fetch_limit):
                    count += 1
                    submission_fullname = submission.fullname
                    # Stop if we hit the latest known submission during an incremental update
                    if not force_refresh and submission_fullname == latest_submission_fullname:
                        logger.debug(f"Reached latest known submission {submission_fullname}. Stopping submission fetch.")
                        break

                    submission_data = {
                        'id': submission.id,
                        'title': submission.title,
                        'text': submission.selftext[:1000] if hasattr(submission, 'selftext') else '', # Increased snippet
                        'score': submission.score,
                        'subreddit': submission.subreddit.display_name,
                        'permalink': f"https://www.reddit.com{submission.permalink}", # Full URL
                        'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                        'fullname': submission_fullname,
                        'url': submission.url,
                        'is_gallery': getattr(submission, 'is_gallery', False),
                        'media_metadata': getattr(submission, 'media_metadata', None),
                        'media': []
                    }

                    # --- Process Media for New Submissions ---
                    media_processed = False
                    # Direct Image/GIF URL
                    if hasattr(submission, 'url') and submission.url and any(submission.url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                         media_path = self._download_media(url=submission.url, platform='reddit', username=username)
                         if media_path:
                             analysis = self._analyze_image(media_path, f"Reddit user u/{username}'s post in r/{submission.subreddit.display_name}")
                             submission_data['media'].append({
                                 'type': 'image', 'analysis': analysis, 'url': submission.url, 'local_path': str(media_path)
                             })
                             if analysis: newly_added_media_analysis.append(analysis)
                             newly_added_media_paths.add(str(media_path))
                             media_processed = True

                    # Reddit Gallery
                    if not media_processed and submission_data['is_gallery'] and submission_data['media_metadata']:
                        for _, media_item in submission_data['media_metadata'].items():
                             # Look for direct image URLs in gallery data ('s' dictionary usually contains 'u' or 'gif')
                             source = media_item.get('s')
                             if source:
                                 image_url = source.get('u') # Prefer high-res URL if available
                                 if not image_url and source.get('gif'): image_url = source.get('gif') # Fallback to GIF

                                 if image_url:
                                     # Reddit often URL encodes '&', need to decode
                                     image_url = image_url.replace('&amp;', '&')
                                     media_path = self._download_media(url=image_url, platform='reddit', username=username)
                                     if media_path:
                                         analysis = self._analyze_image(media_path, f"Reddit user u/{username}'s gallery post in r/{submission.subreddit.display_name}")
                                         submission_data['media'].append({
                                             'type': 'gallery_image', 'analysis': analysis, 'url': image_url, 'local_path': str(media_path)
                                         })
                                         if analysis: newly_added_media_analysis.append(analysis)
                                         newly_added_media_paths.add(str(media_path))

                    # Add submission data to new list
                    new_submissions_data.append(submission_data)

            except prawcore.exceptions.Forbidden:
                logger.warning(f"Access forbidden while fetching submissions for u/{username} (possibly user went private/suspended).")
            logger.info(f"Fetched {len(new_submissions_data)} new submissions for u/{username} (scanned {count}).")


            # --- Fetch New Comments ---
            new_comments_data = []
            count = 0
            logger.debug("Fetching new comments...")
            try:
                # Fetch a batch and filter locally
                for comment in redditor.comments.new(limit=fetch_limit):
                     count += 1
                     comment_fullname = comment.fullname
                     # Stop if we hit the latest known comment during an incremental update
                     if not force_refresh and comment_fullname == latest_comment_fullname:
                         logger.debug(f"Reached latest known comment {comment_fullname}. Stopping comment fetch.")
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
            except prawcore.exceptions.Forbidden:
                 logger.warning(f"Access forbidden while fetching comments for u/{username}.")
            logger.info(f"Fetched {len(new_comments_data)} new comments for u/{username} (scanned {count}).")


            # --- Combine and Prune ---
            new_submissions_data.sort(key=lambda x: x['created_utc'], reverse=True)
            existing_submissions.sort(key=lambda x: datetime.fromisoformat(x['created_utc']) if isinstance(x['created_utc'], str) else x['created_utc'], reverse=True)
            combined_submissions = new_submissions_data + existing_submissions
            final_submissions = combined_submissions[:MAX_CACHE_ITEMS]

            new_comments_data.sort(key=lambda x: x['created_utc'], reverse=True)
            existing_comments.sort(key=lambda x: datetime.fromisoformat(x['created_utc']) if isinstance(x['created_utc'], str) else x['created_utc'], reverse=True)
            combined_comments = new_comments_data + existing_comments
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
                'avg_submission_score': sum(s['score'] for s in final_submissions) / max(total_submissions, 1),
                'avg_comment_score': sum(c['score'] for c in final_comments) / max(total_comments, 1)
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
        """ Fetches Bluesky posts with incremental logic and image handling. """
        cached_data = self._load_cache('bluesky', username)

        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - datetime.fromisoformat(cached_data['timestamp'])) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Bluesky user {username}")
            return cached_data

        logger.info(f"Fetching Bluesky data for {username} (Force Refresh: {force_refresh})")
        latest_post_uri = None
        existing_posts = []
        existing_media_analysis = []
        existing_media_paths = []

        if not force_refresh and cached_data:
            logger.info(f"Attempting incremental fetch for Bluesky {username}")
            existing_posts = cached_data.get('posts', [])
            # Sort existing data by created_at (ISO string) to find the latest
            existing_posts.sort(key=lambda x: x['created_at'], reverse=True)
            if existing_posts:
                latest_post_uri = existing_posts[0]['uri']
                logger.debug(f"Using latest post URI: {latest_post_uri}")

            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])

        try:
            # Check user existence implicitly via get_author_feed or get_profile
            # Optional: Explicit check first
            try:
                 profile = self.bluesky.get_profile(actor=username)
                 logger.debug(f"Bluesky profile found for {username} (DID: {profile.did})")
                 # Store DID if needed later? Maybe add to user_info if creating that section
            except exceptions.AtProtocolError as e:
                 # Handle specific profile lookup errors if necessary
                 # e.g., if str(e) indicates 'Profile not found'
                 logger.warning(f"Could not fetch Bluesky profile for {username}: {e}. Proceeding to fetch feed.")
                 # If feed fetch also fails, it will be caught below.

            # --- Fetch New Posts ---
            new_posts_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            cursor = None
            processed_uris = set() # Track URIs processed in this run to avoid duplicates from API weirdness
            fetch_limit = INCREMENTAL_FETCH_LIMIT # Number of posts to check per page
            max_pages = 3 # Limit how many pages we check for new posts to avoid excessive calls

            logger.debug("Fetching new Bluesky posts...")
            for page_num in range(max_pages):
                stop_fetching = False
                try:
                    response = self.bluesky.get_author_feed(
                        actor=username,
                        cursor=cursor,
                        limit=fetch_limit
                    )
                except exceptions.AtProtocolError as e:
                    if 'rate limit' in str(e).lower():
                        self._handle_rate_limit('Bluesky', exception=e)
                        return None # Rate limit handled
                    # Check for 'could not resolve handle' or similar errors indicating user not found/deleted
                    if 'could not resolve handle' in str(e).lower() or 'Profile not found' in str(e).lower():
                         raise UserNotFoundError(f"Bluesky user {username} not found or handle cannot be resolved.")
                    # Check for forbidden/blocked access
                    if 'blocked by actor' in str(e).lower() or 'blocking actor' in str(e).lower():
                         raise AccessForbiddenError(f"Access to Bluesky user {username}'s feed is blocked.")

                    logger.error(f"Bluesky API error fetching feed for {username}: {e}")
                    # Stop fetching on unexpected errors for this user
                    return None # Return None to indicate fetch failure

                if not response or not response.feed:
                    logger.debug("No more posts found in feed.")
                    break # No more posts in the feed

                logger.debug(f"Processing page {page_num + 1} with {len(response.feed)} items.")
                for feed_item in response.feed:
                    post = feed_item.post
                    post_uri = post.uri

                    # Avoid reprocessing the same post if API returns overlaps
                    if post_uri in processed_uris:
                        continue
                    processed_uris.add(post_uri)

                    # Stop if we hit the latest known post during an incremental update
                    if not force_refresh and post_uri == latest_post_uri:
                        logger.debug(f"Reached latest known post {post_uri}. Stopping feed fetch.")
                        stop_fetching = True
                        break # Stop processing this page

                    record = getattr(post, 'record', None)
                    if not record: continue # Skip if post has no record data

                    created_at_str = getattr(record, 'created_at', None)
                    # Parse datetime string robustly
                    created_at_dt = None
                    if isinstance(created_at_str, str):
                        try:
                            # Handle potential variations in ISO format (e.g., with 'Z')
                            if created_at_str.endswith('Z'):
                                created_at_str = created_at_str[:-1] + '+00:00'
                            created_at_dt = datetime.fromisoformat(created_at_str)
                        except ValueError:
                            logger.warning(f"Could not parse Bluesky created_at timestamp: {created_at_str}")
                            created_at_dt = datetime.now(timezone.utc) # Fallback
                    else:
                        created_at_dt = datetime.now(timezone.utc) # Fallback


                    post_data = {
                        'uri': post.uri,
                        'cid': post.cid,
                        'author_did': post.author.did,
                        'text': getattr(record, 'text', '')[:2000], # Limit text length
                        'created_at': created_at_dt.isoformat(), # Store as ISO string
                        'likes': getattr(post, 'like_count', 0), # Use like_count if available
                        'reposts': getattr(post, 'repost_count', 0), # Use repost_count
                        'reply_count': getattr(post, 'reply_count', 0),
                        'embed': None, # Placeholder for simplified embed info
                        'media': []
                    }

                    # --- Process Media for New Posts ---
                    embed = getattr(record, 'embed', None)
                    image_embeds_to_process = []

                    if embed:
                        # Simplify embed representation for cache
                        embed_type = getattr(embed, '$type', 'unknown').split('.')[-1] # e.g., 'images', 'recordWithMedia'
                        post_data['embed'] = {'type': embed_type}

                        # Case 1: Direct images embed (app.bsky.embed.images)
                        if hasattr(embed, 'images'):
                            image_embeds_to_process.extend(embed.images)

                        # Case 2: Embed contains another record (e.g., quote post) which might have media
                        # (app.bsky.embed.record or app.bsky.embed.recordWithMedia)
                        record_embed = getattr(embed, 'record', None)
                        # Handle recordWithMedia - media is top level alongside record
                        media_embed = getattr(embed, 'media', None)
                        if media_embed and hasattr(media_embed, 'images'):
                             image_embeds_to_process.extend(media_embed.images)

                        # Check inside the nested record too
                        if record_embed and hasattr(record_embed, 'value'):
                            # The actual record content is often under 'value'
                            nested_record_value = getattr(record_embed, 'value', None)
                            if nested_record_value:
                                nested_embed = getattr(nested_record_value, 'embed', None)
                                if nested_embed and hasattr(nested_embed, 'images'):
                                    image_embeds_to_process.extend(nested_embed.images)

                    # Process collected image embeds
                    for image_info in image_embeds_to_process:
                        # image_info structure is typically {'image': {'ref': {'link': CID}, ...}, 'alt': '...'}
                        img_blob = getattr(image_info, 'image', None)
                        if img_blob:
                            cid = None
                            # Check different ways the CID might be stored (ref is common)
                            if hasattr(img_blob, 'ref') and hasattr(img_blob.ref, 'link'):
                                cid = img_blob.ref.link
                            elif hasattr(img_blob, 'cid'): # Sometimes it's directly 'cid'
                                cid = img_blob.cid

                            if cid:
                                # Construct CDN URL (ensure DID is available)
                                author_did = post.author.did
                                # Using feed_fullsize for better quality
                                cdn_url = f"https://cdn.bsky.app/img/feed_fullsize/plain/{quote_plus(author_did)}/{cid}@jpeg"

                                media_path = self._download_media(url=cdn_url, platform='bluesky', username=username)
                                if media_path:
                                    analysis = self._analyze_image(media_path, f"Bluesky user {username}'s post ({post.uri})")
                                    post_data['media'].append({
                                        'type': 'image',
                                        'analysis': analysis,
                                        'url': cdn_url,
                                        'alt_text': getattr(image_info, 'alt', ''),
                                        'local_path': str(media_path)
                                    })
                                    if analysis: newly_added_media_analysis.append(analysis)
                                    newly_added_media_paths.add(str(media_path))
                            else:
                                logger.warning(f"Could not find image CID in embed for post {post.uri}")
                        else:
                             logger.warning(f"Image embed structure missing 'image' blob for post {post.uri}")


                    # Add processed post data to new list
                    new_posts_data.append(post_data)

                if stop_fetching:
                    break # Exit outer loop if we found the latest known post

                cursor = response.cursor
                if not cursor:
                    logger.debug("Reached end of feed (no cursor).")
                    break # No more pages

            logger.info(f"Fetched {len(new_posts_data)} new posts for Bluesky user {username}.")

            # --- Combine and Prune ---
            new_posts_data.sort(key=lambda x: x['created_at'], reverse=True)
            existing_posts.sort(key=lambda x: x['created_at'], reverse=True)
            combined_posts = new_posts_data + existing_posts
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
                'posts': final_posts,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths,
                'stats': stats
                # Consider adding profile info here if fetched earlier
                # 'profile': profile_data if profile else None
            }

            self._save_cache('bluesky', username, final_data)
            logger.info(f"Successfully updated Bluesky cache for {username}. Total posts cached: {total_posts}")
            return final_data

        except RateLimitExceededError:
            return None # Handled
        except (UserNotFoundError, AccessForbiddenError) as user_err:
             logger.error(f"Bluesky fetch failed for {username}: {user_err}")
             return None
        except exceptions.AtProtocolError as e:
            # Catch other generic ATProto errors
            logger.error(f"Bluesky ATProtocol error for {username}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Bluesky data for {username}: {str(e)}", exc_info=True)
            return None


    def fetch_hackernews(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches Hacker News submissions incrementally via Algolia API."""
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
            # Sort existing data by created_at (datetime object) to find the latest
            existing_submissions.sort(key=lambda x: datetime.fromisoformat(x['created_at']) if isinstance(x['created_at'], str) else x['created_at'], reverse=True)
            if existing_submissions:
                # Find the max timestamp_i from the existing data
                latest_timestamp_i = max(s.get('created_at_i', 0) for s in existing_submissions)
                logger.debug(f"Using latest timestamp_i: {latest_timestamp_i}")

        try:
            # Algolia API endpoint
            base_url = "https://hn.algolia.com/api/v1/search"
            params = {
                "tags": f"author_{quote_plus(username)}",
                "hitsPerPage": INCREMENTAL_FETCH_LIMIT if not force_refresh and latest_timestamp_i > 0 else INITIAL_FETCH_LIMIT # Fetch more on full refresh
            }

            # Add numeric filter for incremental fetch
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
                 # This might mean the user has no posts, or API changed. Treat as empty for now.
                 data['hits'] = []

            logger.info(f"Fetched {len(data['hits'])} potential new submissions for HN user {username}.")

            for hit in data.get('hits', []):
                object_id = hit.get('objectID')
                # Skip if already processed (safety check, numeric filter should handle this)
                if not object_id or object_id in processed_ids:
                     continue

                created_at_ts = hit.get('created_at_i')
                if not created_at_ts: continue # Skip if no timestamp

                # Check type: story, comment, pollopt etc.
                tags = hit.get('_tags', [])
                item_type = 'unknown'
                if 'story' in tags: item_type = 'story'
                elif 'comment' in tags: item_type = 'comment'
                elif 'poll' in tags: item_type = 'poll'
                elif 'pollopt' in tags: item_type = 'pollopt' # Option for a poll


                submission_item = {
                    'objectID': object_id, # Unique ID from Algolia
                    'type': item_type,
                    'title': hit.get('title'), # Null for comments
                    'url': hit.get('url'), # Null for comments unless Ask/Show HN with URL
                    'points': hit.get('points'), # Null for comments
                    'num_comments': hit.get('num_comments'), # Null for comments
                    'story_id': hit.get('story_id'), # Relevant for comments
                    'parent_id': hit.get('parent_id'), # Relevant for comments
                    'created_at_i': created_at_ts,
                    'created_at': datetime.fromtimestamp(created_at_ts, tz=timezone.utc), # Convert to datetime
                    'text': hit.get('story_text') or hit.get('comment_text') or '' # Text content
                }
                new_submissions_data.append(submission_item)
                processed_ids.add(object_id)

            # --- Combine and Prune ---
            new_submissions_data.sort(key=lambda x: x['created_at'], reverse=True)
            # existing_submissions are already sorted from load/previous step
            combined_submissions = new_submissions_data + existing_submissions
            final_submissions = combined_submissions[:MAX_CACHE_ITEMS]

            # --- Calculate Stats (Focus on Stories/Submissions) ---
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
                'submissions': final_submissions, # Contains stories and comments
                'stats': stats
            }

            self._save_cache('hackernews', username, final_data)
            logger.info(f"Successfully updated HackerNews cache for {username}. Total items cached: {total_items}")
            return final_data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._handle_rate_limit('HackerNews (Algolia)', e)
            # Check for specific Algolia errors if documented, e.g., bad author format
            elif e.response.status_code == 400:
                 logger.error(f"HN Algolia API Bad Request (400) for {username}: {e.response.text}. Check username format.")
            else:
                 logger.error(f"HN Algolia API HTTP error for {username}: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
             logger.error(f"HN Algolia API network error for {username}: {str(e)}")
             return None
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
                if isinstance(usernames, str): usernames = [usernames]

                fetcher = getattr(self, f'fetch_{platform}', None)
                if not fetcher:
                     logger.warning(f"No fetcher method found for platform: {platform}")
                     failed_fetches.extend([(platform, u, "Fetcher not implemented") for u in usernames])
                     self.progress.advance(collect_task, advance=len(usernames)) # Advance progress for skipped users
                     continue

                for username in usernames:
                    task_desc = f"[cyan]Fetching {platform} for {username}..."
                    self.progress.update(collect_task, description=task_desc)
                    try:
                        # Use force_refresh=False for analysis calls unless explicitly needed later
                        data = fetcher(username=username, force_refresh=False)

                        if data:
                            summary = self._format_text_data(platform, username, data)
                            collected_text_summaries.append(summary)
                            # Collect media analysis only from successful fetches
                            all_media_analyzes.extend(data.get('media_analysis', []))
                            logger.info(f"Successfully collected data for {platform}/{username}")
                        else:
                            # Fetcher returned None, imply failure handled internally (rate limit, not found etc)
                             failed_fetches.append((platform, username, "Data fetch failed (check logs/rate limits)"))
                             logger.warning(f"Data fetch returned None for {platform}/{username}")


                    except RateLimitExceededError as rle:
                        # Fetcher should raise this if it hits a limit
                        failed_fetches.append((platform, username, f"Rate Limited ({rle})"))
                        # No need to print here, _handle_rate_limit does it
                    except (UserNotFoundError, AccessForbiddenError) as afe:
                         failed_fetches.append((platform, username, f"Access Error ({afe})"))
                         self.console.print(f"[yellow]Skipping {platform}/{username}: {afe}[/yellow]")
                    except Exception as e:
                        # Catch unexpected errors during fetch call
                        fetch_error_msg = f"Unexpected error during fetch for {platform}/{username}: {e}"
                        logger.error(fetch_error_msg, exc_info=True)
                        failed_fetches.append((platform, username, fetch_error_msg))
                        self.console.print(f"[red]Error fetching {platform}/{username}: {e}[/red]")
                    finally:
                         self.progress.advance(collect_task) # Advance progress regardless of outcome


            self.progress.remove_task(collect_task) # Remove collection task
            self.progress.stop() # Stop progress bar updates

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
            unique_media_analyzes = sorted(list(set(all_media_analyzes)))

            analysis_components = []
            image_model = os.getenv('IMAGE_ANALYSIS_MODEL', 'google/gemini-pro-vision') # Get model used
            text_model = os.getenv('ANALYSIS_MODEL', 'mistralai/mixtral-8x7b-instruct') # Get text model

            # Add Media Analysis Section (if any)
            if unique_media_analyzes:
                 media_summary = f"## Consolidated Media Analysis (using {image_model}):\n\n"
                 media_summary += "*Note: The following are objective descriptions based on visual content analysis.*\n\n"
                 # Use numbered list for clarity
                 media_summary += "\n".join(f"{i+1}. {analysis.strip()}" for i, analysis in enumerate(unique_media_analyzes))
                 analysis_components.append(media_summary)
                 logger.debug(f"Added {len(unique_media_analyzes)} unique media analyzes to prompt.")

            # Add Text Data Section (if any)
            if collected_text_summaries:
                 text_summary = f"## Collected Textual & Activity Data Summary:\n\n"
                 text_summary += "\n\n---\n\n".join(collected_text_summaries) # Separate platforms clearly
                 analysis_components.append(text_summary)
                 logger.debug(f"Added {len(collected_text_summaries)} platform text summaries to prompt.")

            # Construct the final prompt
            # System prompt remains the same as before
            system_prompt = """**Objective:** Generate a comprehensive behavioral and linguistic profile based on the provided social media data, employing structured analytic techniques focused on objectivity, evidence-based reasoning, and clear articulation.

**Input:** You will receive summaries of user activity (text posts, engagement metrics, descriptive analyzes of images shared) from platforms like Twitter, Reddit, Bluesky, and Hacker News for one or more specified users. The user will also provide a specific analysis query. You may also receive consolidated analyzes of images shared by the user(s).

**Primary Task:** Address the user's specific analysis query using ALL the data provided (text summaries AND image analyzes) and the analytical framework below.

**Analysis Domains (Use these to structure your thinking and response where relevant to the query):**
1.  **Behavioral Patterns:** Analyze interaction frequency, platform-specific activity (e.g., retweets vs. posts, submissions vs. comments), potential engagement triggers, and temporal communication rhythms apparent *in the provided data*. Note differences across platforms if multiple are present.
2.  **Semantic Content & Themes:** Identify recurring topics, keywords, and concepts. Analyze linguistic indicators such as expressed sentiment/tone (positive, negative, neutral, specific emotions if clear), potential ideological leanings *if explicitly stated or strongly implied by language/topics*, and cognitive framing (how subjects are discussed). Assess information source credibility *only if* the user shares external links/content within the provided data AND you can evaluate the source based on common knowledge.
3.  **Interests & Network Context:** Deduce primary interests, hobbies, or professional domains suggested by post content and image analysis. Note any interaction patterns visible *within the provided posts* (e.g., frequent replies to specific user types, retweets of particular accounts, participation in specific communities like subreddits). Avoid inferring broad influence or definitive group membership without strong evidence.
4.  **Communication Style:** Assess linguistic complexity (simple/complex vocabulary, sentence structure), use of jargon/slang, rhetorical strategies (e.g., humor, sarcasm, argumentation), markers of emotional expression (e.g., emoji use, exclamation points, emotionally charged words), and narrative consistency across platforms.
5.  **Visual Data Integration:** Explicitly incorporate insights derived from the provided image analyzes. How do the visual elements (settings, objects, activities depicted) complement, contradict, or add context to the textual data? Note any patterns in the *types* of images shared.

**Analytical Constraints & Guidelines:**
*   **Evidence-Based:** Ground ALL conclusions *strictly and exclusively* on the provided source materials (text summaries AND image analyzes). Reference specific examples or patterns from the data (e.g., "Frequent posts about [topic] on Reddit," "Image analysis of setting suggests [environment]," "Consistent use of technical jargon on HackerNews").
*   **Objectivity & Neutrality:** Maintain analytical neutrality. Avoid speculation, moral judgments, personal opinions, or projecting external knowledge not present in the data. Focus on describing *what the data shows*.
*   **Synthesize, Don't Just List:** Integrate findings from different platforms and data types (text/image) into a coherent narrative that addresses the query. Highlight correlations or discrepancies.
*   **Address the Query Directly:** Structure your response primarily around answering the user's specific question(s). Use the analysis domains as tools to build your answer.
*   **Acknowledge Limitations:** If the data is sparse, lacks specific details needed for the query, or only covers a short time period, explicitly state these limitations (e.g., "Based on the limited posts available...", "Image analysis provides no clues regarding [aspect]"). Do not invent information.
*   **Clarity & Structure:** Use clear language. Employ formatting (markdown headings, bullet points) to organize the response logically, often starting with a direct answer to the query followed by supporting evidence/analysis.

**Output:** A structured analytical report that directly addresses the user's query, rigorously supported by evidence from the provided text and image data, adhering to all constraints. Start with a summary answer, then elaborate with details structured using relevant analysis domains.
"""
            user_prompt = f"**Analysis Query:** {query}\n\n" \
                          f"**Provided Data:**\n\n" + \
                          "\n\n===\n\n".join(analysis_components) # Use a very clear separator

            # --- Call OpenRouter LLM ---
            analysis_task = self.progress.add_task(f"[magenta]Analyzing with {text_model}...", total=None)
            self.progress.start() # Ensure progress active for analysis
            try:
                # Use threading to keep UI responsive during API call
                api_thread = threading.Thread(
                    target=self._call_openrouter,
                    kwargs={
                        "json_data": {
                            "model": text_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "max_tokens": 3000, # Allow for detailed analysis
                            "temperature": 0.5, # Lower temperature for more factual, less creative analysis
                            "stream": False # Wait for full response
                        }
                    }
                )
                api_thread.start()

                while api_thread.is_alive():
                    api_thread.join(0.1) # Check frequently
                    # self.progress.advance(analysis_task) # Spinner handled by Progress itself

                if self._analysis_exception:
                    # Handle specific errors if needed, otherwise re-raise generic
                    if isinstance(self._analysis_exception, httpx.HTTPStatusError):
                        err_details = f"API HTTP {self._analysis_exception.response.status_code}"
                        # Avoid printing potentially huge response text directly to console
                        logger.error(f"Analysis API Error Response: {self._analysis_exception.response.text}")
                        err_details += " (See analyzer.log for full response)"
                        # Check for common user-facing errors
                        try:
                             error_data = self._analysis_exception.response.json()
                             if 'error' in error_data and 'message' in error_data['error']:
                                 err_details = f"API Error: {error_data['error']['message']}"
                        except json.JSONDecodeError:
                             pass # Stick with status code if response isn't JSON

                    else:
                        err_details = str(self._analysis_exception)
                    raise RuntimeError(f"Analysis API request failed: {err_details}")

                if not self._analysis_response:
                     raise RuntimeError("Analysis API call completed but no response object was captured.")

                # Process successful response
                response = self._analysis_response
                response.raise_for_status() # Should be redundant if _call_openrouter did it, but safe check

                response_data = response.json()
                if 'choices' not in response_data or not response_data['choices'] or 'message' not in response_data['choices'][0] or 'content' not in response_data['choices'][0]['message']:
                    logger.error(f"Invalid analysis API response format: {response_data}")
                    return "[red]Analysis failed: Invalid response format from API.[/red]"

                analysis_content = response_data['choices'][0]['message']['content']
                # Add a header to the final output
                final_report = f"# OSINT Analysis Report\n\n**Query:** {query}\n\n**Models Used:**\n- Text Analysis: `{text_model}`\n- Image Analysis: `{image_model}`\n\n---\n\n{analysis_content}"
                return final_report

            finally:
                 # Ensure task is removed and progress stops even if errors occurred
                 if analysis_task is not None and analysis_task in self.progress.task_ids:
                      self.progress.remove_task(analysis_task)
                 self.progress.stop()
                 # Reset state variables
                 self._analysis_response = None
                 self._analysis_exception = None


        except RateLimitExceededError as rle:
             # This might be raised from _handle_rate_limit during image analysis inside analyze()
             # Or potentially if the analysis LLM itself gets rate limited
             self.console.print(f"[bold red]Analysis Aborted: {rle}[/bold red]")
             return f"[red]Analysis aborted due to rate limiting: {rle}[/red]"
        except Exception as e:
             logger.error(f"Unexpected error during analysis phase: {str(e)}", exc_info=True)
             return f"[red]Analysis failed due to unexpected error: {str(e)}[/red]"


    def _format_text_data(self, platform: str, username: str, data: dict) -> str:
        """Formats fetched data into a detailed text block for the analysis LLM."""
        MAX_ITEMS_PER_TYPE = 25  # Max tweets, comments, posts to include per user
        TEXT_SNIPPET_LENGTH = 750 # Max characters for text snippets

        output_lines = []
        platform_display_name = platform.capitalize()
        user_prefix = ""
        if platform == 'twitter': user_prefix = "@"
        elif platform == 'reddit': user_prefix = "u/"
        elif platform == 'hackernews': user_prefix = "" # No standard prefix
        elif platform == 'bluesky': user_prefix = "" # Handle is the username

        output_lines.append(f"### {platform_display_name} Data Summary for: {user_prefix}{username}")

        # --- Platform Specific Formatting ---

        if platform == 'twitter':
            user_info = data.get('user_info', {})
            if user_info:
                output_lines.append(f"- User Profile: '{user_info.get('name')}' ({user_prefix}{user_info.get('username')}), ID: {user_info.get('id')}")
                if user_info.get('created_at'):
                    output_lines.append(f"  - Account Created: {user_info['created_at']}") # Already datetime
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
                    # Ensure created_at is datetime before formatting
                    created_at_dt = t.get('created_at')
                    if isinstance(created_at_dt, str): created_at_dt = datetime.fromisoformat(created_at_dt)
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt else 'N/A'

                    media_count = len(t.get('media', []))
                    media_info = f" (Media Attached: {media_count})" if media_count > 0 else ""
                    text = t.get('text', '[No Text]')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    metrics = t.get('metrics', {})
                    output_lines.append(
                        f"- Tweet {i+1} ({created_at_str}):{media_info}\n"
                        f"  Content: {text_snippet}\n"
                        f"  Metrics: Likes={metrics.get('like_count', 0)}, Retweets={metrics.get('retweet_count', 0)}, Replies={metrics.get('reply_count', 0)}, Quotes={metrics.get('quote_count', 0)}"
                    )

        elif platform == 'reddit':
            stats = data.get('stats', {})
            output_lines.append(
                f"- Activity Overview: Submissions={stats.get('total_submissions', 0)}, Comments={stats.get('total_comments', 0)}, Posts w/ Media={stats.get('submissions_with_media', 0)}, Avg Sub Score={stats.get('avg_submission_score', 0):.1f}, Avg Comment Score={stats.get('avg_comment_score', 0):.1f}"
            )

            submissions = data.get('submissions', [])
            output_lines.append(f"\n**Recent Submissions (up to {MAX_ITEMS_PER_TYPE}):**")
            if not submissions:
                 output_lines.append("- No submissions found.")
            else:
                for i, s in enumerate(submissions[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = s.get('created_utc')
                    if isinstance(created_at_dt, str): created_at_dt = datetime.fromisoformat(created_at_dt)
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt else 'N/A'

                    media_count = len(s.get('media', []))
                    media_info = f" (Media Attached: {media_count})" if media_count > 0 else ""
                    text = s.get('text', '')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    text_info = f"\n  Self-Text Snippet: {text_snippet}" if text_snippet else ""
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
                    created_at_dt = c.get('created_utc')
                    if isinstance(created_at_dt, str): created_at_dt = datetime.fromisoformat(created_at_dt)
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt else 'N/A'

                    text = c.get('text', '[No Text]')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    output_lines.append(
                        f"- Comment {i+1} in r/{c.get('subreddit', 'N/A')} ({created_at_str}):\n"
                        f"  Content: {text_snippet}\n"
                        f"  Score: {c.get('score', 0)}, Permalink: {c.get('permalink', 'N/A')}"
                    )

        elif platform == 'hackernews':
            stats = data.get('stats', {})
            output_lines.append(
                f"- Activity Overview: Items Cached={stats.get('total_items_cached', 0)}, Stories={stats.get('total_stories', 0)}, Comments={stats.get('total_comments', 0)}, Avg Story Pts={stats.get('average_story_points', 0):.1f}, Avg Story Comments={stats.get('average_story_num_comments', 0):.1f}"
            )
            submissions = data.get('submissions', []) # Includes comments & stories
            output_lines.append(f"\n**Recent Activity (Stories & Comments, up to {MAX_ITEMS_PER_TYPE}):**")
            if not submissions:
                output_lines.append("- No activity found.")
            else:
                for i, s in enumerate(submissions[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = s.get('created_at')
                    if isinstance(created_at_dt, str): created_at_dt = datetime.fromisoformat(created_at_dt)
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt else 'N/A'

                    item_type = s.get('type', 'unknown').capitalize()
                    title = s.get('title')
                    text = s.get('text', '')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    hn_link = f"https://news.ycombinator.com/item?id={s.get('objectID')}"

                    output_lines.append(f"- Item {i+1} ({item_type}, {created_at_str}):")
                    if title: output_lines.append(f"  Title: {title}")
                    if s.get('url'): output_lines.append(f"  URL: {s.get('url')}")
                    if text_snippet: output_lines.append(f"  Text: {text_snippet}")
                    if item_type == 'Story':
                        output_lines.append(f"  Stats: Points={s.get('points', 0)}, Comments={s.get('num_comments', 0)}")
                    output_lines.append(f"  HN Link: {hn_link}")


        elif platform == 'bluesky':
            stats = data.get('stats', {})
            output_lines.append(
                 f"- Activity Overview: Posts Cached={stats.get('total_posts', 0)}, Posts w/ Media={stats.get('posts_with_media', 0)}, Avg Likes={stats.get('avg_likes', 0):.1f}, Avg Reposts={stats.get('avg_reposts', 0):.1f}, Avg Replies={stats.get('avg_replies', 0):.1f}"
            )
            posts = data.get('posts', [])
            output_lines.append(f"\n**Recent Posts (up to {MAX_ITEMS_PER_TYPE}):**")
            if not posts:
                 output_lines.append("- No posts found.")
            else:
                for i, p in enumerate(posts[:MAX_ITEMS_PER_TYPE]):
                    created_at_dt = p.get('created_at')
                    if isinstance(created_at_dt, str): created_at_dt = datetime.fromisoformat(created_at_dt)
                    created_at_str = created_at_dt.strftime('%Y-%m-%d %H:%M') if created_at_dt else 'N/A'

                    media_count = len(p.get('media', []))
                    media_info = f" (Media Attached: {media_count})" if media_count > 0 else ""
                    text = p.get('text', '[No Text]')
                    text_snippet = text[:TEXT_SNIPPET_LENGTH] + ('...' if len(text) > TEXT_SNIPPET_LENGTH else '')
                    embed_info = p.get('embed')
                    embed_desc = f" (Embed: {embed_info['type']})" if embed_info else "" # Simple embed type info

                    output_lines.append(
                         f"- Post {i+1} ({created_at_str}):{media_info}{embed_desc}\n"
                         f"  Content: {text_snippet}\n"
                         f"  Stats: Likes={p.get('likes', 0)}, Reposts={p.get('reposts', 0)}, Replies={p.get('reply_count', 0)}\n"
                         f"  URI: {p.get('uri', 'N/A')}"
                     )

        else:
            # Fallback for any other platform
            output_lines.append(f"\n**Data Overview:**")
            # Basic preview, might need adjustment based on actual data structure
            output_lines.append(f"- Raw Data Preview: {str(data)[:TEXT_SNIPPET_LENGTH]}...")

        return "\n".join(output_lines)


    def _call_openrouter(self, json_data: dict):
        """Worker function for making the OpenRouter API call in a thread."""
        self._analysis_response = None
        self._analysis_exception = None
        try:
            response = self.openrouter.post("/chat/completions", json=json_data)
            # Check for HTTP errors immediately
            response.raise_for_status()
            self._analysis_response = response
        except Exception as e:
            # Store the exception to be checked by the main thread
            logger.error(f"OpenRouter API call error: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                try:
                     logger.error(f"Response content: {e.response.text}") # Log full error if possible
                except Exception:
                     logger.error("Could not decode error response content.")
            self._analysis_exception = e


    def _save_output(self, content: str, query: str, platforms_analyzed: List[str], format_type: str = "markdown"):
        """Saves the analysis report to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_dir / 'outputs'
        # Create a safe filename base from query and platforms
        safe_query = "".join(c if c.isalnum() else '_' for c in query[:30])
        safe_platforms = "_".join(sorted(platforms_analyzed))[:20]
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
                    "analysis_report_markdown": content # Store the markdown report
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
                full_content = md_metadata + content
                filename.write_text(full_content, encoding='utf-8')

            self.console.print(f"[green]Analysis saved to: {filename}[/green]")

        except Exception as e:
            self.console.print(f"[bold red]Failed to save output: {str(e)}[/bold red]")
            logger.error(f"Failed to save output file {filename_base}: {e}", exc_info=True)

    def get_available_platforms(self) -> List[str]:
        """Checks environment variables to see which platforms are configured."""
        available = []
        if all(os.getenv(k) for k in ['TWITTER_BEARER_TOKEN']):
            available.append('twitter')
        if all(os.getenv(k) for k in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']):
            available.append('reddit')
        if all(os.getenv(k) for k in ['BLUESKY_IDENTIFIER', 'BLUESKY_APP_SECRET']):
             available.append('bluesky')
        # HackerNews requires no keys, always available conceptually
        available.append('hackernews')
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

        available_platforms = self.get_available_platforms()
        if not available_platforms:
            self.console.print("[bold red]Error: No API credentials found for any platform.[/bold red]")
            self.console.print("Please set credentials in a `.env` file (e.g., TWITTER_BEARER_TOKEN).")
            return # Exit if no platforms usable

        while True:
            self.console.print("\n[bold cyan]Select Platform(s) for Analysis:[/bold cyan]")

            # Reordering happens here:
            platform_priority = {
                'twitter': 1,
                'bluesky': 2,
                'reddit': 3,
                'hackernews': 4,
            }

            available_platforms.sort(key=lambda x: platform_priority.get(x, 999))  # Sort by priority or put at the end

            platform_options = {str(i+1): p for i, p in enumerate(available_platforms)}
            platform_options[str(len(available_platforms) + 1)] = "cross-platform" # Add cross-platform option
            platform_options[str(len(available_platforms) + 2)] = "exit"

            for key, name in platform_options.items():
                 self.console.print(f"{key}. {name.capitalize()}")

            choice = Prompt.ask("Enter number(s) (e.g., 1 or 1,2 or 5 for cross-platform)", default=str(len(platform_options))).strip()

            if choice == str(len(available_platforms) + 2) or choice.lower() == 'exit': # Exit option
                break

            selected_platform_keys = []
            is_cross_platform = False
            if choice == str(len(available_platforms) + 1) or choice.lower() == 'cross-platform':
                 selected_platform_keys = list(platform_options.keys())[:-2] # All except cross-platform and exit
                 is_cross_platform = True
                 self.console.print(f"Selected: Cross-Platform Analysis ({', '.join(available_platforms)})")
            else:
                 # Handle comma-separated input like "1,2"
                 raw_keys = [k.strip() for k in choice.split(',')]
                 valid_keys = [k for k in raw_keys if k in platform_options and k != str(len(available_platforms)+1)] # Exclude cross-platform itself
                 if not valid_keys:
                     self.console.print("[yellow]Invalid selection. Please enter numbers corresponding to the options.[/yellow]")
                     continue
                 selected_platform_keys = valid_keys
                 selected_names = [platform_options[k].capitalize() for k in selected_platform_keys]
                 self.console.print(f"Selected: {', '.join(selected_names)}")


            platforms_to_query: Dict[str, List[str]] = {}
            try:
                for key in selected_platform_keys:
                     platform_name = platform_options[key]
                     prompt_message = f"{platform_name.capitalize()} username(s) (comma-separated"
                     if platform_name == 'twitter': prompt_message += ", no '@'"
                     elif platform_name == 'reddit': prompt_message += ", no 'u/'"
                     elif platform_name == 'bluesky': prompt_message += ", e.g., 'handle.bsky.social'"
                     prompt_message += ")"

                     user_input = Prompt.ask(prompt_message, default="").strip()
                     if user_input:
                         # Split, strip whitespace, and filter out empty strings
                         usernames = [u.strip() for u in user_input.split(',') if u.strip()]
                         if usernames:
                             platforms_to_query[platform_name] = usernames

                if not platforms_to_query:
                    self.console.print("[yellow]No usernames entered for selected platform(s). Returning to menu.[/yellow]")
                    continue

                # Start the analysis loop for the chosen platforms/users
                self._run_analysis_loop(platforms_to_query)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
                if Confirm.ask("Exit program?", default=False):
                    break
            except RuntimeError as e:
                 # Catch setup errors (missing keys, failed auth)
                 self.console.print(f"[bold red]Configuration Error:[/bold red] {e}")
                 self.console.print("Please check your .env file and API keys.")
                 if Confirm.ask("Try again?", default=False):
                     continue
                 else:
                     break # Exit if setup fails and user doesn't want to retry
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                self.console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
                if Confirm.ask("Try again?", default=False):
                    continue
                else:
                    break

        self.console.print("[blue]Exiting Social Media analyzer.[/blue]")


    def _run_analysis_loop(self, platforms: Dict[str, List[str]]):
        """Inner loop for performing analysis queries on selected targets."""
        platform_labels = []
        platform_names_list = sorted(platforms.keys()) # For saving output

        for pf, users in platforms.items():
             user_prefix = ""
             if pf == 'twitter': user_prefix = "@"
             elif pf == 'reddit': user_prefix = "u/"
             display_users = [f"{user_prefix}{u}" for u in users]
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
                         refresh_task = self.progress.add_task("[yellow]Refreshing data...", total=sum(len(u) for u in platforms.values()))
                         self.progress.start()
                         failed_refreshes = []
                         for platform, usernames in platforms.items():
                             fetcher = getattr(self, f'fetch_{platform}', None)
                             if not fetcher: continue
                             for username in usernames:
                                 self.progress.update(refresh_task, description=f"[yellow]Refreshing {platform}/{username}...")
                                 try:
                                     fetcher(username, force_refresh=True)
                                 except Exception as e:
                                     failed_refreshes.append((platform, username))
                                     logger.error(f"Refresh failed for {platform}/{username}: {e}", exc_info=False)
                                     self.console.print(f"[red]Refresh failed for {platform}/{username}: {e}[/red]")
                                 finally:
                                      self.progress.advance(refresh_task)
                         self.progress.remove_task(refresh_task)
                         self.progress.stop()
                         if failed_refreshes:
                              self.console.print(f"[yellow]Data refreshed, but failed for {len(failed_refreshes)} target(s).[/yellow]")
                         else:
                              self.console.print("[green]Data refreshed successfully for all targets.[/green]")
                    continue # Go back to prompt after refresh attempt


                # --- Perform Analysis ---
                self.console.print(f"[cyan]Starting analysis for query:[/cyan] '{query}'", highlight=False)
                analysis_result = self.analyze(platforms, query)

                # Display and handle saving based on auto-save flag
                if analysis_result:
                    is_error = analysis_result.strip().startswith(("[red]", "[yellow]"))
                    border_col = "red" if analysis_result.strip().startswith("[red]") else "yellow" if analysis_result.strip().startswith("[yellow]") else "green"

                    self.console.print(Panel(
                        Markdown(analysis_result),
                        title="Analysis Report",
                        border_style=border_col,
                        expand=False
                    ))

                    # --- Saving Logic ---
                    if not is_error: # Only attempt to save successful reports
                        if self.args.no_auto_save:
                            # Prompt user because auto-save is disabled
                            if Confirm.ask("Save this analysis report?", default=True):
                                # Use the format from args as the default for the prompt
                                save_format = Prompt.ask("Save format?", choices=["markdown", "json"], default=self.args.format)
                                self._save_output(analysis_result, query, platform_names_list, save_format)
                        else:
                            # Auto-save is enabled (default behavior)
                            self.console.print(f"[cyan]Auto-saving analysis report as {self.args.format}...[/cyan]")
                            # Use the format specified in args (or its default)
                            self._save_output(analysis_result, query, platform_names_list, self.args.format)
                    # --- End Saving Logic ---

                else:
                    self.console.print("[red]Analysis returned no result.[/red]")

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Analysis query cancelled.[/yellow]")
                if Confirm.ask("\nExit this analysis session?", default=False):
                    break
            except RateLimitExceededError as rle:
                 # This might be caught if the LLM analysis call itself gets rate limited
                 self.console.print(f"\n[bold red]Rate Limit Error during analysis: {rle}[/bold red]")
                 self.console.print("[yellow]Please wait before trying again.[/yellow]")
                 # Optionally offer to exit or continue
                 if Confirm.ask("Exit analysis session due to rate limit?", default=False):
                     break
            except Exception as e:
                 logger.error(f"Unexpected error during analysis loop: {e}", exc_info=True)
                 self.console.print(f"\n[bold red]An unexpected error occurred during analysis:[/bold red] {e}")
                 # Decide whether to break or allow continuation
                 if not Confirm.ask("An error occurred. Continue session?", default=True):
                     break


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

            # Validate platform usernames are lists
            valid_platforms = {}
            available = self.get_available_platforms()
            for platform, usernames in platforms.items():
                 if platform not in available:
                     logger.warning(f"Platform '{platform}' specified in stdin is not configured or supported. Skipping.")
                     continue
                 if isinstance(usernames, str):
                     # Allow single string, convert to list
                     valid_platforms[platform] = [usernames] if usernames else []
                 elif isinstance(usernames, list) and all(isinstance(u, str) for u in usernames):
                     valid_platforms[platform] = [u for u in usernames if u] # Filter empty strings in list
                 else:
                     logger.warning(f"Invalid username format for platform '{platform}' in stdin. Expected string or list of strings. Skipping.")
                     continue # Skip platform with invalid username format


            if not valid_platforms:
                 raise ValueError("No valid platforms or usernames found in the processed input.")

            # Run analysis (will handle internal progress etc.)
            analysis_report = self.analyze(valid_platforms, query)

            if analysis_report:
                 is_error = analysis_report.strip().startswith(("[red]", "[yellow]"))

                 if not is_error:
                    # Analysis succeeded
                    platform_names_list = sorted(valid_platforms.keys())
                    if self.args.no_auto_save:
                        # Auto-save disabled: Print report to stdout instead of saving
                        self.console.print("[yellow]--no-auto-save flag set. Printing report to stdout instead of saving:[/yellow]")
                        self.console.print(Markdown(analysis_report)) # Print nicely formatted
                        sys.exit(0) # Success, but didn't save to file
                    else:
                        # Auto-save enabled (default): Save the output using the format from args
                        self._save_output(analysis_report, query, platform_names_list, self.args.format)
                        self.console.print(f"[green]Analysis complete. Output auto-saved ({self.args.format}).[/green]")
                        sys.exit(0) # Success exit code
                 else:
                    # Analysis failed or produced error message
                    self.console.print(f"[bold red]Analysis failed or produced an error report:[/bold red]")
                    self.console.print(analysis_report) # Print the error message from analyze()
                    sys.exit(1) # Failure exit code
            else:
                # Analysis returned nothing at all
                self.console.print(f"[bold red]Analysis returned no result.[/bold red]")
                sys.exit(1) # Failure exit code

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
             sys.exit(1) # Use a specific exit code?
        except Exception as e:
            logger.error(f"Unexpected error during stdin processing: {e}", exc_info=True)
            sys.stderr.write(f"Error: An unexpected error occurred - {e}\n")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Social Media OSINT analyzer using LLMs. Fetches user data from various platforms, performs text and image analysis, and generates reports.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve newline formatting in help
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
             '    "bluesky": ["handle1.bsky.social"]\n'
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
        action='store_true', # Sets args.no_auto_save to True if the flag is present
        help="Disable automatic saving of analysis reports; prompt the user before saving in interactive mode."
    )
    # Add verbosity?
    # parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase logging verbosity (e.g., -v, -vv)")

    args = parser.parse_args()

    # Configure logging level based on verbosity? (Example)
    # log_level = logging.WARN # Default
    # if args.verbose == 1: log_level = logging.INFO
    # elif args.verbose >= 2: log_level = logging.DEBUG
    # logging.getLogger('SocialOSINTLM').setLevel(log_level)
    # # Configure root logger too, or specific libraries if needed
    # logging.basicConfig(level=log_level, # Apply to root logger
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #                     handlers=[logging.FileHandler('analyzer.log'), logging.StreamHandler()])

    try:
        # Pass the parsed args to the constructor
        analyzer = SocialOSINTLM(args=args)
        if args.stdin:
            # process_stdin already uses self.args.format, but now needs self.args.no_auto_save check
            analyzer.process_stdin(args.format)
        else:
            # run -> _run_analysis_loop needs access to self.args
            analyzer.run()

    except RuntimeError as e:
         # Catch critical setup errors during initialization
         logging.getLogger('SocialOSINTLM').critical(f"Initialization failed: {e}", exc_info=False)
         sys.stderr.write(f"\n[bold red]CRITICAL ERROR:[/bold red] {e}\n")
         sys.stderr.write("Ensure necessary API keys (OpenRouter) and platform credentials are set in .env\n")
         sys.exit(1)
    except Exception as e:
         # Catch any other unexpected top-level errors
         logging.getLogger('SocialOSINTLM').critical(f"An unexpected critical error occurred: {e}", exc_info=True)
         sys.stderr.write(f"\n[bold red]UNEXPECTED CRITICAL ERROR:[/bold red] {e}\n")
         sys.exit(1)
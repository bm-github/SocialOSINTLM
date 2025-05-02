"""
SocialOSINTLM: OSINT analysis tool for social media using LLMs.

Collects data from Twitter, Reddit, Bluesky, Mastodon, and Hacker News,
analyzes text and images, and generates reports based on user queries.
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus, urlparse

import httpx
import praw
import prawcore
import tweepy
from atproto import Client
from atproto import exceptions as atproto_exceptions
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mastodon import (
    Mastodon,
    MastodonError,
    MastodonNotFoundError,
    MastodonRatelimitError,
    MastodonUnauthorizedError,
    MastodonVersionError,
)
from PIL import Image, UnidentifiedImageError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TaskID
from rich.prompt import Confirm, Prompt

load_dotenv()  # Load .env file if available

logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('analyzer.log'), logging.StreamHandler()],
)
logger = logging.getLogger('SocialOSINTLM')

# --- Constants ---
CACHE_EXPIRY_HOURS = 24
MAX_CACHE_ITEMS = 200  # Max tweets/posts/submissions per user/platform cache
REQUEST_TIMEOUT = 20.0  # Default timeout for HTTP requests
INITIAL_FETCH_LIMIT = 50  # How many items to fetch on first run/force_refresh
INCREMENTAL_FETCH_LIMIT = 50  # Items fetched during incremental updates
MASTODON_FETCH_LIMIT = 40  # Mastodon API max is often 40
LLM_ANALYSIS_TIMEOUT = 70.0 # Timeout for OpenRouter call + buffer
# Define supported image extensions as a static variable
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
# Define min datetime once for reuse
MIN_DATETIME_UTC = datetime.min.replace(tzinfo=timezone.utc)


# --- Custom Exceptions ---
class RateLimitExceededError(Exception):
    """Custom exception for API rate limit errors."""


class UserNotFoundError(Exception):
    """Custom exception for when a user is not found on a platform."""


class AccessForbiddenError(Exception):
    """Custom exception for access denied errors (private profile, block)."""


# --- JSON Encoder ---
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects."""

    def default(self, obj):
        """Convert datetime objects to ISO format strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# --- Helper Function (Global Scope) ---
def get_sort_key(item: Dict[str, Any], dt_key: str) -> datetime:
    """
    Safely get and parse a datetime string, object, or timestamp for sorting.

    Args:
        item: The dictionary containing the datetime information.
        dt_key: The key in the dictionary where the datetime value is stored.

    Returns:
        A timezone-aware datetime object, or a minimum datetime if parsing fails.
    """
    dt_val = item.get(dt_key)

    if isinstance(dt_val, str):
        try:
            # Handle 'Z' suffix for UTC timezone
            if dt_val.endswith('Z'):
                dt_val = dt_val[:-1] + '+00:00'
            # Ensure it's not an empty string before parsing
            if not dt_val:
                return MIN_DATETIME_UTC
            dt_obj = datetime.fromisoformat(dt_val)
            # Add UTC timezone if naive
            return dt_obj if dt_obj.tzinfo else dt_obj.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.debug(
                "Could not parse datetime string: '%s' for key '%s'. "
                "Using fallback.", dt_val, dt_key
            )
            return MIN_DATETIME_UTC
    elif isinstance(dt_val, datetime):
        # Add UTC timezone if naive
        return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    elif isinstance(dt_val, (int, float)):
        try:
            # Check for plausible timestamp range if needed
            if dt_val < 0:  # Negative timestamps are invalid
                logger.debug(
                    "Invalid negative timestamp: %s for key '%s'. Using fallback.",
                    dt_val, dt_key
                )
                return MIN_DATETIME_UTC
            return datetime.fromtimestamp(dt_val, tz=timezone.utc)
        # Catch potential errors during timestamp conversion
        except (ValueError, OSError, OverflowError):
            logger.debug(
                "Could not convert timestamp: %s for key '%s'. Using fallback.",
                dt_val, dt_key
            )
            return MIN_DATETIME_UTC

    logger.debug(
        "Using fallback datetime for key '%s' with value type: %s",
        dt_key, type(dt_val)
    )
    return MIN_DATETIME_UTC


# --- Main Class ---
class SocialOSINTLM:
    """
    Core class for fetching, caching, and analyzing social media data.
    """

    # Define supported image extensions as a class attribute
    supported_image_extensions = SUPPORTED_IMAGE_EXTENSIONS

    def __init__(self, args=None):
        """
        Initialize the analyzer.

        Args:
            args: Parsed command-line arguments (optional).
        """
        self.console = Console()
        self.base_dir = Path("data")
        self._setup_directories()
        self.progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            transient=True,
            console=self.console,
            refresh_per_second=10,
        )
        self.current_task: Optional[TaskID] = None
        # State variables for threaded analysis call
        self._analysis_response: Optional[httpx.Response] = None
        self._analysis_exception: Optional[Exception] = None
        self.args = args if args else argparse.Namespace()
        self._verify_env_vars()

    def _verify_env_vars(self):
        """Verify necessary environment variables are set."""
        required_core = ['OPENROUTER_API_KEY', 'IMAGE_ANALYSIS_MODEL']
        # Check for at least one platform credential set
        platforms_configured = any([
            all(os.getenv(k) for k in ['TWITTER_BEARER_TOKEN']),
            all(os.getenv(k) for k in [
                'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT'
            ]),
            all(os.getenv(k) for k in [
                'BLUESKY_IDENTIFIER', 'BLUESKY_APP_SECRET'
            ]),
            all(os.getenv(k) for k in [
                'MASTODON_ACCESS_TOKEN', 'MASTODON_API_BASE_URL'
            ]),
        ])
        # HN needs no keys, check if *any* platform capability exists
        has_any_platform = platforms_configured or 'hackernews' in \
            self.get_available_platforms(check_creds=False)

        if not has_any_platform:
            logger.warning(
                "No platform API credentials found in environment variables. "
                "Only HackerNews might work if explicitly requested."
            )
            self.console.print(
                "[bold yellow]Warning:[/bold yellow] No platform API credentials "
                "found. Only HackerNews analysis might be possible."
            )

        missing_core = [var for var in required_core if not os.getenv(var)]
        if missing_core:
            raise RuntimeError(
                f"Missing critical environment variables: {', '.join(missing_core)}"
            )

    def _setup_directories(self):
        """Create necessary data directories."""
        for dir_name in ['cache', 'media', 'outputs']:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # --- Property-based Client Initializers ---
    @property
    def bluesky(self) -> Client:
        """Lazy initialize and return the Bluesky client."""
        if not hasattr(self, '_bluesky_client'):
            identifier = os.getenv('BLUESKY_IDENTIFIER')
            secret = os.getenv('BLUESKY_APP_SECRET')
            if not identifier or not secret:
                raise RuntimeError(
                    "Bluesky credentials (BLUESKY_IDENTIFIER, "
                    "BLUESKY_APP_SECRET) not set in environment."
                )
            try:
                client = Client(request_timeout=REQUEST_TIMEOUT)
                client.login(identifier, secret)
                self._bluesky_client = client
                logger.debug("Bluesky login successful")
            except (KeyError, atproto_exceptions.AtProtocolError, RuntimeError) as e:
                logger.error("Bluesky setup failed: %s", e)
                raise RuntimeError(f"Bluesky setup failed: {e}") from e
        return self._bluesky_client

    @property
    def mastodon(self) -> Mastodon:
        """Lazy initialize and return the Mastodon client."""
        if not hasattr(self, '_mastodon_client'):
            token = os.getenv('MASTODON_ACCESS_TOKEN')
            base_url = os.getenv('MASTODON_API_BASE_URL')
            if not token or not base_url:
                raise RuntimeError(
                    "Mastodon credentials (MASTODON_ACCESS_TOKEN, "
                    "MASTODON_API_BASE_URL) not set."
                )
            try:
                # Validate URL format
                parsed_url = urlparse(base_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise RuntimeError(
                        f"Invalid MASTODON_API_BASE_URL format: {base_url}. "
                        "Should be like 'https://mastodon.social'."
                    )

                client = Mastodon(
                    access_token=token,
                    api_base_url=base_url,
                    request_timeout=REQUEST_TIMEOUT,
                )
                # Test connection by getting instance info
                client.instance()
                self._mastodon_client = client
                logger.debug("Mastodon client initialized for %s.", base_url)
            except (KeyError, MastodonError, RuntimeError) as e:
                logger.error("Mastodon setup failed: %s", e)
                raise RuntimeError(f"Mastodon setup failed: {e}") from e
        return self._mastodon_client

    @property
    def openrouter(self) -> httpx.Client:
        """Lazy initialize and return the OpenRouter httpx client."""
        if not hasattr(self, '_openrouter'):
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise RuntimeError("Missing OpenRouter API key (OPENROUTER_API_KEY)")
            try:
                self._openrouter = httpx.Client(
                    base_url="https://openrouter.ai/api/v1",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "http://localhost:3000",  # Replace if needed
                        "X-Title": "Social Media analyzer",
                        "Content-Type": "application/json",
                    },
                    timeout=LLM_ANALYSIS_TIMEOUT, # Use specific timeout for LLM
                )
            except Exception as e: # Catch potential httpx setup errors too
                raise RuntimeError(
                    f"Failed to initialize OpenRouter client: {e}"
                ) from e
        return self._openrouter

    @property
    def reddit(self) -> praw.Reddit:
        """Lazy initialize and return the PRAW Reddit client."""
        if not hasattr(self, '_reddit'):
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            user_agent = os.getenv('REDDIT_USER_AGENT')
            if not all([client_id, client_secret, user_agent]):
                raise RuntimeError(
                    "Reddit credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, "
                    "REDDIT_USER_AGENT) not fully set in environment."
                )
            try:
                self._reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                    read_only=True,  # Explicitly set read-only mode
                    requestor_kwargs={"timeout": REQUEST_TIMEOUT},  # Set timeout
                )
                self._reddit.auth.scopes()  # Test connection/auth early
                logger.debug("Reddit client initialized.")
            except (
                KeyError, prawcore.exceptions.OAuthException,
                prawcore.exceptions.ResponseException, RuntimeError
            ) as e:
                logger.error("Reddit setup failed: %s", e)
                raise RuntimeError(f"Reddit setup failed: {e}") from e
        return self._reddit

    @property
    def twitter(self) -> tweepy.Client:
        """Lazy initialize and return the Tweepy client."""
        if not hasattr(self, '_twitter'):
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            if not bearer_token:
                raise RuntimeError(
                    "Twitter Bearer Token (TWITTER_BEARER_TOKEN) not set."
                )
            try:
                self._twitter = tweepy.Client(
                    bearer_token=bearer_token,
                    wait_on_rate_limit=False,
                    # Tweepy passes timeout to underlying request lib (httpx)
                    timeout=REQUEST_TIMEOUT,
                )
                # Test connection using a known, public account
                self._twitter.get_user(username="twitterdev")
                logger.debug("Twitter client initialized.")
            except (KeyError, tweepy.errors.TweepyException, RuntimeError) as e:
                logger.error("Twitter setup failed: %s", e)
                raise RuntimeError(f"Twitter setup failed: {e}") from e
        return self._twitter

    # --- Utility Methods ---
    def _handle_rate_limit(
        self, platform: str, exception: Optional[Exception] = None
    ):
        """Handles API rate limit errors, logs info, and raises custom error."""
        error_message = f"{platform} API rate limit exceeded."
        reset_info = ""
        wait_seconds = 900  # Default wait 15 mins if unknown

        if isinstance(exception, tweepy.TooManyRequests):
            rate_limit_reset = exception.response.headers.get('x-rate-limit-reset')
            if rate_limit_reset:
                try:
                    reset_ts = int(rate_limit_reset)
                    reset_time = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
                    current_time = datetime.now(timezone.utc)
                    # Add 5s buffer, ensure wait is at least 1 second
                    wait_seconds = max(
                        int((reset_time - current_time).total_seconds()) + 5, 1
                    )
                    reset_info = (f"Try again after: "
                                  f"{reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                except (ValueError, TypeError):
                    reset_info = f"Wait ~{wait_seconds // 60} mins before retrying."
                    logger.debug("Could not parse rate limit reset time.")
            else:
                reset_info = f"Wait ~{wait_seconds // 60} mins before retrying."
        elif isinstance(exception, (prawcore.exceptions.RequestException,
                                    httpx.HTTPStatusError)):
            # Check specifically for 429
            if (hasattr(exception, 'response') and
                    exception.response is not None and
                    exception.response.status_code == 429):
                reset_info = f"Wait ~{wait_seconds // 60} mins before retrying."
            else:
                # Re-raise other HTTP errors if not 429
                logger.error("Unhandled HTTP Error for %s: %s", platform, exception)
                raise exception
        elif isinstance(exception, MastodonRatelimitError):
            reset_info = f"Wait ~{wait_seconds // 60} mins before retrying."
            if (hasattr(exception, 'response') and
                    hasattr(exception.response, 'headers')):
                logger.debug(
                    "Mastodon rate limit headers: %s", exception.response.headers
                )
            else:
                logger.debug(
                    "Mastodon rate limit error, but no specific headers found."
                )
        elif (isinstance(exception, atproto_exceptions.AtProtocolError) and
              'rate limit' in str(exception).lower()):
            reset_info = f"Wait ~{wait_seconds // 60} mins before retrying."
        else:
            # Default wait for unknown rate limit scenarios
            reset_info = f"Wait ~{wait_seconds // 60} mins before retrying."

        self.console.print(Panel(
            f"[bold red]Rate Limit Blocked: {platform}[/bold red]\n"
            f"{error_message}\n"
            f"{reset_info}",
            title="🚫 Rate Limit",
            border_style="red",
        ))
        # Raise specific error after logging
        raise RateLimitExceededError(f"{error_message} ({reset_info})")

    def _get_media_path(self, url: str) -> Path:
        """
        Generates a consistent path stub for media based on URL hash.
        Platform/username are not needed here as hash ensures uniqueness.
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        # Use a generic extension initially; will be refined upon download.
        return self.base_dir / 'media' / f"{url_hash}.media"

    def _download_media(
        self, url: str, platform: str, username: str,
        headers: Optional[dict] = None
    ) -> Optional[Path]:
        """
        Downloads media, saves with correct extension, returns path if successful.

        Args:
            url: The URL of the media to download.
            platform: The platform the media originated from (for context/auth).
            username: The username associated with the media (for context).
            headers: Optional additional headers for the request.

        Returns:
            The Path object to the downloaded file, or None if download failed.
        """
        media_path_stub = self._get_media_path(url)
        media_dir = self.base_dir / 'media'

        # Check if any file with this hash stem exists (might have diff extensions)
        existing_files = list(media_dir.glob(f'{media_path_stub.stem}.*'))
        if existing_files:
            # Prefer common image types if multiple exist
            common_image_exts = ['.jpg', '.png', '.webp', '.gif']
            for ext in common_image_exts:
                found_path = media_dir / f"{media_path_stub.stem}{ext}"
                if found_path.exists():
                    logger.debug("Media cache hit (specific ext): %s", found_path)
                    return found_path
            # Return the first one found if no preferred type exists
            logger.debug("Media cache hit (generic): %s", existing_files[0])
            return existing_files[0]

        valid_types = {
            'image/jpeg': '.jpg', 'image/png': '.png',
            'image/gif': '.gif', 'image/webp': '.webp',
            'video/mp4': '.mp4', 'video/webm': '.webm', # Common video types
        }
        final_media_path = None
        download_headers = headers.copy() if headers else {} # Work on a copy

        try:
            # --- Platform-specific AUTHENTICATION Adjustments ---
            # Mastodon media URLs are typically public CDN links, no auth needed.
            if platform == 'twitter':
                self.twitter # Ensure client ready via property access
                bearer = os.getenv("TWITTER_BEARER_TOKEN")
                if not bearer:
                    raise RuntimeError("TWITTER_BEARER_TOKEN not found for download.")
                download_headers['Authorization'] = f'Bearer {bearer}'
            elif platform == 'bluesky':
                self.bluesky # Ensure client ready via property access
                # Accessing protected member _session is necessary here
                # TODO: Check if atproto lib offers a cleaner way in future versions
                session = getattr(self.bluesky, '_session', None)
                access_token = getattr(session, 'access_jwt', None) if session else None
                if not access_token:
                    raise RuntimeError("Bluesky access token missing for download.")
                download_headers['Authorization'] = f"Bearer {access_token}"
                # Known Bluesky CDN issue sometimes returns http, force https
                url = url.replace('http://', 'https://')

            # --- Download ---
            with httpx.Client(follow_redirects=True,
                              timeout=REQUEST_TIMEOUT) as client:
                resp = client.get(url, headers=download_headers)
                resp.raise_for_status() # Check for 4xx/5xx errors

            content_type = resp.headers.get('content-type', '').lower().split(';')[0]
            extension = valid_types.get(content_type)

            if not extension:
                logger.warning(
                    "Unsupported or non-media type '%s' for URL: %s",
                    content_type, url
                )
                return None # Might be HTML page, etc.

            final_media_path = media_path_stub.with_suffix(extension)
            final_media_path.write_bytes(resp.content)
            logger.debug("Downloaded media to: %s", final_media_path)
            return final_media_path

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Use platform context, raises RateLimitExceededError
                self._handle_rate_limit(f"{platform} Media Download", e)
            elif e.response.status_code in [404, 403, 401]:
                logger.warning(
                    "Media access error (%d) for %s. Skipping.",
                    e.response.status_code, url
                )
            else:
                logger.error(
                    "HTTP error %d downloading %s: %s",
                    e.response.status_code, url, e
                )
            return None
        except RuntimeError as e: # Catch auth errors raised above
            logger.error("Media download failed for %s: %s", url, e)
            return None
        except Exception as e:
            logger.error(
                "Unexpected media download failed for %s: %s", url, str(e),
                exc_info=False # Keep log concise for download errors
            )
            return None

    def _analyze_image(self, file_path: Path, context: str = "") -> Optional[str]:
        """
        Analyzes image using OpenRouter, handles resizing, format conversion,
        and errors.

        Args:
            file_path: Path to the local image file.
            context: String providing context about the image's origin.

        Returns:
            A string containing the analysis text, or None if analysis fails.
        """
        if not file_path or not file_path.exists():
            logger.warning(
                "Image analysis skipped: file path invalid or missing (%s)",
                file_path
            )
            return None

        # Use class attribute for supported extensions
        if file_path.suffix.lower() not in self.supported_image_extensions:
            logger.debug(
                "Skipping analysis for non-image file (ext %s not supported): %s",
                file_path.suffix, file_path
            )
            return None

        temp_path: Optional[Path] = None # Track temporary file for cleanup
        analysis_file_path = file_path # Default to original path

        try:
            with Image.open(file_path) as img:
                img_format_lower = img.format.lower() if img.format else ''
                # Double-check format beyond extension
                if img_format_lower not in ['jpeg', 'png', 'webp', 'gif']:
                    logger.warning(
                        "Unsupported image type for analysis: %s at %s",
                        img.format, file_path
                    )
                    return None

                # --- Image Preprocessing for Vision Model ---
                max_dimension = 1536 # Common dimension limit for vision models
                width, height = img.size
                scale_factor = min(max_dimension / width, max_dimension / height, 1.0)

                needs_processing = False
                is_animated_gif = (img_format_lower == 'gif' and
                                   getattr(img, 'is_animated', False))

                # Resize if too large
                if scale_factor < 1.0:
                    needs_processing = True
                    new_size = (int(width * scale_factor), int(height * scale_factor))
                    if is_animated_gif:
                        img.seek(0) # Analyze only the first frame
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.debug("Resized image for analysis: %s", file_path)

                # Convert format if not JPEG or has alpha/palette
                # Vision models often prefer JPEG, RGB
                elif img_format_lower != 'jpeg' or img.mode in ("RGBA", "P"):
                    needs_processing = True
                    if is_animated_gif:
                        img.seek(0)
                    if img.mode in ("RGBA", "P"):
                        # Ensure conversion to RGB which is widely supported
                        img = img.convert("RGB")
                    logger.debug("Converting image to JPEG/RGB for analysis: %s", file_path)

                # Save processed image temporarily if changes were made
                if needs_processing:
                    # Use JPEG format for the processed file
                    temp_suffix = f'.processed{file_path.suffix}.jpg'
                    temp_path = file_path.with_suffix(temp_suffix)
                    # Adjust quality based on whether resizing occurred
                    jpeg_quality = 85 if scale_factor < 1.0 else 90
                    img.save(temp_path, 'JPEG', quality=jpeg_quality)
                    analysis_file_path = temp_path # Use temp file for analysis
                # else: analysis_file_path remains the original file_path

            # --- Prepare API Request ---
            base64_image = base64.b64encode(
                analysis_file_path.read_bytes()
            ).decode('utf-8')

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

            model_to_use = os.getenv('IMAGE_ANALYSIS_MODEL')

            # Ensure OpenRouter client is ready
            self.openrouter # Access property

            # --- Make API Call ---
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
                                "detail": "high", # Request high detail analysis
                            }}
                        ]
                    }],
                    "max_tokens": 1024, # Allow longer response
                }
            )
            response.raise_for_status() # Check for HTTP errors (4xx/5xx)
            result = response.json()

            # --- Process API Response ---
            # Check for API-level errors sometimes returned in 200 OK
            if 'error' in result:
                error_msg = result['error'].get('message', 'Unknown API error')
                logger.error("Image analysis API error: %s", error_msg)
                # Handle specific errors like content filtering gracefully
                if 'block' in error_msg.lower():
                    return "[Image analysis blocked by content filter]"
                return None # Other API errors
            # Validate response structure
            if (not result.get('choices') or
                    not result['choices'][0].get('message') or
                    'content' not in result['choices'][0]['message']):
                logger.error("Invalid image analysis API response structure: %s", result)
                return None

            analysis_text = result['choices'][0]['message']['content']
            logger.debug("Image analysis successful for: %s", file_path)
            return analysis_text.strip()

        # --- Error Handling ---
        except (IOError, Image.DecompressionBombError, UnidentifiedImageError,
                SyntaxError) as img_err:
            logger.error("Image processing error for %s: %s", file_path, img_err)
            return None
        except httpx.RequestError as req_err:
            # Network errors are often transient, log but don't raise fatal
            logger.error(
                "Network error during image analysis API call: %s", req_err
            )
            return None
        except httpx.HTTPStatusError as status_err:
            model_name = os.getenv('IMAGE_ANALYSIS_MODEL', 'default_vision')
            status_code = status_err.response.status_code
            if status_code == 429:
                # Let _handle_rate_limit raise RateLimitExceededError
                self._handle_rate_limit(f"Image Analysis ({model_name})", status_err)
            elif status_code == 401:
                logger.error(
                    "HTTP 401 Unauthorized during image analysis (%s). "
                    "Check OPENROUTER_API_KEY.", model_name
                )
                logger.error("API Response: %s", status_err.response.text)
            else:
                logger.error(
                    "HTTP error %d during image analysis: %s",
                    status_code, status_err.response.text
                )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error during image analysis for %s: %s",
                file_path, e, exc_info=True # Log full traceback
            )
            return None
        finally:
            # Clean up temporary file if created
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug("Deleted temporary analysis file: %s", temp_path)
                except OSError as e:
                    logger.warning(
                        "Could not delete temporary analysis file %s: %s",
                        temp_path, e
                    )

    # --- Cache Management ---
    @lru_cache(maxsize=128) # Cache path generation results
    def _get_cache_path(self, platform: str, username: str) -> Path:
        """
        Generates a sanitized cache file path for a given platform and username.
        """
        # Sanitize username (esp. for Mastodon user@instance)
        safe_username = "".join(
            c if c.isalnum() or c in ['-', '_', '.', '@'] else '_'
            for c in username
        )
        return self.base_dir / 'cache' / f"{platform}_{safe_username}.json"

    def _load_cache(self, platform: str, username: str) -> Optional[Dict[str, Any]]:
        """
        Loads cache data if it exists, is valid, and not expired.
        Returns expired data if found (for incremental updates).
        """
        cache_path = self._get_cache_path(platform, username)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            timestamp_str = data.get('timestamp')
            if not timestamp_str:
                logger.warning(
                    "Cache for %s/%s missing timestamp. Discarding.",
                    platform, username
                )
                cache_path.unlink(missing_ok=True)
                return None

            timestamp = datetime.fromisoformat(timestamp_str)
            now_utc = datetime.now(timezone.utc)
            is_expired = now_utc - timestamp >= timedelta(hours=CACHE_EXPIRY_HOURS)

            # Define expected keys for cache integrity check
            platform_keys = {
                'twitter': ['tweets', 'user_info', 'media_analysis', 'media_paths'],
                'reddit': ['submissions', 'comments', 'stats',
                           'media_analysis', 'media_paths'],
                'bluesky': ['posts', 'stats', 'profile_info',
                            'media_analysis', 'media_paths'],
                'hackernews': ['submissions', 'stats'],
                'mastodon': ['posts', 'user_info', 'stats',
                             'media_analysis', 'media_paths'],
            }
            required_keys = ['timestamp'] + platform_keys.get(platform, [])

            # Check for required keys
            if not all(key in data for key in required_keys):
                missing = [key for key in required_keys if key not in data]
                logger.warning(
                    "Cache for %s/%s incomplete (missing: %s). Discarding.",
                    platform, username, missing
                )
                cache_path.unlink(missing_ok=True)
                return None

            # Return data if not expired, otherwise return expired data baseline
            if not is_expired:
                logger.debug("Cache hit for %s/%s", platform, username)
                return data
            else:
                logger.info("Cache expired for %s/%s", platform, username)
                return data # Return expired data for incremental update

        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError) as e:
            logger.warning(
                "Failed to load/parse cache for %s/%s: %s. Discarding cache.",
                platform, username, e
            )
            cache_path.unlink(missing_ok=True)
            return None
        except Exception as e:
            # Catch any other unexpected errors during cache load
            logger.error(
                "Unexpected error loading cache for %s/%s: %s",
                platform, username, e, exc_info=True
            )
            cache_path.unlink(missing_ok=True)
            return None

    def _save_cache(self, platform: str, username: str, data: Dict[str, Any]):
        """Saves data to cache, ensuring timestamp is updated and lists sorted."""
        cache_path = self._get_cache_path(platform, username)
        try:
            # Ensure main list(s) are sorted newest first before saving
            sort_key_map = {
                'twitter': [('tweets', 'created_at')],
                'reddit': [('submissions', 'created_utc'),
                           ('comments', 'created_utc')],
                'bluesky': [('posts', 'created_at')],
                'hackernews': [('submissions', 'created_at')], # HN uses datetime obj
                'mastodon': [('posts', 'created_at')],
            }

            # Sort relevant lists using the global helper function
            if platform in sort_key_map:
                items_to_sort_config = sort_key_map[platform]
                for list_key, dt_key in items_to_sort_config:
                    if (list_key in data and
                            isinstance(data[list_key], list) and data[list_key]):
                        data[list_key].sort(
                            key=lambda x: get_sort_key(x, dt_key), reverse=True
                        )

            # Always update timestamp before saving
            data['timestamp'] = datetime.now(timezone.utc).isoformat()

            # Write to file using the custom DateTimeEncoder
            cache_path.write_text(
                json.dumps(data, indent=2, cls=DateTimeEncoder),
                encoding='utf-8'
            )
            logger.debug("Saved cache for %s/%s", platform, username)
        except Exception as e:
            logger.error(
                "Failed to save cache for %s/%s: %s",
                platform, username, e, exc_info=True
            )

    # --- Platform Fetch Methods (with Incremental Logic) ---

    def fetch_twitter(self, username: str, force_refresh: bool = False
                     ) -> Optional[Dict[str, Any]]:
        """
        Fetches Twitter user info and tweets, using cache and incremental updates.

        Args:
            username: Twitter username (without '@').
            force_refresh: If True, ignore cache and fetch all data fresh.

        Returns:
            A dictionary containing user info, tweets, and media analysis,
            or None if fetching fails.
        """
        cached_data = self._load_cache('twitter', username)
        now_utc = datetime.now(timezone.utc)

        # Use cache if valid, recent, and not forced refresh
        if not force_refresh and cached_data and \
           (now_utc - datetime.fromisoformat(cached_data['timestamp'])) < \
           timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info("Using recent cache for Twitter @%s", username)
            return cached_data

        logger.info(
            "Fetching Twitter data for @%s (Force Refresh: %s)",
            username, force_refresh
        )
        since_id = None
        existing_tweets = []
        existing_media_analysis = []
        existing_media_paths = []
        user_info = None

        # Use cached data as baseline for incremental fetch
        if not force_refresh and cached_data:
            logger.info("Attempting incremental fetch for Twitter @%s", username)
            existing_tweets = cached_data.get('tweets', [])
            user_info = cached_data.get('user_info')
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])
            # Sort existing to find the latest ID reliably
            existing_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'),
                                 reverse=True)
            if existing_tweets:
                since_id = existing_tweets[0].get('id')
                if since_id:
                    logger.debug("Using since_id: %s", since_id)
            else:
                logger.debug("No existing tweets found for incremental fetch.")

        try:
            # Fetch user info if missing or forced refresh
            if not user_info or force_refresh:
                try:
                    self.twitter # Ensure client ready
                    user_fields = ['created_at', 'public_metrics',
                                   'profile_image_url', 'description',
                                   'location', 'verified', 'name', 'username']
                    user_response = self.twitter.get_user(username=username,
                                                          user_fields=user_fields)
                    if not user_response or not user_response.data:
                        raise UserNotFoundError(f"Twitter user @{username} not found.")

                    user = user_response.data
                    created_at = user.created_at
                    user_info = {
                        'id': user.id, 'name': user.name, 'username': user.username,
                        'created_at': created_at.isoformat() if created_at else None,
                        'public_metrics': user.public_metrics,
                        'profile_image_url': user.profile_image_url,
                        'description': user.description, 'location': user.location,
                        'verified': user.verified,
                    }
                    logger.debug("Fetched user info for @%s", username)
                except tweepy.NotFound:
                    raise UserNotFoundError(f"Twitter user @{username} not found.")
                except tweepy.Forbidden:
                    raise AccessForbiddenError(
                        f"Access forbidden to Twitter user @{username}'s profile."
                    )

            user_id = user_info['id']
            new_tweets_data: List[tweepy.Tweet] = []
            new_media_includes: Dict[str, List[Any]] = {}
            # Determine fetch limit based on refresh type
            fetch_limit = (INITIAL_FETCH_LIMIT if (force_refresh or not since_id)
                           else INCREMENTAL_FETCH_LIMIT)
            pagination_token = None
            tweets_fetch_count = 0
            max_pages = 5 # Limit pagination requests

            # --- Paginate through tweets ---
            while tweets_fetch_count < fetch_limit and max_pages > 0:
                max_pages -= 1
                current_page_limit = min(fetch_limit - tweets_fetch_count, 100) # Max 100
                if current_page_limit <= 0:
                    break

                try:
                    self.twitter # Ensure client ready
                    tweet_fields = ['created_at', 'public_metrics',
                                    'attachments', 'entities',
                                    'conversation_id', 'in_reply_to_user_id']
                    expansions = ['attachments.media_keys', 'author_id']
                    media_fields = ['url', 'preview_image_url', 'type',
                                    'media_key', 'width', 'height', 'alt_text']

                    tweets_response = self.twitter.get_users_tweets(
                        id=user_id, max_results=current_page_limit,
                        since_id=since_id if not force_refresh else None,
                        pagination_token=pagination_token,
                        tweet_fields=tweet_fields, expansions=expansions,
                        media_fields=media_fields
                    )
                except tweepy.TooManyRequests as e:
                    self._handle_rate_limit('Twitter', exception=e) # Raises exception
                except tweepy.NotFound:
                    raise UserNotFoundError(
                        f"Tweets not found for user ID {user_id} (@{username}). "
                        "User might be protected or deleted."
                    )
                except tweepy.Forbidden as e:
                    raise AccessForbiddenError(
                        f"Access forbidden to @{username}'s tweets. Details: {e}"
                    )

                # Process fetched data
                if tweets_response.data:
                    page_count = len(tweets_response.data)
                    new_tweets_data.extend(tweets_response.data)
                    tweets_fetch_count += page_count
                    logger.debug(
                        "Fetched %d tweets page (Total this run: %d).",
                        page_count, tweets_fetch_count
                    )
                # Consolidate unique media objects from includes
                if tweets_response.includes and 'media' in tweets_response.includes:
                    if 'media' not in new_media_includes:
                        new_media_includes['media'] = []
                    # Avoid duplicates if pagination returns overlapping includes
                    existing_keys = {item.get('media_key') for item in
                                     new_media_includes['media'] if item}
                    for item in tweets_response.includes['media']:
                        if item and item.media_key not in existing_keys:
                            new_media_includes['media'].append(item)
                            existing_keys.add(item.media_key)

                # Get next page token
                pagination_token = tweets_response.meta.get('next_token')
                if not pagination_token:
                    logger.debug("No more tweet pages found.")
                    break
            logger.info(
                "Fetched %d total new tweets for @%s.", tweets_fetch_count, username
            )

            # --- Process New Tweets and Media ---
            processed_new_tweets = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            # Create a lookup for faster media object access
            all_media_objects = {
                m.media_key: m for m in new_media_includes.get('media', []) if m
            }

            for tweet in new_tweets_data:
                media_items_for_tweet = []
                if tweet.attachments and 'media_keys' in tweet.attachments:
                    for media_key in tweet.attachments['media_keys']:
                        media = all_media_objects.get(media_key)
                        if media:
                            # Prefer full URL for photos/gifs, preview for videos
                            url = (media.url if media.type in ['photo', 'gif']
                                   else media.preview_image_url)
                            if url:
                                # Download and potentially analyze media
                                media_path = self._download_media(
                                    url=url, platform='twitter', username=username
                                )
                                if media_path:
                                    analysis = None
                                    # Analyze only supported image types
                                    if media_path.suffix.lower() in \
                                       self.supported_image_extensions:
                                        analysis = self._analyze_image(
                                            media_path,
                                            f"Twitter user @{username}'s "
                                            f"tweet ({tweet.id})"
                                        )
                                    media_items_for_tweet.append({
                                        'type': media.type,
                                        'analysis': analysis,
                                        'url': url, # Store original URL fetched
                                        'alt_text': media.alt_text,
                                        'local_path': str(media_path)
                                    })
                                    if analysis:
                                        newly_added_media_analysis.append(analysis)
                                    newly_added_media_paths.add(str(media_path))
                                else:
                                    logger.warning("Failed to download media: %s", url)

                # Structure tweet data for caching
                tweet_created_at = tweet.created_at
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': (tweet_created_at.isoformat()
                                   if tweet_created_at else None),
                    'metrics': tweet.public_metrics,
                    'entities': tweet.entities,
                    'media': media_items_for_tweet,
                    'conversation_id': tweet.conversation_id,
                    'in_reply_to_user_id': tweet.in_reply_to_user_id,
                }
                processed_new_tweets.append(tweet_data)

            # --- Combine, Sort, Trim, and Save ---
            existing_ids = {t['id'] for t in existing_tweets}
            unique_new_tweets = [t for t in processed_new_tweets
                                 if t['id'] not in existing_ids]

            combined_tweets = unique_new_tweets + existing_tweets
            combined_tweets.sort(key=lambda x: get_sort_key(x, 'created_at'),
                                 reverse=True)
            # Trim to max cache size
            final_tweets = combined_tweets[:MAX_CACHE_ITEMS]

            # Combine media analysis/paths, ensuring uniqueness and trimming
            final_media_analysis = list(dict.fromkeys(
                newly_added_media_analysis + existing_media_analysis
            ))[:MAX_CACHE_ITEMS] # Limit analysis entries too
            final_media_paths = sorted(list(
                newly_added_media_paths.union(existing_media_paths)
            ))[:MAX_CACHE_ITEMS * 2] # Allow more paths than tweets

            # Final data structure for cache
            final_data = {
                'timestamp': now_utc.isoformat(),
                'user_info': user_info,
                'tweets': final_tweets,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths
            }
            self._save_cache('twitter', username, final_data)
            logger.info(
                "Successfully updated Twitter cache for @%s. "
                "Total tweets cached: %d", username, len(final_tweets)
            )
            return final_data

        # --- Error Handling for Fetch ---
        except RateLimitExceededError:
            return None # Already handled by _handle_rate_limit
        except (UserNotFoundError, AccessForbiddenError) as user_err:
            logger.error("Twitter fetch failed for @%s: %s", username, user_err)
            return None
        except tweepy.errors.TweepyException as e:
            logger.error("Twitter API error for @%s: %s", username, e)
            # Check for critical auth errors
            err_str = str(e).lower()
            if "authentication credentials" in err_str or "bearer token" in err_str:
                raise RuntimeError(f"Twitter auth failed: {e}") from e
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching Twitter data for @%s: %s",
                username, e, exc_info=True
            )
            return None


    def fetch_reddit(self, username: str, force_refresh: bool = False
                    ) -> Optional[Dict[str, Any]]:
        """
        Fetches Reddit user info, submissions, and comments.

        Args:
            username: Reddit username (without 'u/').
            force_refresh: If True, ignore cache and fetch all data fresh.

        Returns:
            A dictionary containing profile stats, submissions, comments,
            and media analysis, or None if fetching fails.
        """
        cached_data = self._load_cache('reddit', username)
        now_utc = datetime.now(timezone.utc)

        # Use cache if valid, recent, and not forced refresh
        if not force_refresh and cached_data and \
           (now_utc - datetime.fromisoformat(cached_data['timestamp'])) < \
           timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info("Using recent cache for Reddit u/%s", username)
            return cached_data

        logger.info(
            "Fetching Reddit data for u/%s (Force Refresh: %s)",
            username, force_refresh
        )
        latest_submission_fullname = None
        latest_comment_fullname = None
        existing_submissions = []
        existing_comments = []
        existing_media_analysis = []
        existing_media_paths = []

        # Baseline from cache for incremental fetch
        if not force_refresh and cached_data:
            logger.info("Attempting incremental fetch for Reddit u/%s", username)
            existing_submissions = cached_data.get('submissions', [])
            existing_comments = cached_data.get('comments', [])
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])
            # Sort to find latest fullnames
            existing_submissions.sort(key=lambda x: get_sort_key(x, 'created_utc'),
                                      reverse=True)
            existing_comments.sort(key=lambda x: get_sort_key(x, 'created_utc'),
                                   reverse=True)
            if existing_submissions:
                latest_submission_fullname = existing_submissions[0].get('fullname')
                logger.debug("Latest sub fullname: %s", latest_submission_fullname)
            if existing_comments:
                latest_comment_fullname = existing_comments[0].get('fullname')
                logger.debug("Latest comment fullname: %s", latest_comment_fullname)

        try:
            self.reddit # Ensure client ready
            redditor = self.reddit.redditor(username)
            try:
                # Fetch basic user info/status early
                redditor_info = {
                    'id': redditor.id,
                    'name': redditor.name,
                    'created_utc': getattr(redditor, 'created_utc', None),
                    'link_karma': getattr(redditor, 'link_karma', 0),
                    'comment_karma': getattr(redditor, 'comment_karma', 0),
                    'is_suspended': getattr(redditor, 'is_suspended', False),
                }
                logger.debug(
                    "Reddit user u/%s found (ID: %s). Suspended: %s",
                    username, redditor_info['id'], redditor_info['is_suspended']
                )
                if redditor_info['is_suspended']:
                    raise AccessForbiddenError(f"Reddit user u/{username} is suspended.")
            except prawcore.exceptions.NotFound:
                raise UserNotFoundError(f"Reddit user u/{username} not found.")
            except prawcore.exceptions.Forbidden:
                raise AccessForbiddenError(
                    f"Access forbidden to Reddit user u/{username} (shadowbanned?)."
                )
            except AttributeError as ae:
                # Handle cases where some attributes might be missing (rare)
                logger.warning(
                    "Could not fetch all attributes for u/%s: %s", username, ae
                )
                redditor_info = {'name': username} # Fallback


            new_submissions_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            # Fetch slightly more for potential overlap in incremental
            fetch_limit = INCREMENTAL_FETCH_LIMIT + 10
            sub_count = 0
            processed_submission_ids = {s['id'] for s in existing_submissions}

            logger.debug("Fetching new Reddit submissions...")
            try:
                params = {}
                # Use 'before' for incremental fetching (newest items)
                if not force_refresh and latest_submission_fullname:
                    params['before'] = latest_submission_fullname
                # Iterate through new submissions
                for submission in redditor.submissions.new(limit=fetch_limit,
                                                          params=params):
                    sub_count += 1
                    if submission.id in processed_submission_ids:
                        continue # Already have this one

                    media_items = []
                    media_processed_inline = False
                    sub_url = getattr(submission, 'url', '')
                    # Check if submission URL itself points to media
                    if sub_url and any(sub_url.lower().endswith(ext) for ext in
                                      self.supported_image_extensions +
                                      ['.mp4', '.webm']):
                        media_path = self._download_media(url=sub_url,
                                                          platform='reddit',
                                                          username=username)
                        if media_path:
                            analysis = None
                            if media_path.suffix.lower() in \
                               self.supported_image_extensions:
                                analysis = self._analyze_image(
                                    media_path,
                                    f"Reddit user u/{username}'s post in "
                                    f"r/{submission.subreddit.display_name}"
                                )
                            media_items.append({
                                'type': ('image' if media_path.suffix.lower()
                                         in self.supported_image_extensions
                                         else 'video'),
                                'analysis': analysis, 'url': sub_url,
                                'local_path': str(media_path)
                            })
                            if analysis: newly_added_media_analysis.append(analysis)
                            newly_added_media_paths.add(str(media_path))
                            media_processed_inline = True
                        else:
                            logger.warning("Failed media download: %s", sub_url)

                    # Handle Reddit galleries if not already processed
                    is_gallery = getattr(submission, 'is_gallery', False)
                    media_metadata = getattr(submission, 'media_metadata', None)
                    if not media_processed_inline and is_gallery and media_metadata:
                        gallery_items = []
                        for media_id, media_item in media_metadata.items():
                            source = media_item.get('s') # Source object
                            if source:
                                # Prefer direct image URL ('u') or GIF URL ('gif')
                                image_url_raw = source.get('u') or source.get('gif')
                                if image_url_raw:
                                    # Reddit URLs might have HTML entities
                                    image_url = BeautifulSoup(
                                        image_url_raw, "html.parser"
                                    ).text
                                    media_path = self._download_media(
                                        url=image_url, platform='reddit',
                                        username=username
                                    )
                                    if media_path:
                                        analysis = None
                                        if media_path.suffix.lower() in \
                                           self.supported_image_extensions:
                                            analysis = self._analyze_image(
                                                media_path,
                                                f"Reddit user u/{username}'s "
                                                f"gallery post in "
                                                f"r/{submission.subreddit.display_name}"
                                            )
                                        gallery_items.append({
                                            'type': 'gallery_image',
                                            'analysis': analysis, 'url': image_url,
                                            'local_path': str(media_path)
                                        })
                                        if analysis:
                                            newly_added_media_analysis.append(analysis)
                                        newly_added_media_paths.add(str(media_path))
                                    else:
                                        logger.warning(
                                            "Failed gallery download: %s", image_url
                                        )
                        if gallery_items:
                            media_items = gallery_items # Overwrite if gallery found

                    submission_text = getattr(submission, 'selftext', '')
                    submission_data = {
                        'id': submission.id, 'title': submission.title,
                        'text': submission_text[:1500], # Limit text length
                        'score': submission.score,
                        'subreddit': submission.subreddit.display_name,
                        'permalink': f"https://www.reddit.com{submission.permalink}",
                        'created_utc': submission.created_utc, # Store timestamp
                        'fullname': submission.fullname, 'url': sub_url,
                        'is_gallery': is_gallery,
                        'num_comments': submission.num_comments,
                        'upvote_ratio': getattr(submission, 'upvote_ratio', None),
                        'media': media_items
                    }
                    new_submissions_data.append(submission_data)
                    processed_submission_ids.add(submission.id)
            except prawcore.exceptions.Forbidden:
                logger.warning(
                    "Access forbidden fetching submissions for u/%s.", username
                )
            logger.info(
                "Fetched %d new submissions for u/%s (scanned %d items).",
                len(new_submissions_data), username, sub_count
            )

            # --- Fetch New Comments ---
            new_comments_data = []
            comment_count = 0
            processed_comment_ids = {c['id'] for c in existing_comments}
            logger.debug("Fetching new Reddit comments...")
            try:
                params = {}
                if not force_refresh and latest_comment_fullname:
                    params['before'] = latest_comment_fullname
                for comment in redditor.comments.new(limit=fetch_limit,
                                                   params=params):
                    comment_count += 1
                    if comment.id in processed_comment_ids:
                        continue
                    new_comments_data.append({
                        'id': comment.id,
                        'text': comment.body[:1500], # Limit text length
                        'score': comment.score,
                        'subreddit': comment.subreddit.display_name,
                        'permalink': f"https://www.reddit.com{comment.permalink}",
                        'created_utc': comment.created_utc, # Store timestamp
                        'fullname': comment.fullname,
                        'is_submitter': comment.is_submitter,
                        'link_id': comment.link_id, # ID of the submission
                        'parent_id': comment.parent_id, # ID of parent comment/sub
                    })
                    processed_comment_ids.add(comment.id)
            except prawcore.exceptions.Forbidden:
                logger.warning(
                    "Access forbidden fetching comments for u/%s.", username
                )
            logger.info(
                "Fetched %d new comments for u/%s (scanned %d items).",
                len(new_comments_data), username, comment_count
            )

            # --- Combine, Sort, Trim, and Calculate Stats ---
            combined_submissions = new_submissions_data + existing_submissions
            combined_comments = new_comments_data + existing_comments
            combined_submissions.sort(key=lambda x: get_sort_key(x, 'created_utc'),
                                      reverse=True)
            combined_comments.sort(key=lambda x: get_sort_key(x, 'created_utc'),
                                   reverse=True)
            final_submissions = combined_submissions[:MAX_CACHE_ITEMS]
            final_comments = combined_comments[:MAX_CACHE_ITEMS]

            final_media_analysis = list(dict.fromkeys(
                newly_added_media_analysis + existing_media_analysis
            ))[:MAX_CACHE_ITEMS]
            final_media_paths = sorted(list(
                newly_added_media_paths.union(existing_media_paths)
            ))[:MAX_CACHE_ITEMS * 2]

            # Calculate aggregate stats
            total_submissions = len(final_submissions)
            total_comments = len(final_comments)
            submissions_with_media = len([s for s in final_submissions
                                         if s.get('media')])
            stats = {
                'user_profile': redditor_info,
                'total_submissions': total_submissions,
                'total_comments': total_comments,
                'submissions_with_media': submissions_with_media,
                'total_media_items_processed': len(final_media_paths),
                'avg_submission_score': (
                    sum(s.get('score', 0) for s in final_submissions) /
                    max(total_submissions, 1)
                ),
                'avg_comment_score': (
                    sum(c.get('score', 0) for c in final_comments) /
                    max(total_comments, 1)
                )
            }

            # --- Save Cache ---
            final_data = {
                'timestamp': now_utc.isoformat(),
                'submissions': final_submissions,
                'comments': final_comments,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths,
                'stats': stats
            }
            self._save_cache('reddit', username, final_data)
            logger.info(
                "Successfully updated Reddit cache for u/%s. "
                "Cached subs: %d, comments: %d",
                username, total_submissions, total_comments
            )
            return final_data

        # --- Error Handling for Fetch ---
        except RateLimitExceededError:
            return None # Already handled
        except prawcore.exceptions.RequestException as e:
            # Check for specific 429 rate limit error
            if (hasattr(e, 'response') and e.response is not None and
                    e.response.status_code == 429):
                self._handle_rate_limit('Reddit', exception=e) # Raises exception
            else:
                logger.error("Reddit request failed for u/%s: %s", username, e)
            return None
        except (UserNotFoundError, AccessForbiddenError) as user_err:
            logger.error("Reddit fetch failed for u/%s: %s", username, user_err)
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching Reddit data for u/%s: %s",
                username, e, exc_info=True
            )
            return None


    def fetch_bluesky(self, username: str, force_refresh: bool = False
                     ) -> Optional[Dict[str, Any]]:
        """
        Fetches Bluesky user profile and posts.

        Args:
            username: Bluesky handle (e.g., 'handle.bsky.social').
            force_refresh: If True, ignore cache and fetch all data fresh.

        Returns:
            A dictionary containing profile info, posts, stats, and media analysis,
            or None if fetching fails.
        """
        cached_data = self._load_cache('bluesky', username)
        now_utc = datetime.now(timezone.utc)

        # Use cache if valid, recent, and not forced refresh
        if not force_refresh and cached_data and \
           (now_utc - datetime.fromisoformat(cached_data['timestamp'])) < \
           timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info("Using recent cache for Bluesky user %s", username)
            return cached_data

        logger.info(
            "Fetching Bluesky data for %s (Force Refresh: %s)",
            username, force_refresh
        )
        latest_post_datetime: Optional[datetime] = None
        existing_posts = []
        existing_media_analysis = []
        existing_media_paths = []
        profile_info = None

        # Baseline from cache for incremental fetch
        if not force_refresh and cached_data:
            logger.info("Attempting incremental fetch for Bluesky %s", username)
            existing_posts = cached_data.get('posts', [])
            profile_info = cached_data.get('profile_info')
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])
            existing_posts.sort(key=lambda x: get_sort_key(x, 'created_at'),
                                reverse=True)
            if existing_posts:
                # Get datetime of the most recent known post
                latest_post_datetime = get_sort_key(existing_posts[0], 'created_at')
                if latest_post_datetime > MIN_DATETIME_UTC:
                    logger.debug(
                        "Latest known post datetime: %s", latest_post_datetime.isoformat()
                    )
                else:
                    latest_post_datetime = None # Ignore invalid date
            else:
                logger.debug("No existing posts found for incremental baseline.")

        try:
            self.bluesky # Ensure client ready

            # Fetch profile info if missing or forced refresh
            if not profile_info or force_refresh:
                try:
                    profile = self.bluesky.get_profile(actor=username)
                    profile_info = {
                        'did': profile.did, 'handle': profile.handle,
                        'display_name': profile.display_name,
                        'description': profile.description, 'avatar': profile.avatar,
                        'banner': profile.banner,
                        'followers_count': getattr(profile, 'followers_count', 'N/A'),
                        'follows_count': getattr(profile, 'follows_count', 'N/A'),
                        'posts_count': getattr(profile, 'posts_count', 'N/A')
                    }
                    logger.debug("Bluesky profile fetched/updated for %s", username)
                except atproto_exceptions.AtProtocolError as e:
                    err_str = str(e).lower()
                    handle_not_found = ('profile not found' in err_str or
                                        'could not resolve handle' in err_str)
                    is_blocked = ('blocked by actor' in err_str or
                                  'blocking actor' in err_str)

                    if handle_not_found:
                        raise UserNotFoundError(f"Bluesky user {username} not found.")
                    elif is_blocked:
                        raise AccessForbiddenError(
                            f"Blocked from accessing Bluesky profile for {username}."
                        )
                    else:
                        # Log unexpected profile fetch errors but maybe continue?
                        logger.error(
                            "Unexpected error fetching Bluesky profile for %s: %s",
                            username, e
                        )
                        # Raise AccessForbiddenError as a general access issue
                        raise AccessForbiddenError(f"Error fetching profile: {e}")

            actor_did = profile_info.get('did') if profile_info else None
            if not actor_did:
                raise RuntimeError(f"Could not determine DID for Bluesky user {username}")

            # --- Fetch Posts ---
            new_posts_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            cursor = None
            processed_post_uris = {p['uri'] for p in existing_posts}
            # Determine fetch limit based on refresh type
            limit_type = (INITIAL_FETCH_LIMIT if (force_refresh or not latest_post_datetime)
                          else INCREMENTAL_FETCH_LIMIT + 10) # +10 for overlap
            fetch_limit_per_page = min(limit_type, 100) # API max is 100
            total_fetched_this_run = 0
            max_fetches_total = (INITIAL_FETCH_LIMIT if (force_refresh or not latest_post_datetime)
                                else INCREMENTAL_FETCH_LIMIT)
            page_count = 0
            max_pages = 5 # Limit pagination requests

            logger.debug(
                "Fetching new Bluesky posts for DID %s (Handle: %s)...",
                actor_did, username
            )
            while total_fetched_this_run < max_fetches_total and page_count < max_pages:
                page_count += 1
                stop_fetching = False
                logger.debug(
                    "Requesting feed page %d for %s with cursor: %s",
                    page_count, username, cursor
                )
                try:
                    response = self.bluesky.get_author_feed(
                        actor=actor_did, cursor=cursor, limit=fetch_limit_per_page
                    )
                except atproto_exceptions.AtProtocolError as e:
                    err_str = str(e).lower()
                    if 'rate limit' in err_str:
                        self._handle_rate_limit('Bluesky', exception=e) # Raises
                    elif ('could not resolve handle' in err_str or
                          'profile not found' in err_str):
                        # User might have been deleted between profile and feed fetch
                        raise UserNotFoundError(
                            f"Bluesky user {username} not found during feed fetch."
                        )
                    elif ('blocked by actor' in err_str or
                          'blocking actor' in err_str):
                        raise AccessForbiddenError(
                            f"Access to Bluesky feed blocked for {username}."
                        )
                    else:
                        logger.error(
                            "Bluesky API error fetching feed for %s: %s", username, e
                        )
                        break # Stop fetching on unexpected API errors

                if not response or not response.feed:
                    logger.debug("No more posts found for %s.", username)
                    break
                logger.debug(
                    "Processing feed page %d for %s with %d items.",
                    page_count, username, len(response.feed)
                )

                # Process posts in the current page
                for feed_item in response.feed:
                    post = feed_item.post
                    if not post: continue # Skip if post data is missing
                    post_uri = post.uri
                    if post_uri in processed_post_uris:
                        continue # Already processed this post

                    record = getattr(post, 'record', None)
                    if not record:
                        continue # Skip if record data is missing

                    # Check creation time for incremental fetch cutoff
                    created_at_dt = get_sort_key(record.to_dict(), 'created_at')
                    if not force_refresh and latest_post_datetime and \
                       created_at_dt <= latest_post_datetime:
                        logger.debug(
                            "Reached or passed latest known post datetime (%s)."
                            " Stopping feed processing for %s.",
                            latest_post_datetime.isoformat(), username
                        )
                        stop_fetching = True
                        break

                    # --- Process Media in Post ---
                    media_items_for_post = []
                    embed = getattr(record, 'embed', None)
                    image_embeds_to_process = []
                    # Extract embed type string cleanly
                    embed_type_str = ('unknown' if not embed else
                                      getattr(embed, '$type', 'unknown').split('.')[-1])

                    if embed:
                        potential_img_sources = []
                        # Check common embed structures for images
                        if hasattr(embed, 'images'):
                            potential_img_sources.append(embed.images)
                        # Case: Embed is media with images (e.g., external link card)
                        if hasattr(embed, 'media') and hasattr(embed.media, 'images'):
                           potential_img_sources.append(embed.media.images)
                        # Case: Embed is a record containing other embeds (quote post)
                        if hasattr(embed, 'record') and hasattr(embed.record, 'embeds'):
                           for nested_embed in getattr(embed.record, 'embeds', []):
                               if hasattr(nested_embed, 'images'):
                                   potential_img_sources.append(nested_embed.images)

                        # Flatten the list of potential image sources
                        for image_list in potential_img_sources:
                             if isinstance(image_list, list):
                                image_embeds_to_process.extend(image_list)

                    # Deduplicate and process found image embeds
                    processed_image_cids = set()
                    for image_info in image_embeds_to_process:
                        img_blob = getattr(image_info, 'image', None)
                        if img_blob:
                            # Extract CID (content identifier) using common attributes
                            cid_ref = getattr(img_blob, 'ref', None)
                            img_cid = (getattr(cid_ref, '$link', None) if cid_ref
                                       else getattr(img_blob, 'cid', None))

                            if img_cid and img_cid not in processed_image_cids:
                                processed_image_cids.add(img_cid)
                                author_did = post.author.did
                                # Construct the CDN URL (requires quoting DID/CID)
                                cdn_url = (f"https://cdn.bsky.app/img/feed_fullsize/plain/"
                                           f"{quote_plus(author_did)}/"
                                           f"{quote_plus(img_cid)}@jpeg")

                                media_path = self._download_media(url=cdn_url,
                                                                  platform='bluesky',
                                                                  username=username)
                                if media_path:
                                    analysis = None
                                    if media_path.suffix.lower() in \
                                       self.supported_image_extensions:
                                        analysis = self._analyze_image(
                                            media_path,
                                            f"Bluesky user {username}'s "
                                            f"post ({post.uri})"
                                        )
                                    media_items_for_post.append({
                                        'type': 'image', 'analysis': analysis,
                                        'url': cdn_url, # Store the CDN URL
                                        'alt_text': getattr(image_info, 'alt', ''),
                                        'local_path': str(media_path)
                                    })
                                    if analysis:
                                        newly_added_media_analysis.append(analysis)
                                    newly_added_media_paths.add(str(media_path))
                                else:
                                    logger.warning(
                                        "Failed Bluesky media download: %s", cdn_url
                                    )
                            elif not img_cid:
                                logger.warning(
                                    "Could not find image CID/link for post %s", post.uri
                                )
                        else:
                            logger.warning(
                                "Image embed structure missing 'image' blob for "
                                "post %s", post.uri
                            )

                    # --- Structure Post Data ---
                    post_text = getattr(record, 'text', '')
                    post_data = {
                        'uri': post.uri, 'cid': post.cid,
                        'author_did': post.author.did,
                        'text': post_text[:2000], # Limit text length
                        'created_at': created_at_dt.isoformat(), # Store ISO string
                        'likes': getattr(post, 'like_count', 0),
                        'reposts': getattr(post, 'repost_count', 0),
                        'reply_count': getattr(post, 'reply_count', 0),
                        'embed': {'type': embed_type_str} if embed else None,
                        'media': media_items_for_post
                    }
                    new_posts_data.append(post_data)
                    processed_post_uris.add(post_uri)
                    total_fetched_this_run += 1
                    # Check if fetch limit for this run is reached
                    if total_fetched_this_run >= max_fetches_total:
                        logger.info(
                            "Reached fetch limit (%d) for %s.",
                             max_fetches_total, username
                         )
                        stop_fetching = True
                        break

                if stop_fetching: break # Exit outer loop if needed

                # Prepare for next page
                cursor = response.cursor
                if not cursor:
                    logger.debug("Reached end of feed for %s.", username)
                    break
                if page_count >= max_pages:
                    logger.warning(
                        "Reached max page limit (%d) for %s.", max_pages, username
                    )
                    break # Prevent excessive requests

            logger.info(
                "Fetched %d new posts for Bluesky user %s.",
                len(new_posts_data), username
            )

            # --- Combine, Sort, Trim, Calculate Stats ---
            existing_uris = {p['uri'] for p in existing_posts}
            unique_new_posts = [p for p in new_posts_data
                                if p['uri'] not in existing_uris]

            combined_posts = unique_new_posts + existing_posts
            combined_posts.sort(key=lambda x: get_sort_key(x, 'created_at'),
                                reverse=True)
            final_posts = combined_posts[:MAX_CACHE_ITEMS] # Trim to cache size

            final_media_analysis = list(dict.fromkeys(
                newly_added_media_analysis + existing_media_analysis
            ))[:MAX_CACHE_ITEMS]
            final_media_paths = sorted(list(
                newly_added_media_paths.union(existing_media_paths)
            ))[:MAX_CACHE_ITEMS * 2]

            total_posts = len(final_posts)
            posts_with_media = len([p for p in final_posts if p.get('media')])
            stats = {
                'total_posts_cached': total_posts,
                'posts_with_media': posts_with_media,
                'total_media_items_processed': len(final_media_paths),
                'avg_likes': (sum(p.get('likes', 0) for p in final_posts) /
                              max(total_posts, 1)),
                'avg_reposts': (sum(p.get('reposts', 0) for p in final_posts) /
                               max(total_posts, 1)),
                'avg_replies': (sum(p.get('reply_count', 0) for p in final_posts) /
                               max(total_posts, 1))
            }

            # --- Save Cache ---
            final_data = {
                'timestamp': now_utc.isoformat(),
                'profile_info': profile_info,
                'posts': final_posts,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths,
                'stats': stats
            }
            self._save_cache('bluesky', username, final_data)
            logger.info(
                "Successfully updated Bluesky cache for %s. "
                "Total posts cached: %d", username, total_posts
            )
            return final_data

        # --- Error Handling for Fetch ---
        except RateLimitExceededError:
            return None # Already handled
        except (UserNotFoundError, AccessForbiddenError, RuntimeError) as user_err:
            # Catch profile fetch errors too
            logger.error("Bluesky fetch failed for %s: %s", username, user_err)
            return None
        except atproto_exceptions.AtProtocolError as e:
            # Catch other unexpected ATProto errors during feed fetch
            logger.error("Bluesky ATProtocol error for %s: %s", username, e)
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching Bluesky data for %s: %s",
                username, e, exc_info=True
            )
            return None


    def fetch_mastodon(self, username: str, force_refresh: bool = False
                      ) -> Optional[Dict[str, Any]]:
        """
        Fetches Mastodon user info and statuses (posts/boosts).

        Args:
            username: Mastodon username in 'user@instance.domain' format.
            force_refresh: If True, ignore cache and fetch all data fresh.

        Returns:
            A dictionary containing user info, posts, stats, and media analysis,
            or None if fetching fails.
        """
        # Validate username format early
        if '@' not in username or '.' not in username.split('@', 1)[1]:
            err_msg = (f"Invalid Mastodon username format: '{username}'. "
                       "Must be 'user@instance.domain'.")
            logger.error(err_msg)
            # Raise ValueError to be caught in the calling function
            raise ValueError(err_msg)

        cache_key_username = username # Use validated username for cache key
        cached_data = self._load_cache('mastodon', cache_key_username)
        now_utc = datetime.now(timezone.utc)

        # Use cache if valid, recent, and not forced refresh
        if not force_refresh and cached_data and \
           (now_utc - datetime.fromisoformat(cached_data['timestamp'])) < \
           timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info("Using recent cache for Mastodon user %s", cache_key_username)
            return cached_data

        logger.info(
            "Fetching Mastodon data for %s (Force Refresh: %s)",
            cache_key_username, force_refresh
        )
        since_id = None
        existing_posts = []
        existing_media_analysis = []
        existing_media_paths = []
        user_info = None

        # Baseline from cache for incremental fetch
        if not force_refresh and cached_data:
            logger.info(
                "Attempting incremental fetch for Mastodon %s", cache_key_username
            )
            existing_posts = cached_data.get('posts', [])
            user_info = cached_data.get('user_info')
            existing_media_analysis = cached_data.get('media_analysis', [])
            existing_media_paths = cached_data.get('media_paths', [])
            existing_posts.sort(key=lambda x: get_sort_key(x, 'created_at'),
                                reverse=True)
            if existing_posts:
                since_id = existing_posts[0].get('id')
                if since_id: logger.debug("Using since_id: %s", since_id)

        try:
            self.mastodon # Ensure client ready

            # Fetch user info if missing or forced refresh
            if not user_info or force_refresh:
                try:
                    logger.debug(
                        "Looking up Mastodon account: %s using client for %s",
                        username, self.mastodon.api_base_url
                    )
                    # Use the original username (which includes @instance) for lookup
                    account = self.mastodon.account_lookup(acct=username)
                    created_at = account.get('created_at')
                    user_info = {
                        'id': account['id'], 'username': account['username'],
                        'acct': account['acct'], # Full user@instance
                        'display_name': account['display_name'],
                        'note_html': account.get('note', ''), # Bio HTML
                        'url': account['url'], # Profile URL
                        'avatar': account['avatar'], 'header': account['header'],
                        'followers_count': account['followers_count'],
                        'following_count': account['following_count'],
                        'statuses_count': account['statuses_count'],
                        'created_at': (created_at.isoformat()
                                       if created_at else None),
                        'locked': account.get('locked', False), # Private account
                        'bot': account.get('bot', False),
                        'fields': account.get('fields', []) # Profile metadata
                    }
                    logger.info(
                        "Fetched Mastodon user info for %s", cache_key_username
                    )
                except MastodonNotFoundError:
                    raise UserNotFoundError(
                        f"Mastodon user {username} not found via "
                        f"{self.mastodon.api_base_url}."
                    )
                except MastodonUnauthorizedError:
                    # Could be locked profile we don't follow
                    raise AccessForbiddenError(
                        f"Unauthorized access to Mastodon user {username}'s info. "
                        f"(Profile might be locked or server requires auth)."
                    )
                except (MastodonError, MastodonVersionError) as me:
                    logger.error(
                        "Mastodon API error looking up %s: %s", username, me
                    )
                    raise AccessForbiddenError(
                        f"API error accessing {username}: {me}"
                    )

            user_id = user_info['id']

            # --- Fetch Statuses (Posts/Boosts) ---
            new_posts_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            # Fetch slightly more for incremental overlap, up to API limit
            fetch_limit = (INITIAL_FETCH_LIMIT if (force_refresh or not since_id)
                           else INCREMENTAL_FETCH_LIMIT + 10)
            api_limit = min(fetch_limit, MASTODON_FETCH_LIMIT)
            processed_status_ids = {p['id'] for p in existing_posts}

            logger.debug(
                "Fetching new statuses for Mastodon user ID %s (%s) "
                "(since_id: %s, limit: %d)",
                user_id, cache_key_username, since_id, api_limit
            )
            try:
                # Fetch statuses, including replies and boosts (reblogs)
                new_statuses = self.mastodon.account_statuses(
                    id=user_id,
                    limit=api_limit,
                    since_id=since_id if not force_refresh else None,
                    exclude_replies=False, # Include replies
                    exclude_reblogs=False  # Include boosts
                )
            except MastodonRatelimitError as e:
                self._handle_rate_limit('Mastodon', exception=e) # Raises
            except MastodonNotFoundError:
                # Should not happen if account lookup succeeded, but handle defensively
                raise UserNotFoundError(
                    f"Mastodon user ID {user_id} not found during status fetch."
                )
            except (MastodonUnauthorizedError, MastodonVersionError,
                    MastodonError) as me:
                logger.error("Error fetching statuses for %s: %s", username, me)
                raise AccessForbiddenError(
                    f"Access error fetching statuses for {username}."
                )

            logger.info(
                "Fetched %d new raw statuses for Mastodon user %s.",
                len(new_statuses), cache_key_username
            )
            count_added = 0
            for status in new_statuses:
                status_id = status['id']
                if status_id in processed_status_ids:
                    continue # Skip duplicates

                # --- Clean HTML Content ---
                cleaned_text = ''
                try:
                    # Use BeautifulSoup to parse HTML content and get plain text
                    soup = BeautifulSoup(status.get('content', ''), 'html.parser')
                    # Replace <br> with newlines, add newline after <p> for structure
                    for br in soup.find_all("br"):
                        br.replace_with("\n")
                    for p_tag in soup.find_all("p"):
                        p_tag.append("\n")
                    cleaned_text = soup.get_text(separator=' ', strip=True)
                except Exception as parse_err:
                    logger.warning(
                        "HTML parsing failed for Mastodon status %s: %s",
                        status_id, parse_err
                    )
                    # Fallback to raw content if parsing fails
                    cleaned_text = status.get('content', '')

                # --- Process Media Attachments ---
                media_items_for_post = []
                for attachment in status.get('media_attachments', []):
                    media_url = attachment.get('url') # Usually the full media
                    preview_url = attachment.get('preview_url') # Often for video/gif
                    media_type = attachment.get('type', 'unknown') # e.g., image, video, gifv, audio
                    # Prefer full URL, but use preview if needed (e.g., video thumb)
                    url_to_download = (media_url if media_url and
                                       media_type not in ['video', 'gifv']
                                       else preview_url)
                    if url_to_download:
                        media_path = self._download_media(
                            url=url_to_download, platform='mastodon',
                            username=cache_key_username
                        )
                        if media_path:
                            analysis = None
                            # Analyze only supported image types
                            if media_type == 'image' and \
                               media_path.suffix.lower() in \
                               self.supported_image_extensions:
                                image_context = (
                                    f"Mastodon user {cache_key_username}'s post "
                                    f"({status.get('url', status_id)})"
                                )
                                analysis = self._analyze_image(media_path, image_context)

                            media_items_for_post.append({
                                'type': media_type, 'analysis': analysis,
                                'url': media_url, # Store original full URL
                                'preview_url': preview_url,
                                'description': attachment.get('description'), # Alt text
                                'local_path': str(media_path)
                            })
                            if analysis: newly_added_media_analysis.append(analysis)
                            newly_added_media_paths.add(str(media_path))
                        else:
                            logger.warning(
                                "Failed Mastodon download: %s", url_to_download
                            )

                # --- Identify Reblogs (Boosts) ---
                is_reblog = status.get('reblog') is not None
                reblog_info = status.get('reblog') if is_reblog else None
                reblog_author = None
                reblog_url = None
                if reblog_info and 'account' in reblog_info:
                    reblog_author = reblog_info['account'].get('acct') # Original author
                    reblog_url = reblog_info.get('url') # Link to original status

                # --- Structure Post Data ---
                status_created_at = status.get('created_at')
                post_data = {
                    'id': status_id,
                    'created_at': (status_created_at.isoformat()
                                   if status_created_at else None),
                    'url': status.get('url'), # Link to this status/reblog
                    'text_html': status.get('content', ''),
                    'text_cleaned': cleaned_text[:2000], # Limit text length
                    'spoiler_text': status.get('spoiler_text', ''), # Content Warning
                    'reblogs_count': status.get('reblogs_count', 0),
                    'favourites_count': status.get('favourites_count', 0),
                    'replies_count': status.get('replies_count', 0),
                    'in_reply_to_id': status.get('in_reply_to_id'),
                    'in_reply_to_account_id': status.get('in_reply_to_account_id'),
                    'is_reblog': is_reblog,
                    'reblog_original_author': reblog_author,
                    'reblog_original_url': reblog_url,
                    'language': status.get('language'),
                    'visibility': status.get('visibility'), # public, unlisted, etc.
                    'tags': [{'name': tag.get('name'), 'url': tag.get('url')}
                             for tag in status.get('tags', [])],
                    'media': media_items_for_post
                }
                new_posts_data.append(post_data)
                processed_status_ids.add(status_id)
                count_added += 1

            logger.info(
                "Processed %d new unique statuses for Mastodon user %s.",
                count_added, cache_key_username
            )

            # --- Combine, Sort, Trim, Calculate Stats ---
            combined_posts = new_posts_data + existing_posts
            combined_posts.sort(key=lambda x: get_sort_key(x, 'created_at'),
                                reverse=True)
            final_posts = combined_posts[:MAX_CACHE_ITEMS] # Trim cache

            final_media_analysis = list(dict.fromkeys(
                newly_added_media_analysis + existing_media_analysis
            ))[:MAX_CACHE_ITEMS]
            final_media_paths = sorted(list(
                newly_added_media_paths.union(existing_media_paths)
            ))[:MAX_CACHE_ITEMS * 2]

            # Calculate stats based on cached posts
            total_posts_cached = len(final_posts)
            original_posts = [p for p in final_posts if not p.get('is_reblog')]
            total_original_posts = len(original_posts)
            total_reblogs = total_posts_cached - total_original_posts
            posts_with_media = len([p for p in final_posts if p.get('media')])
            # Calculate averages based on original posts only for relevance
            avg_favs = (sum(p['favourites_count'] for p in original_posts) /
                        max(total_original_posts, 1))
            avg_boosts_received = (sum(p['reblogs_count'] for p in original_posts) /
                                   max(total_original_posts, 1))
            avg_replies = (sum(p['replies_count'] for p in original_posts) /
                           max(total_original_posts, 1))
            stats = {
                'total_posts_cached': total_posts_cached,
                'total_original_posts': total_original_posts,
                'total_reblogs_in_cache': total_reblogs,
                'posts_with_media': posts_with_media,
                'total_media_items_processed': len(final_media_paths),
                'avg_favourites_on_originals': avg_favs,
                'avg_boosts_on_originals': avg_boosts_received,
                'avg_replies_on_originals': avg_replies
            }

            # --- Save Cache ---
            final_data = {
                'timestamp': now_utc.isoformat(),
                'user_info': user_info,
                'posts': final_posts,
                'media_analysis': final_media_analysis,
                'media_paths': final_media_paths,
                'stats': stats
            }
            self._save_cache('mastodon', cache_key_username, final_data)
            logger.info(
                "Successfully updated Mastodon cache for %s. "
                "Total posts cached: %d", cache_key_username, total_posts_cached
            )
            return final_data

        # --- Error Handling for Fetch ---
        except ValueError as ve: # Catch the format validation error
            logger.error("Mastodon fetch failed for %s: %s", username, ve)
            # Re-raise or return None depending on desired flow
            # Here, we return None to allow skipping this user in loops
            return None
        except RateLimitExceededError:
            return None # Already handled
        except (UserNotFoundError, AccessForbiddenError) as user_err:
            logger.error("Mastodon fetch failed for %s: %s", username, user_err)
            return None
        except MastodonError as e: # Catch other Mastodon API errors
            logger.error("Mastodon API error for %s: %s", username, e)
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching Mastodon data for %s: %s",
                username, e, exc_info=True
            )
            return None


    def fetch_hackernews(self, username: str, force_refresh: bool = False
                        ) -> Optional[Dict[str, Any]]:
        """
        Fetches Hacker News submissions and comments using the Algolia API.

        Args:
            username: Hacker News username (case-sensitive).
            force_refresh: If True, ignore cache and fetch all data fresh.

        Returns:
            A dictionary containing submissions/comments and stats, or None if fails.
        """
        cached_data = self._load_cache('hackernews', username)
        now_utc = datetime.now(timezone.utc)

        # Use cache if valid, recent, and not forced refresh
        if not force_refresh and cached_data and \
           (now_utc - datetime.fromisoformat(cached_data['timestamp'])) < \
           timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info("Using recent cache for HackerNews user %s", username)
            return cached_data

        logger.info(
            "Fetching HackerNews data for %s (Force Refresh: %s)",
            username, force_refresh
        )
        latest_timestamp_i = 0 # Unix timestamp integer
        existing_submissions = [] # Stores both stories and comments

        # Baseline from cache for incremental fetch
        if not force_refresh and cached_data:
            logger.info("Attempting incremental fetch for HackerNews %s", username)
            existing_submissions = cached_data.get('submissions', [])
            # Sort by integer timestamp for finding the latest
            existing_submissions.sort(key=lambda x: get_sort_key(x, 'created_at_i'),
                                      reverse=True)
            if existing_submissions:
                # Find the max valid integer timestamp from existing items
                valid_timestamps = [
                    s.get('created_at_i') for s in existing_submissions
                    if isinstance(s.get('created_at_i'), (int, float)) and
                       s['created_at_i'] > 0 # Ensure positive timestamp
                ]
                if valid_timestamps:
                    latest_timestamp_i = int(max(valid_timestamps))
                    logger.debug("Using latest timestamp_i: %d", latest_timestamp_i)
                else:
                    logger.warning("No valid 'created_at_i' found in HN cache.")

        try:
            # HN Algolia API endpoint for user search
            base_url = "https://hn.algolia.com/api/v1/search"
            # Determine fetch limit
            hits_per_page = (INITIAL_FETCH_LIMIT if (force_refresh or not latest_timestamp_i)
                           else INCREMENTAL_FETCH_LIMIT + 10) # Fetch more for overlap

            # Prepare query parameters
            params = {
                "tags": f"author_{quote_plus(username)}", # Filter by author
                "hitsPerPage": hits_per_page,
                "typoTolerance": False # Exact username match
            }
            # Add time filter for incremental fetches
            if not force_refresh and latest_timestamp_i > 0:
                # Fetch items created *after* the latest known timestamp
                params["numericFilters"] = f"created_at_i>{latest_timestamp_i}"

            new_submissions_data = []
            # Keep track of IDs already in cache to avoid duplicates if API overlaps
            processed_ids = {s['objectID'] for s in existing_submissions
                             if 'objectID' in s}

            logger.debug(
                "Querying HN Algolia: %s with params: %s", base_url, params
            )
            # --- Make API Call ---
            # Note: HN Algolia doesn't typically have strict rate limits,
            # but httpx handles standard HTTP errors.
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                 response = client.get(base_url, params=params)
                 response.raise_for_status() # Check for HTTP errors
                 data = response.json()

            # Handle cases where 'hits' might be missing or empty
            if 'hits' not in data:
                # Check if it's a valid empty response
                if response.status_code == 200 and data.get('nbHits', 0) == 0:
                    logger.info("No hits found for HN user %s.", username)
                    # Create empty cache structure if user exists but has no posts
                    empty_data = {
                        'timestamp': now_utc.isoformat(),
                        'submissions': [],
                        'stats': {'total_items_cached': 0, 'total_stories': 0,
                                  'total_comments': 0, 'average_story_points': 0,
                                  'average_story_num_comments': 0}
                    }
                    self._save_cache('hackernews', username, empty_data)
                    return empty_data
                else:
                    logger.warning(
                        "No 'hits' key in HN response for %s: %s", username, data
                    )
                    data['hits'] = [] # Ensure hits is a list for iteration

            logger.info(
                "Fetched %d potential new items for HN user %s.",
                len(data.get('hits', [])), username
            )

            # --- Process Hits ---
            for hit in data.get('hits', []):
                object_id = hit.get('objectID')
                # Skip if ID is missing or already processed
                if not object_id or object_id in processed_ids:
                    continue
                created_at_ts = hit.get('created_at_i')
                # Ensure timestamp is valid
                if not isinstance(created_at_ts, (int, float)) or created_at_ts <= 0:
                    logger.warning(
                        "Skipping HN hit %s, invalid timestamp: %s",
                        object_id, created_at_ts
                    )
                    continue

                # Determine item type from tags
                tags = hit.get('_tags', [])
                item_type = 'unknown'
                if 'story' in tags: item_type = 'story'
                elif 'comment' in tags: item_type = 'comment'
                elif 'poll' in tags: item_type = 'poll'
                elif 'pollopt' in tags: item_type = 'pollopt'
                elif 'job' in tags: item_type = 'job'

                # Extract and clean text content
                raw_text = hit.get('comment_text') or hit.get('story_text') or ''
                cleaned_text = (BeautifulSoup(raw_text, 'html.parser')
                                .get_text(separator=' ', strip=True)
                                if '<' in raw_text else raw_text)

                # Convert timestamp to datetime object and ISO string
                created_at_dt = datetime.fromtimestamp(created_at_ts, tz=timezone.utc)

                submission_item = {
                    'objectID': object_id, # Unique ID from Algolia
                    'type': item_type,
                    'title': hit.get('title'), # Present for stories, polls, jobs
                    'url': hit.get('url') or hit.get('story_url'), # External link
                    'points': hit.get('points'), # Score
                    'num_comments': hit.get('num_comments'), # For stories
                    'story_id': hit.get('story_id'), # Link comment to story
                    'parent_id': hit.get('parent_id'), # Link comment reply chain
                    'created_at_i': created_at_ts, # Original integer timestamp
                    'created_at': created_at_dt.isoformat(), # Store ISO string
                    'text': cleaned_text[:2000] # Limit text length
                }
                new_submissions_data.append(submission_item)
                processed_ids.add(object_id)

            # --- Combine, Sort, Trim, Calculate Stats ---
            combined_submissions = new_submissions_data + existing_submissions
            combined_submissions.sort(key=lambda x: get_sort_key(x, 'created_at_i'),
                                      reverse=True)
            final_submissions = combined_submissions[:MAX_CACHE_ITEMS] # Trim cache

            # Calculate stats from cached items
            story_submissions = [s for s in final_submissions
                                 if s.get('type') == 'story']
            total_items = len(final_submissions)
            total_stories = len(story_submissions)
            total_comments = len([s for s in final_submissions
                                 if s.get('type') == 'comment'])
            avg_story_pts = (sum(s.get('points', 0) or 0 for s in story_submissions) /
                            max(total_stories, 1))
            avg_story_comments = (sum(s.get('num_comments', 0) or 0
                                    for s in story_submissions) /
                                  max(total_stories, 1))
            stats = {
                'total_items_cached': total_items,
                'total_stories': total_stories,
                'total_comments': total_comments,
                'average_story_points': avg_story_pts,
                'average_story_num_comments': avg_story_comments
            }

            # --- Save Cache ---
            final_data = {
                'timestamp': now_utc.isoformat(),
                'submissions': final_submissions,
                'stats': stats
            }
            self._save_cache('hackernews', username, final_data)
            logger.info(
                "Successfully updated HackerNews cache for %s. "
                "Total items cached: %d", username, total_items
            )
            return final_data

        # --- Error Handling for Fetch ---
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 429:
                # Although rare, handle potential rate limits
                self._handle_rate_limit('HackerNews (Algolia)', e) # Raises
            elif status_code == 400:
                # Bad request often means invalid username format or filter
                logger.error(
                    "HN Algolia Bad Request (400) for %s: %s",
                    username, e.response.text
                )
                # Raise UserNotFoundError as it implies the user might not exist
                # or the query is malformed due to the username.
                raise UserNotFoundError(
                    f"HN username '{username}' caused bad request (400)."
                )
            else:
                logger.error(
                    "HN Algolia HTTP error %d for %s: %s",
                    status_code, username, e.response.text
                )
            return None
        except httpx.RequestError as e:
            # Network errors during request
            logger.error("HN Algolia network error for %s: %s", username, e)
            return None
        except UserNotFoundError as e:
            # Re-raise specific user not found errors
            logger.error("HN fetch failed for %s: %s", username, e)
            raise e
        except Exception as e:
            logger.error(
                "Unexpected error fetching HN data for %s: %s",
                username, e, exc_info=True
            )
            return None


    # --- Analysis Core ---
    def analyze(self, platforms: Dict[str, Union[str, List[str]]], query: str
               ) -> str:
        """
        Collects data (using fetch methods) and performs LLM analysis.

        Args:
            platforms: Dict mapping platform name to list of usernames.
            query: The user's analysis query.

        Returns:
            A formatted string containing the analysis report or an error message.
        """
        collected_text_summaries: List[str] = []
        all_media_analyzes: List[str] = []
        failed_fetches: List[tuple[str, str, str]] = []

        # Normalize platforms input to ensure usernames are lists
        platforms_normalized: Dict[str, List[str]] = {}
        for p, u in platforms.items():
            if isinstance(u, str):
                platforms_normalized[p] = [u] if u.strip() else []
            elif isinstance(u, list):
                platforms_normalized[p] = [usr for usr in u if isinstance(usr, str) and usr.strip()]
            # Remove platforms with empty user lists
            if not platforms_normalized.get(p):
                 platforms_normalized.pop(p, None)


        platform_count = sum(len(users) for users in platforms_normalized.values())
        if platform_count == 0:
            return "[yellow]No valid platforms or users specified for analysis.[/yellow]"

        collect_task: Optional[TaskID] = None
        try:
            # --- Data Collection Phase ---
            collect_task = self.progress.add_task(
                f"[cyan]Collecting data for {platform_count} target(s)...",
                total=platform_count
            )
            self.progress.start()

            for platform, usernames in platforms_normalized.items():
                fetcher = getattr(self, f'fetch_{platform}', None)
                if not fetcher:
                    logger.warning("No fetcher implemented for platform: %s", platform)
                    failed_fetches.extend([(platform, u, "Fetcher not implemented")
                                           for u in usernames])
                    if collect_task:
                        self.progress.advance(collect_task, advance=len(usernames))
                    continue

                for username in usernames:
                    # Create display name (e.g., @user, u/user)
                    user_prefix = {"twitter": "@", "reddit": "u/"}.get(platform, "")
                    display_name = f"{user_prefix}{username}"

                    if collect_task:
                        self.progress.update(
                            collect_task,
                            description=f"[cyan]Fetching {platform} for "
                                        f"{display_name}..."
                        )
                    try:
                        # Call the appropriate fetch method
                        data = fetcher(username=username, force_refresh=False)
                        if data:
                            # Format fetched data into text summary
                            summary = self._format_text_data(platform, username, data)
                            if summary:
                                collected_text_summaries.append(summary)
                            # Extract media analysis results
                            media = [ma for ma in data.get('media_analysis', []) if ma]
                            if media:
                                all_media_analyzes.extend(media)
                            logger.info(
                                "Successfully collected data for %s/%s",
                                platform, display_name
                            )
                        else:
                            # Handle cases where fetcher returns None (expected errors)
                            failed_fetches.append(
                                (platform, display_name, "Data fetch failed or N/A")
                            )
                            logger.warning(
                                "Data fetch returned None for %s/%s",
                                platform, display_name
                            )
                    except RateLimitExceededError:
                        failed_fetches.append(
                            (platform, display_name, "Rate Limited")
                        )
                    except (UserNotFoundError, AccessForbiddenError, ValueError) as afe:
                        # Handle expected user/access/input errors gracefully
                        err_type = type(afe).__name__
                        failed_fetches.append(
                            (platform, display_name, f"Access/Input Error ({err_type})")
                        )
                        self.console.print(
                            f"[yellow]Skipping {platform}/{display_name}: "
                            f"{afe}[/yellow]", style="yellow"
                        )
                    except RuntimeError as rte:
                        # Handle configuration/runtime errors during fetch
                        failed_fetches.append(
                            (platform, display_name, "Runtime/Config Error")
                        )
                        self.console.print(
                            f"[red]Fetch Runtime Error for {platform}/{display_name}: "
                            f"{rte}[/red]", style="red"
                        )
                    except Exception as e:
                        # Catch unexpected errors during fetch
                        logger.error(
                            "Unexpected fetch error for %s/%s: %s",
                            platform, display_name, e, exc_info=True
                        )
                        failed_fetches.append(
                            (platform, display_name, "Unexpected Fetch Error")
                        )
                        self.console.print(
                            f"[red]Unexpected Fetch Error for {platform}/{display_name}:"
                            f" {e}[/red]", style="red"
                        )
                    finally:
                        # Advance progress regardless of success/failure for this user
                        if collect_task:
                            self.progress.advance(collect_task)

            # --- Post-Collection Reporting ---
            if collect_task and collect_task in self.progress.task_ids:
                self.progress.update(
                    collect_task, completed=platform_count,
                    description="[green]Data collection finished."
                )
            if self.progress.live.is_started:
                self.progress.stop() # Stop collection progress display

            if failed_fetches:
                self.console.print("\n[bold yellow]Data Collection Issues:[/bold yellow]")
                # Use set for unique failures, then sort for consistent display
                unique_failures = sorted(list(set(failed_fetches)))
                for pf, user, reason in unique_failures:
                    self.console.print(f"- {pf}/{user}: {reason}")
                self.console.print(
                    "[yellow]Analysis will proceed with available data.[/yellow]\n"
                )

            # Check if *any* data was successfully collected
            if not collected_text_summaries and not all_media_analyzes:
                if failed_fetches and platform_count == len(failed_fetches):
                    return "[red]Data collection failed for all targets.[/red]"
                else:
                    return "[red]No data successfully collected or formatted.[/red]"

            # --- Prepare Data for LLM ---
            # Ensure media analysis list is unique and free of None/empty strings
            unique_media_analyzes = sorted(list(set(filter(None, all_media_analyzes))))
            analysis_components = []
            image_model = os.getenv('IMAGE_ANALYSIS_MODEL', 'google/gemini-pro-vision')
            text_model = os.getenv('ANALYSIS_MODEL', 'mistralai/mixtral-8x7b-instruct')

            # Add consolidated media analysis section if available
            if unique_media_analyzes:
                media_summary = (
                    f"## Consolidated Media Analysis (using {image_model}):\n\n"
                    "*Note: Objective descriptions based on visual content.*\n\n"
                )
                media_summary += "\n\n".join(
                    f"{i+1}. {analysis.strip()}"
                    for i, analysis in enumerate(unique_media_analyzes)
                )
                analysis_components.append(media_summary)
                logger.debug(
                    "Added %d unique media analyzes to LLM prompt.",
                    len(unique_media_analyzes)
                )

            # Add collected text/activity data summaries
            if collected_text_summaries:
                text_summary = "## Collected Textual & Activity Data Summary:\n\n"
                # Join summaries, ensuring no empty summaries are added
                text_summary += "\n\n---\n\n".join(
                    summary.strip() for summary in collected_text_summaries
                    if summary.strip()
                )
                analysis_components.append(text_summary)
                logger.debug(
                    "Added %d platform text summaries to LLM prompt.",
                    len(collected_text_summaries)
                )

            if not analysis_components:
                return "[red]No data could be formatted for LLM analysis.[/red]"

            # --- Define LLM Prompts ---
            system_prompt = """**Objective:** Generate a comprehensive behavioral and linguistic profile based on the provided social media data, employing structured analytic techniques focused on objectivity, evidence-based reasoning, and clear articulation.

**Input:** You will receive summaries of user activity (text posts, engagement metrics, descriptive analyzes of images shared) from platforms like Twitter, Reddit, Bluesky, Mastodon, and Hacker News for one or more specified users. The user will also provide a specific analysis query. You may also receive consolidated analyzes of images shared by the user(s).

**Primary Task:** Address the user's specific analysis query using ALL the data provided (text summaries AND image analyzes) and the analytical framework below.

**Analysis Domains (Use these to structure your thinking and response where relevant to the query):**
1.  **Behavioral Patterns:** Analyze interaction frequency, platform-specific activity (e.g., retweets vs. posts, submissions vs. comments, boosts vs. original posts), potential engagement triggers, and temporal communication rhythms apparent *in the provided data*. Note differences across platforms if multiple are present.
2.  **Semantic Content & Themes:** Identify recurring topics, keywords, and concepts. Analyze linguistic indicators such as expressed sentiment/tone (positive, negative, neutral, specific emotions if clear), potential ideological leanings *if explicitly stated or strongly implied by language/topics*, and cognitive framing (how subjects are discussed). Assess information source credibility *only if* the user shares external links/content within the provided data AND you can evaluate the source based on common knowledge. Note use of content warnings/spoilers (e.g., on Mastodon).
3.  **Interests & Network Context:** Deduce primary interests, hobbies, or professional domains suggested by post content and image analysis. Note any interaction patterns visible *within the provided posts* (e.g., frequent replies to specific user types, retweets/boosts of particular accounts, participation in specific communities like subreddits or Mastodon hashtags/local timelines if mentioned). Avoid inferring broad influence or definitive group membership without strong evidence.
4.  **Communication Style:** Assess linguistic complexity (simple/complex vocabulary, sentence structure), use of jargon/slang, rhetorical strategies (e.g., humor, sarcasm, argumentation), markers of emotional expression (e.g., emoji use, exclamation points, emotionally charged words), and narrative consistency across platforms. Note use of HTML/rich text formatting (e.g., in Mastodon) or markdown (Reddit).
5.  **Visual Data Integration:** Explicitly incorporate insights derived from the provided image analyzes. How do the visual elements (settings, objects, activities depicted) complement, contradict, or add context to the textual data? Note any patterns in the *types* of images shared (photos, screenshots, art) or use/lack of alt text.

**Analytical Constraints & Guidelines:**
*   **Evidence-Based:** Ground ALL conclusions *strictly and exclusively* on the provided source materials (text summaries AND image analyzes). Reference specific examples or patterns from the data (e.g., "Frequent posts about [topic] on Reddit," "Image analysis of setting suggests [environment]," "Consistent use of technical jargon on HackerNews", "Use of spoiler tags on Mastodon for [topic]").
*   **Objectivity & Neutrality:** Maintain analytical neutrality. Avoid speculation, moral judgments, personal opinions, or projecting external knowledge not present in the data. Focus on describing *what the data shows*.
*   **Synthesize, Don't Just List:** Integrate findings from different platforms and data types (text/image) into a coherent narrative that addresses the query. Highlight correlations or discrepancies.
*   **Address the Query Directly:** Structure your response primarily around answering the user's specific question(s). Use the analysis domains as tools to build your answer.
*   **Acknowledge Limitations:** If the data is sparse, lacks specific details needed for the query, only covers a short time period, or certain data types were excluded (e.g., replies/boosts), explicitly state these limitations (e.g., "Based on the limited posts available...", "Image analysis provides no clues regarding [aspect]", "Mastodon data primarily includes original posts..."). Do not invent information.
*   **Clarity & Structure:** Use clear language. Employ formatting (markdown headings, bullet points) to organize the response logically, often starting with a direct answer to the query followed by supporting evidence/analysis.

**Output:** A structured analytical report that directly addresses the user's query, rigorously supported by evidence from the provided text and image data, adhering to all constraints. Start with a summary answer, then elaborate with details structured using relevant analysis domains.
"""
            # Combine all collected data components with separators for the user prompt
            user_prompt = (
                f"**Analysis Query:** {query}\n\n"
                f"**Provided Data:**\n\n" +
                "\n\n===[DATA SEPARATOR]===\n\n".join(analysis_components)
            )

            # --- Perform LLM Analysis (in a thread) ---
            analysis_task: Optional[TaskID] = None
            try:
                self.openrouter # Ensure client ready

                analysis_task = self.progress.add_task(
                    f"[magenta]Analyzing with {text_model}...", total=None
                )
                self.progress.start() # Start analysis progress spinner

                # Reset state variables before starting thread
                self._analysis_response = None
                self._analysis_exception = None

                # Prepare data for the API call
                llm_payload = {
                    "model": text_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 3800, # Allow ample response length
                    "temperature": 0.4, # Lower temperature for more factual output
                    "stream": False # Wait for full response
                }

                # Run API call in a separate thread to keep UI responsive
                api_thread = threading.Thread(
                    target=self._call_openrouter,
                    kwargs={"json_data": llm_payload},
                    daemon=True # Allow program exit even if thread hangs (unlikely)
                )
                api_thread.start()
                # Wait for thread completion with a timeout based on httpx client setting
                api_thread.join(timeout=self.openrouter.timeout.read + 10.0)

                # Check thread status and results
                if api_thread.is_alive():
                    logger.error("LLM analysis API call timed out.")
                    raise TimeoutError("Analysis API call timed out.")
                if self._analysis_exception:
                    # Re-raise exception captured from the thread
                    raise self._analysis_exception
                if not self._analysis_response:
                    # Should not happen if thread finished and no exception, but check
                    raise RuntimeError("Analysis API call finished without response/exception.")

                # Process successful response (retrieved from instance var)
                response = self._analysis_response
                # raise_for_status was called in thread, but double-check defensively
                response.raise_for_status()
                response_data = response.json()

                # Validate LLM response structure
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content')
                if not content:
                    logger.error(
                        "Invalid analysis API response format: %s", response_data
                    )
                    return "[red]Analysis failed: Invalid response format from LLM.[/red]"

                # Format the final report
                final_report = (
                    f"# OSINT Analysis Report\n\n"
                    f"**Query:** {query}\n\n"
                    f"**Models Used:**\n"
                    f"- Text Analysis: `{text_model}`\n"
                    f"- Image Analysis: `{image_model}`\n\n"
                    f"---\n\n"
                    f"{content.strip()}" # Use the content from the LLM
                )
                return final_report

            # --- LLM Analysis Error Handling ---
            except httpx.HTTPStatusError as status_err:
                err_code = status_err.response.status_code
                err_text = ""
                try:
                    # Limit logged error text length
                    err_text = status_err.response.text[:500]
                except Exception: pass # Ignore errors reading response text
                logger.error("Analysis API HTTP Error %d: %s", err_code, err_text)
                if err_code == 429:
                    # Let handler raise RateLimitExceededError
                    self._handle_rate_limit(f"LLM Analysis ({text_model})", status_err)
                elif err_code == 401:
                    return ("[red]Analysis failed: Auth error (401). "
                            "Check OpenRouter API Key.[/red]")
                elif err_code == 400:
                    error_detail = "Bad Request"
                    try: # Attempt to get more detail from JSON error response
                        error_detail = status_err.response.json()['error']['message']
                    except Exception: pass
                    return (f"[red]Analysis failed: Bad Request (400). Check "
                            f"prompt/model. Detail: {error_detail}[/red]")
                elif err_code == 404:
                    return f"[red]Analysis failed: Model not found (404): {text_model}[/red]"
                else:
                    return (f"[red]Analysis failed: API request failed with "
                            f"status {err_code}.[/red]")
            except httpx.RequestError as req_err:
                logger.error("Analysis API Network Error: %s", req_err)
                return f"[red]Analysis failed: Network error. Details: {req_err}[/red]"
            except TimeoutError as timeout_err:
                return f"[red]Analysis failed: {timeout_err}[/red]"
            except Exception as e:
                # Catch unexpected errors during the analysis API call phase
                logger.error(
                    "Unexpected error during final analysis processing: %s", e,
                    exc_info=True
                )
                return f"[red]Analysis failed: Unexpected error. Details: {e}[/red]"
            finally:
                # Ensure analysis progress spinner stops
                if analysis_task is not None and analysis_task in self.progress.task_ids:
                    self.progress.remove_task(analysis_task)
                if self.progress.live.is_started:
                    self.progress.stop()
                # Reset state variables after attempt
                self._analysis_response = None
                self._analysis_exception = None

        # --- General Analysis Phase Error Handling ---
        except RateLimitExceededError as rle:
            # Catch rate limits from the data collection phase
            return f"[red]Analysis aborted due to rate limiting: {rle}[/red]"
        except Exception as e:
            # Catch unexpected errors in the broader analysis logic
            logger.error(
                "Unexpected error during analysis phase: %s", e, exc_info=True
            )
            return f"[red]Analysis failed unexpectedly: {e}[/red]"


    def _format_text_data(self, platform: str, username: str, data: dict) -> str:
        """Formats fetched data into a detailed text block for the analysis LLM."""
        if not data:
            return "" # Handle empty data case

        MAX_ITEMS_PER_TYPE = 30 # Limit items per section (tweets, comments, etc.)
        TEXT_SNIPPET_LENGTH = 1000 # Max length for text snippets in summary
        output_lines = []
        platform_display_name = platform.capitalize()
        user_prefix = ""
        display_username = username

        # --- Determine Display Name & Prefix ---
        # Use specific identifiers from fetched data if available
        user_info = data.get('user_info')
        profile_info = data.get('profile_info')
        stats_profile = data.get('stats', {}).get('user_profile')

        if platform == 'twitter' and user_info:
            display_username = user_info.get('username', username)
            user_prefix = "@"
        elif platform == 'reddit' and stats_profile:
            display_username = stats_profile.get('name', username)
            user_prefix = "u/"
        elif platform == 'mastodon' and user_info:
            display_username = user_info.get('acct', username) # Use full user@instance
        elif platform == 'bluesky' and profile_info:
            display_username = profile_info.get('handle', username)
        # HackerNews uses the provided username directly

        output_lines.append(
            f"### {platform_display_name} Data Summary for: {user_prefix}{display_username}"
        )

        # --- Twitter Formatting ---
        if platform == 'twitter' and user_info:
            created_at_dt = get_sort_key(user_info, 'created_at')
            created_at_str = (created_at_dt.strftime('%Y-%m-%d')
                              if created_at_dt > MIN_DATETIME_UTC else 'N/A')
            output_lines.append(
                f"- User Profile: '{user_info.get('name')}' "
                f"({user_prefix}{user_info.get('username')}), "
                f"ID: {user_info.get('id')}, Created: {created_at_str}, "
                f"Verified: {user_info.get('verified')}"
            )
            output_lines.append(f"  - Location: {user_info.get('location') or 'N/A'}")
            output_lines.append(f"  - Bio: {user_info.get('description') or 'N/A'}")
            pm = user_info.get('public_metrics', {})
            output_lines.append(
                f"  - Stats: Followers={pm.get('followers_count', 'N/A')}, "
                f"Following={pm.get('following_count', 'N/A')}, "
                f"Tweets={pm.get('tweet_count', 'N/A')}"
            )
            tweets = data.get('tweets', [])
            output_lines.append(f"\n**Recent Tweets (up to {MAX_ITEMS_PER_TYPE}):**")
            if not tweets:
                output_lines.append("- No tweets found in cache.")
            else:
                for i, t in enumerate(tweets[:MAX_ITEMS_PER_TYPE]):
                    created_dt = get_sort_key(t, 'created_at')
                    created_str = (created_dt.strftime('%Y-%m-%d %H:%M')
                                   if created_dt > MIN_DATETIME_UTC else 'N/A')
                    media_info = ""
                    if t.get('media'):
                        types = [m.get('type','?') for m in t['media']]
                        # Check if any media item has alt text
                        has_alt = any(m.get('alt_text') for m in t['media']
                                      if m.get('alt_text'))
                        media_info = (f" (Media: {len(t['media'])} {','.join(types)}"
                                      f"{'; Has Alt' if has_alt else ''})")
                    text = t.get('text', '[No Text]')
                    text_snippet = (text[:TEXT_SNIPPET_LENGTH] +
                                    ('...' if len(text) > TEXT_SNIPPET_LENGTH else ''))
                    metrics = t.get('metrics', {})
                    reply_info = (f", ReplyTo: {t.get('in_reply_to_user_id')}"
                                  if t.get('in_reply_to_user_id') else "")
                    output_lines.append(
                        f"- Tweet {i+1} ({created_str}):{media_info}{reply_info}\n"
                        f"  Content: {text_snippet}\n"
                        f"  Metrics: Likes={metrics.get('like_count', 0)}, "
                        f"RTs={metrics.get('retweet_count', 0)}, "
                        f"Replies={metrics.get('reply_count', 0)}, "
                        f"Quotes={metrics.get('quote_count', 0)}"
                    )

        # --- Reddit Formatting ---
        elif platform == 'reddit' and 'stats' in data:
            stats = data['stats']
            profile = stats.get('user_profile', {})
            if profile:
                created_dt = get_sort_key(profile, 'created_utc')
                created_str = (created_dt.strftime('%Y-%m-%d')
                               if created_dt > MIN_DATETIME_UTC else 'N/A')
                output_lines.append(
                    f"- User Info: Created: {created_str}, "
                    f"Link Karma: {profile.get('link_karma', 'N/A')}, "
                    f"Comment Karma: {profile.get('comment_karma', 'N/A')}, "
                    f"Suspended: {profile.get('is_suspended', 'N/A')}"
                )
            output_lines.append(
                f"- Cached Activity: Subs={stats.get('total_submissions', 0)}, "
                f"Comments={stats.get('total_comments', 0)}, "
                f"Media Posts={stats.get('submissions_with_media', 0)}, "
                f"Avg Sub Score={stats.get('avg_submission_score', 0):.1f}, "
                f"Avg Comment Score={stats.get('avg_comment_score', 0):.1f}"
            )
            submissions = data.get('submissions', [])
            output_lines.append(
                f"\n**Recent Submissions (up to {MAX_ITEMS_PER_TYPE}):**"
            )
            if not submissions:
                output_lines.append("- No submissions found in cache.")
            else:
                for i, s in enumerate(submissions[:MAX_ITEMS_PER_TYPE]):
                    created_dt = get_sort_key(s, 'created_utc')
                    created_str = (created_dt.strftime('%Y-%m-%d %H:%M')
                                   if created_dt > MIN_DATETIME_UTC else 'N/A')
                    media_info = (f" (Media: {len(s.get('media', []))})"
                                  if s.get('media') else "")
                    text = s.get('text', '')
                    text_snippet = (text[:TEXT_SNIPPET_LENGTH] +
                                    ('...' if len(text) > TEXT_SNIPPET_LENGTH else ''))
                    text_info = f"\n  Self-Text: {text_snippet}" if text_snippet else ""
                    output_lines.append(
                        f"- Sub {i+1} r/{s.get('subreddit', '?')} ({created_str}):"
                        f"{media_info}\n"
                        f"  Title: {s.get('title', '[No Title]')}\n"
                        f"  Score: {s.get('score', 0)}, "
                        f"Comments: {s.get('num_comments', 0)}, "
                        f"UpvoteRatio: {s.get('upvote_ratio', 'N/A')}\n"
                        f"  URL: {s.get('url', 'N/A')}{text_info}"
                    )
            comments = data.get('comments', [])
            output_lines.append(
                f"\n**Recent Comments (up to {MAX_ITEMS_PER_TYPE}):**"
            )
            if not comments:
                output_lines.append("- No comments found in cache.")
            else:
                for i, c in enumerate(comments[:MAX_ITEMS_PER_TYPE]):
                    created_dt = get_sort_key(c, 'created_utc')
                    created_str = (created_dt.strftime('%Y-%m-%d %H:%M')
                                   if created_dt > MIN_DATETIME_UTC else 'N/A')
                    text = c.get('text', '[No Text]')
                    text_snippet = (text[:TEXT_SNIPPET_LENGTH] +
                                    ('...' if len(text) > TEXT_SNIPPET_LENGTH else ''))
                    parent_info = (f", Parent: {c.get('parent_id')}"
                                   if c.get('parent_id') else "")
                    output_lines.append(
                        f"- Comment {i+1} r/{c.get('subreddit', '?')} "
                        f"on {c.get('link_id')} ({created_str}):\n"
                        f"  Content: {text_snippet}\n"
                        f"  Score: {c.get('score', 0)}, "
                        f"Is Submitter: {c.get('is_submitter')}{parent_info}, "
                        f"Link: {c.get('permalink', 'N/A')}"
                    )

        # --- HackerNews Formatting ---
        elif platform == 'hackernews' and 'stats' in data:
            stats = data['stats']
            output_lines.append(
                f"- Cached Activity: Items={stats.get('total_items_cached', 0)}, "
                f"Stories={stats.get('total_stories', 0)}, "
                f"Comments={stats.get('total_comments', 0)}, "
                f"Avg Story Pts={stats.get('average_story_points', 0):.1f}, "
                f"Avg Story Comments={stats.get('average_story_num_comments', 0):.1f}"
            )
            submissions = data.get('submissions', [])
            output_lines.append(
                f"\n**Recent Activity (up to {MAX_ITEMS_PER_TYPE}):**"
            )
            if not submissions:
                output_lines.append("- No activity found in cache.")
            else:
                for i, s in enumerate(submissions[:MAX_ITEMS_PER_TYPE]):
                    created_dt = get_sort_key(s, 'created_at') # Use datetime object
                    created_str = (created_dt.strftime('%Y-%m-%d %H:%M')
                                   if created_dt > MIN_DATETIME_UTC else 'N/A')
                    item_type = s.get('type', '?').capitalize()
                    title = s.get('title')
                    text = s.get('text', '')
                    text_snippet = (text[:TEXT_SNIPPET_LENGTH] +
                                    ('...' if len(text) > TEXT_SNIPPET_LENGTH else ''))
                    hn_link = f"https://news.ycombinator.com/item?id={s.get('objectID')}"
                    parent_info = (f", Parent: {s.get('parent_id')}"
                                   if s.get('parent_id') and item_type == 'Comment'
                                   else "")
                    story_info = (f", Story: {s.get('story_id')}"
                                  if s.get('story_id') and item_type == 'Comment'
                                  else "")
                    output_lines.append(f"- Item {i+1} ({item_type}, {created_str}):")
                    if title:
                        output_lines.append(f"  Title: {title}")
                    if s.get('url'):
                        output_lines.append(f"  URL: {s.get('url')}")
                    if text_snippet:
                        output_lines.append(f"  Text: {text_snippet}")
                    # Add points/comments for stories, points for comments
                    points = s.get('points')
                    num_comments = s.get('num_comments')
                    if item_type == 'Story':
                        output_lines.append(
                            f"  Stats: Pts={points if points is not None else 'N/A'}, "
                            f"Comments={num_comments if num_comments is not None else 'N/A'}"
                        )
                    elif item_type == 'Comment' and points is not None:
                        output_lines.append(f"  Stats: Pts={points}")
                    output_lines.append(f"  HN Link: {hn_link}{story_info}{parent_info}")

        # --- Bluesky Formatting ---
        elif platform == 'bluesky' and profile_info: # Use profile_info directly
            if profile_info:
                desc = profile_info.get('description', '').strip()
                desc_snippet = (desc[:150] + ('...' if len(desc) > 150 else ''))
                output_lines.append(
                    f"- Profile: '{profile_info.get('display_name')}' "
                    f"({profile_info.get('handle')}), DID: {profile_info.get('did')}"
                )
                if desc_snippet:
                    output_lines.append(f"  - Bio: {desc_snippet}")
                output_lines.append(
                    f"  - Stats: Posts={profile_info.get('posts_count', 'N/A')}, "
                    f"Following={profile_info.get('follows_count', 'N/A')}, "
                    f"Followers={profile_info.get('followers_count', 'N/A')}"
                )
            stats = data.get('stats', {})
            output_lines.append(
                f"- Cached Activity: Posts={stats.get('total_posts_cached', 0)}, "
                f"Media Posts={stats.get('posts_with_media', 0)}, "
                f"Avg Likes={stats.get('avg_likes', 0):.1f}, "
                f"Avg Reposts={stats.get('avg_reposts', 0):.1f}, "
                f"Avg Replies={stats.get('avg_replies', 0):.1f}"
            )
            posts = data.get('posts', [])
            output_lines.append(f"\n**Recent Posts (up to {MAX_ITEMS_PER_TYPE}):**")
            if not posts:
                output_lines.append("- No posts found in cache.")
            else:
                for i, p in enumerate(posts[:MAX_ITEMS_PER_TYPE]):
                    created_dt = get_sort_key(p, 'created_at')
                    created_str = (created_dt.strftime('%Y-%m-%d %H:%M')
                                   if created_dt > MIN_DATETIME_UTC else 'N/A')
                    media_info = ""
                    if p.get('media'):
                        has_alt = any(m.get('alt_text') for m in p['media']
                                      if m.get('alt_text'))
                        media_info = (f" (Media: {len(p['media'])}"
                                      f"{'; Has Alt' if has_alt else ''})")
                    text = p.get('text', '[No Text]')
                    text_snippet = (text[:TEXT_SNIPPET_LENGTH] +
                                    ('...' if len(text) > TEXT_SNIPPET_LENGTH else ''))
                    embed_info = p.get('embed')
                    embed_desc = (f" (Embed: {embed_info['type']})"
                                  if embed_info else "")
                    output_lines.append(
                        f"- Post {i+1} ({created_str}):{media_info}{embed_desc}\n"
                        f"  Content: {text_snippet}\n"
                        f"  Stats: Likes={p.get('likes', 0)}, "
                        f"Reposts={p.get('reposts', 0)}, "
                        f"Replies={p.get('reply_count', 0)}\n"
                        f"  URI: {p.get('uri', 'N/A')}"
                    )

        # --- Mastodon Formatting ---
        elif platform == 'mastodon' and user_info: # Use user_info directly
            if user_info:
                created_dt = get_sort_key(user_info, 'created_at')
                created_str = (created_dt.strftime('%Y-%m-%d')
                               if created_dt > MIN_DATETIME_UTC else 'N/A')
                bio_text = '[Bio Parse Error]'
                try:
                    # Safely parse bio HTML
                    bio_soup = BeautifulSoup(user_info.get('note_html', ''),
                                             'html.parser')
                    bio_text = bio_soup.get_text(separator=' ', strip=True)
                except Exception as bio_parse_err:
                    logger.warning("Error parsing Mastodon bio HTML: %s", bio_parse_err)
                bio_snippet = (bio_text[:150] + ('...' if len(bio_text) > 150 else ''))
                output_lines.append(
                    f"- User Profile: '{user_info.get('display_name')}' "
                    f"({user_info.get('acct')}), ID: {user_info.get('id')}, "
                    f"Created: {created_str}, Locked: {user_info.get('locked')}, "
                    f"Bot: {user_info.get('bot')}"
                )
                output_lines.append(f"  - Bio: {bio_snippet}")
                # Safely dump profile fields as JSON string
                fields_str = '[]'
                try:
                    fields_str = json.dumps(user_info.get('fields', []))
                except TypeError: pass # Ignore if fields aren't serializable
                output_lines.append(f"  - Profile Fields: {fields_str}")
                output_lines.append(
                    f"  - Stats: Followers={user_info.get('followers_count', 'N/A')}, "
                    f"Following={user_info.get('following_count', 'N/A')}, "
                    f"Posts={user_info.get('statuses_count', 'N/A')}"
                )
            stats = data.get('stats', {})
            output_lines.append(
                f"- Cached Activity: Posts={stats.get('total_posts_cached', 0)}, "
                f"Originals={stats.get('total_original_posts', 0)}, "
                f"Reblogs={stats.get('total_reblogs_in_cache', 0)}, "
                f"Media={stats.get('posts_with_media', 0)}, "
                f"Avg Favs(Orig)={stats.get('avg_favourites_on_originals', 0):.1f}, "
                f"Avg Boosts(Orig)={stats.get('avg_boosts_on_originals', 0):.1f}"
            )
            posts = data.get('posts', [])
            output_lines.append(
                f"\n**Recent Posts (incl. Reblogs, up to {MAX_ITEMS_PER_TYPE}):**"
            )
            if not posts:
                output_lines.append("- No posts found in cache.")
            else:
                for i, p in enumerate(posts[:MAX_ITEMS_PER_TYPE]):
                    created_dt = get_sort_key(p, 'created_at')
                    created_str = (created_dt.strftime('%Y-%m-%d %H:%M')
                                   if created_dt > MIN_DATETIME_UTC else 'N/A')
                    media_info = ""
                    if p.get('media'):
                        # Use 'description' for Mastodon alt text
                        has_alt = any(m.get('description') for m in p['media']
                                      if m.get('description'))
                        media_info = (f" (Media: {len(p['media'])}"
                                      f"{'; Has Alt' if has_alt else ''})")
                    spoiler = p.get('spoiler_text', '')
                    spoiler_info = f" (CW: {spoiler})" if spoiler else ""
                    is_boost = p.get('is_reblog', False)
                    boost_info = ""
                    if is_boost:
                        boost_info = (f" (Boost of "
                                      f"{p.get('reblog_original_author', '?')})")
                    # Use cleaned text
                    text_snippet = p.get('text_cleaned', '')
                    # Determine display text based on type and content
                    if spoiler and not text_snippet:
                        text_display = "[CW Text Only]"
                    elif is_boost and not text_snippet:
                        # Boosts might have no text of their own
                        text_display = "[Boost Content Only]"
                    else:
                        text_display = (
                            text_snippet[:TEXT_SNIPPET_LENGTH] +
                            ('...' if len(text_snippet) > TEXT_SNIPPET_LENGTH else '')
                            or "[No Text]" # Handle empty cleaned text
                        )
                    reply_info = (f", ReplyTo: {p.get('in_reply_to_id')}"
                                  if p.get('in_reply_to_id') else "")
                    tags = ", ".join([t['name'] for t in p.get('tags', []) if t.get('name')])
                    tag_info = f", Tags: [{tags}]" if tags else ""
                    output_lines.append(
                        f"- Post {i+1} ({p.get('visibility', '?')}, "
                        f"Lang: {p.get('language', '?')}, {created_str}):"
                        f"{boost_info}{spoiler_info}{media_info}{reply_info}\n"
                        f"  Content: {text_display}\n"
                        f"  Stats: Favs={p.get('favourites_count', 0)}, "
                        f"Boosts={p.get('reblogs_count', 0)}, "
                        f"Replies={p.get('replies_count', 0)}{tag_info}\n"
                        f"  Link: {p.get('url', 'N/A')}"
                    )
                    # Add link to original post if it's a boost
                    if is_boost and p.get('reblog_original_url'):
                        output_lines.append(
                            f"  Original Post: {p['reblog_original_url']}"
                        )

        # --- Fallback Formatting (If platform logic missing/data incomplete) ---
        else:
            logger.warning(
                "Using fallback formatting for platform '%s', user '%s'. "
                "Data structure might be incomplete or platform not fully handled.",
                platform, username
            )
            output_lines.append("\n**Generic Data Overview:**")
            # Provide a truncated preview of the raw data
            try:
                raw_preview = json.dumps(data, sort_keys=True, default=str)
                output_lines.append(
                    f"- Raw Preview: {raw_preview[:TEXT_SNIPPET_LENGTH]}..."
                )
            except Exception:
                 output_lines.append(f"- Raw Preview: {str(data)[:TEXT_SNIPPET_LENGTH]}...")


        return "\n".join(output_lines)


    def _call_openrouter(self, json_data: dict):
        """
        Worker function to make the OpenRouter API call in a thread.
        Sets instance variables `_analysis_response` and `_analysis_exception`.
        """
        thread_response: Optional[httpx.Response] = None
        thread_exception: Optional[Exception] = None
        try:
            # Access property to ensure client is initialized
            client = self.openrouter
            thread_response = client.post("/chat/completions", json=json_data)
            thread_response.raise_for_status() # Raise HTTP errors in thread
        except Exception as e:
            logger.error(
                "OpenRouter API call error in thread: %s: %s", type(e).__name__, e
            )
            if isinstance(e, httpx.HTTPStatusError) and e.response:
                logger.error("Error Response status: %d", e.response.status_code)
                try:
                    logger.error("Error Response content: %s", e.response.text[:500])
                except Exception: pass # Ignore errors reading response text
            thread_exception = e # Store exception to be re-raised in main thread
        finally:
            # Set shared instance variables *after* call finishes/fails
            self._analysis_response = thread_response
            self._analysis_exception = thread_exception


    def _save_output(self, content: str, query: str,
                     platforms_analyzed: List[str], format_type: str = "markdown"):
        """Saves the analysis report to a file in markdown or JSON format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_dir / 'outputs'
        # Sanitize query and platforms for filename
        safe_query = "".join(c if c.isalnum() else '_' for c in query[:40]).strip('_')
        safe_platforms = "_".join(sorted(platforms_analyzed))[:40].strip('_')
        filename_base = f"analysis_{timestamp}_{safe_platforms}_{safe_query}"

        try:
            if format_type == "json":
                filename = output_dir / f"{filename_base}.json"
                # Create JSON structure including metadata and the raw markdown report
                data_to_save = {
                    "analysis_metadata": {
                         "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                         "query": query,
                         "platforms_analyzed": platforms_analyzed,
                         "output_format": "json",
                         "text_model": os.getenv('ANALYSIS_MODEL', 'unknown'),
                         "image_model": os.getenv('IMAGE_ANALYSIS_MODEL', 'unknown'),
                    },
                    "analysis_report_markdown": content, # Store the raw markdown
                }
                filename.write_text(json.dumps(data_to_save, indent=2),
                                    encoding='utf-8')
            else: # Default to markdown
                filename = output_dir / f"{filename_base}.md"
                # Create YAML front matter for metadata in Markdown file
                # Basic escaping for query in YAML string
                formatted_query = query.replace('"', '\\"')
                md_metadata = f"""---
query: "{formatted_query}"
platforms: [{', '.join(platforms_analyzed)}]
timestamp_utc: {datetime.now(timezone.utc).isoformat()}
text_model: {os.getenv('ANALYSIS_MODEL', 'unknown')}
image_model: {os.getenv('IMAGE_ANALYSIS_MODEL', 'unknown')}
---

"""
                # Content is already markdown formatted by the analyze method
                full_content = md_metadata + content
                filename.write_text(full_content, encoding='utf-8')

            self.console.print(f"[green]Analysis saved to: {filename}[/green]")

        except Exception as e:
            self.console.print(f"[bold red]Failed to save output: {e}[/bold red]")
            logger.error(
                "Failed to save output file %s.%s: %s",
                filename_base, format_type, e, exc_info=True
            )

    def get_available_platforms(self, check_creds=True) -> List[str]:
        """Checks environment variables to see which platforms are configured."""
        available = set() # Use a set to avoid duplicates

        # Conditionally check credentials based on flag
        if not check_creds or os.getenv('TWITTER_BEARER_TOKEN'):
            available.add('twitter')
        if not check_creds or all(os.getenv(k) for k in [
                'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']):
            available.add('reddit')
        if not check_creds or all(os.getenv(k) for k in [
                'BLUESKY_IDENTIFIER', 'BLUESKY_APP_SECRET']):
             available.add('bluesky')
        if not check_creds or all(os.getenv(k) for k in [
                'MASTODON_ACCESS_TOKEN', 'MASTODON_API_BASE_URL']):
             # Also check if Mastodon URL is valid-looking
             base_url = os.getenv('MASTODON_API_BASE_URL')
             parsed_url = urlparse(base_url) if base_url else None
             if parsed_url and parsed_url.scheme and parsed_url.netloc:
                 available.add('mastodon')
             elif check_creds: # Log warning only if creds exist but URL is bad
                 logger.warning(
                    "Mastodon creds found, but MASTODON_API_BASE_URL "
                    "is invalid or missing."
                 )

        # HackerNews doesn't require credentials, always potentially available
        available.add('hackernews')
        return sorted(list(available))

    # --- Interactive Mode ---
    def run(self):
        """Runs the interactive command-line interface."""
        self.console.print(Panel(
            "[bold blue]Social Media OSINT analyzer[/bold blue]\n"
            "Collects and analyzes user activity across multiple platforms using LLMs.\n"
            "Ensure API keys and identifiers are set in your `.env` file.",
            title="Welcome",
            border_style="blue",
        ))

        # Get platforms that *actually* have credentials configured
        available_platforms_with_creds = self.get_available_platforms(check_creds=True)
        all_potentially_available = self.get_available_platforms(check_creds=False)

        if not available_platforms_with_creds:
            self.console.print(
                "[bold yellow]Warning:[/bold yellow] No API credentials seem to "
                "be configured in your `.env` file."
            )
            if 'hackernews' in all_potentially_available:
                self.console.print("Only HackerNews analysis will be available.")
                # Set available list to only HN for the menu
                available_platforms_with_creds = ['hackernews']
            else:
                self.console.print(
                    "[bold red]Error: No platforms available (credentials missing "
                    "and HackerNews fetcher seems unavailable?). Exiting.[/bold red]"
                )
                return
        elif 'hackernews' not in available_platforms_with_creds:
             # Add HN if other platforms are configured but HN wasn't explicitly listed
             # (get_available_platforms includes it implicitly)
             available_platforms_with_creds.append('hackernews')
             available_platforms_with_creds.sort()


        while True:
            self.console.print("\n[bold cyan]Select Platform(s) for Analysis:[/bold cyan]")
            # Use the list of platforms with credentials (plus HN) for the menu
            current_available_menu = available_platforms_with_creds[:]

            # Sort platforms for consistent menu order
            platform_priority = {'twitter': 1, 'bluesky': 2, 'mastodon': 3,
                                 'reddit': 4, 'hackernews': 5}
            current_available_menu.sort(key=lambda x: platform_priority.get(x, 999))

            # Build menu options dictionary
            platform_options = {str(i+1): p
                                for i, p in enumerate(current_available_menu)}
            num_platforms = len(current_available_menu)
            cross_platform_key = str(num_platforms + 1)
            exit_key = str(num_platforms + 2)
            # Add 'cross-platform' option only if multiple platforms are available
            if num_platforms > 1:
                platform_options[cross_platform_key] = "cross-platform"
            platform_options[exit_key] = "exit"

            # Display menu
            for key, name in platform_options.items():
                display_name = (name.capitalize()
                                if key not in [cross_platform_key, exit_key]
                                else name)
                self.console.print(f"{key}. {display_name}")

            choice = Prompt.ask(
                "Enter number(s) (e.g., 1 or 1,3 or 'cross-platform')",
                default=exit_key).strip().lower()

            if choice == exit_key or choice == 'exit':
                break

            selected_platform_keys = []
            # Handle 'cross-platform' selection
            if (cross_platform_key in platform_options and
                    (choice == cross_platform_key or choice == 'cross-platform')):
                # Select all available platform keys except the special ones
                selected_platform_keys = [k for k, v in platform_options.items()
                                          if v not in ["cross-platform", "exit"]]
                selected_names = [platform_options[k].capitalize()
                                  for k in selected_platform_keys]
                self.console.print(
                    f"Selected: Cross-Platform Analysis ({', '.join(selected_names)})"
                )
            else:
                # Handle selection by number(s)
                raw_keys = [k.strip() for k in choice.split(',')]
                # Filter valid keys from the user input
                valid_keys = [k for k in raw_keys if k in platform_options and
                              k not in [cross_platform_key, exit_key]]
                if not valid_keys:
                    self.console.print("[yellow]Invalid selection.[/yellow]")
                    continue
                selected_platform_keys = valid_keys
                selected_names = [platform_options[k].capitalize()
                                  for k in selected_platform_keys]
                self.console.print(f"Selected: {', '.join(selected_names)}")

            # --- Get Usernames for Selected Platforms ---
            platforms_to_query: Dict[str, List[str]] = {}
            all_users_valid = True # Track if any user input was invalid
            try:
                for key in selected_platform_keys:
                    platform_name = platform_options.get(key)
                    if not platform_name: continue # Should not happen

                    # Build prompt message with platform-specific guidance
                    prompt_parts = [f"{platform_name.capitalize()} username(s)"]
                    if platform_name == 'twitter': prompt_parts.append("(no '@')")
                    elif platform_name == 'reddit': prompt_parts.append("(no 'u/')")
                    elif platform_name == 'bluesky': prompt_parts.append("(e.g., 'handle.bsky.social')")
                    elif platform_name == 'mastodon': prompt_parts.append("(format: 'user@instance.domain')")
                    elif platform_name == 'hackernews': prompt_parts.append("(case-sensitive)")
                    prompt_parts.append("(comma-separated)")
                    prompt_message = " ".join(prompt_parts)

                    user_input = Prompt.ask(prompt_message, default="").strip()
                    if not user_input:
                        # Only require username for HN, others can be skipped
                        if platform_name == 'hackernews':
                            self.console.print(f"[red]Username required for {platform_name}.[/red]")
                            all_users_valid = False
                        else:
                             self.console.print(f"[yellow]No usernames entered for {platform_name}. Skipping.[/yellow]")
                        continue # Skip to next platform if no input

                    # Split and clean usernames
                    usernames = [u.strip() for u in user_input.split(',') if u.strip()]
                    if not usernames:
                         self.console.print(f"[yellow]No valid usernames parsed for {platform_name}. Skipping.[/yellow]")
                         continue # Skip if only whitespace/commas entered

                    # --- Validate Usernames per Platform ---
                    validated_users = []
                    if platform_name == 'mastodon':
                        for u in usernames:
                            if '@' in u and '.' in u.split('@', 1)[1]:
                                validated_users.append(u)
                            else:
                                # Try to infer instance from configured URL
                                default_instance_url = os.getenv('MASTODON_API_BASE_URL', '')
                                default_domain = urlparse(default_instance_url).netloc if default_instance_url else None
                                if default_domain:
                                    assumed_user = f"{u}@{default_domain}"
                                    if Confirm.ask(
                                        f"[yellow]Assume '{u}' is on "
                                        f"'{default_domain}' ('{assumed_user}')?"
                                        f"[/yellow]", default=True):
                                        validated_users.append(assumed_user)
                                    else:
                                        self.console.print(f"[yellow]Skipping '{u}'.[/yellow]")
                                else:
                                    # Cannot infer instance
                                    self.console.print(
                                        f"[red]Invalid Mastodon format: '{u}'. "
                                        "Cannot infer instance. Skipping.[/red]"
                                    )
                                    all_users_valid = False
                    elif platform_name == 'bluesky':
                         for u in usernames:
                             # Basic check for handle format (contains a dot)
                             if '.' in u:
                                 validated_users.append(u)
                             else:
                                 self.console.print(
                                     f"[red]Invalid Bluesky handle format: '{u}'. "
                                     "Skipping.[/red]"
                                 )
                                 all_users_valid = False
                    else:
                        # No specific validation needed for Twitter, Reddit, HN here
                        validated_users = usernames

                    # Add validated users to the query dict
                    if validated_users:
                        if platform_name not in platforms_to_query:
                            platforms_to_query[platform_name] = []
                        platforms_to_query[platform_name].extend(validated_users)
                    elif usernames: # Log if users were entered but none validated
                         logger.warning("No valid users remained for %s after validation.", platform_name)


                # Check if any targets remain after input/validation
                if not platforms_to_query:
                    if not all_users_valid:
                        self.console.print("[yellow]No valid targets specified after validation. Returning to menu.[/yellow]")
                    else:
                        self.console.print("[yellow]No targets entered. Returning to menu.[/yellow]")
                    continue # Go back to platform selection

                # --- Start Analysis Session ---
                self._run_analysis_loop(platforms_to_query)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Operation cancelled by user.[/yellow]")
                # Optionally ask if they want to exit the whole program
                # if Confirm.ask("Exit analyzer completely?", default=False): break
                continue # Go back to platform selection by default
            except RuntimeError as e:
                # Catch config/runtime errors (e.g., missing keys detected late)
                self.console.print(f"\n[bold red]Config/Runtime Error:[/bold red] {e}")
                self.console.print("Check .env file, API keys, network, URLs.")
                if not Confirm.ask("Exit analyzer?", default=True):
                    continue # Try again if they don't want to exit
                else:
                    break # Exit main loop
            except Exception as e:
                # Catch unexpected errors in the main loop
                logger.error("Unexpected error in main run loop: %s", e, exc_info=True)
                self.console.print(f"\n[bold red]Unexpected error:[/bold red] {e}")
                if Confirm.ask("Try again?", default=False):
                    continue # Try again if requested
                else:
                    break # Exit main loop

        self.console.print("\n[blue]Exiting Social Media analyzer.[/blue]")


    def _run_analysis_loop(self, platforms: Dict[str, List[str]]):
        """Inner loop for performing analysis queries on selected targets."""
        # Create descriptive label for the current session targets
        platform_labels = []
        platform_names_list = sorted(platforms.keys()) # For saving output later
        for pf, users in sorted(platforms.items()): # Sort for consistent display
            user_prefix = {"twitter": "@", "reddit": "u/"}.get(pf, "")
            display_users = [f"{user_prefix}{u}" for u in users]
            platform_labels.append(f"{pf.capitalize()}: {', '.join(display_users)}")
        platform_info = " | ".join(platform_labels)

        self.console.print(Panel(
            f"Targets: {platform_info}\n"
            "Commands: `exit` (return), `refresh` (force fetch), `help`",
            title="🔎 Analysis Session", border_style="cyan", expand=False
        ))

        while True:
            try:
                query = Prompt.ask("\n[bold green]Analysis Query>[/bold green]").strip()
                if not query:
                    continue # Ask again if empty input

                cmd = query.lower()
                # --- Handle Commands ---
                if cmd == 'exit':
                    self.console.print("[yellow]Returning to platform selection.[/yellow]")
                    break # Exit inner loop
                if cmd == 'help':
                    self.console.print(Panel(
                        "**Commands:**\n"
                        "- `exit`: Return to platform selection.\n"
                        "- `refresh`: Force data fetch for all current targets.\n"
                        "- `help`: Show this help message.\n\n"
                        "**To analyze:** Type your question or prompt.",
                        title="Help", border_style="blue", expand=False
                    ))
                    continue
                if cmd == 'refresh':
                    if Confirm.ask(
                        "[yellow]Force refresh data for all current targets? "
                        "(This will use API calls)[/yellow]", default=False):
                        self._force_refresh_targets(platforms) # Call helper
                    continue # Ask for next query after refresh attempt

                # --- Perform Analysis ---
                self.console.print(
                    f"[cyan]Starting analysis for query:[/cyan] '{query}'",
                    highlight=False
                )
                # Ensure any previous progress bars are stopped
                if self.progress.live.is_started:
                    self.progress.stop()
                # Call the main analysis method
                # Pass a copy of platforms dict in case analyze modifies it (it shouldn't)
                analysis_result = self.analyze(platforms.copy(), query)

                # --- Display and Save Results ---
                if analysis_result:
                    result_lower = analysis_result.strip().lower()
                    # Check if the result indicates an error or warning
                    is_error = result_lower.startswith(("[red]", "error:",
                                                        "analysis failed",
                                                        "analysis aborted"))
                    is_warning = result_lower.startswith(("[yellow]", "warning:"))
                    border_col = "red" if is_error else ("yellow" if is_warning else "green")

                    # Display result using Rich Markdown and Panel
                    self.console.print(Panel(
                        Markdown(analysis_result),
                        title="Analysis Report",
                        border_style=border_col,
                        expand=False # Prevent panel from taking full width
                    ))

                    # Handle saving the report
                    if not is_error: # Only offer to save successful reports
                        save_report = False
                        # Check --no-auto-save flag from args
                        no_auto_save = self.args and self.args.no_auto_save
                        if no_auto_save:
                            # Ask user explicitly if they want to save
                            if Confirm.ask("Save this report?", default=True):
                                save_report = True
                        else:
                            # Auto-save enabled
                            save_format = self.args.format if self.args else 'markdown'
                            self.console.print(
                                f"[cyan]Auto-saving analysis as {save_format}...[/cyan]"
                            )
                            save_report = True

                        if save_report:
                            # Determine format (use arg or prompt if needed)
                            save_format_final = self.args.format if self.args else 'markdown'
                            if no_auto_save: # If auto-save disabled, ask for format too
                                save_format_final = Prompt.ask(
                                    "Save format?", choices=["markdown", "json"],
                                    default=save_format_final
                                )
                            self._save_output(analysis_result, query,
                                              platform_names_list, save_format_final)
                else:
                    self.console.print("[red]Analysis returned no result.[/red]")

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Query cancelled by user.[/yellow]")
                if Confirm.ask("\nExit analysis session?", default=False):
                    break # Exit inner loop
                # else: continue to ask for next query
            except RateLimitExceededError as rle:
                # Already handled in analyze/fetch, but show message here too
                self.console.print(f"[yellow]Rate limit hit during operation: {rle} "
                                   "Please wait before retrying.[/yellow]")
                if Confirm.ask("Exit analysis session due to rate limit?", default=False):
                    break # Exit inner loop
            except Exception as e:
                # Catch unexpected errors in the analysis loop itself
                logger.error(
                    "Unexpected error in analysis loop: %s", e, exc_info=True
                )
                self.console.print(f"\n[bold red]Unexpected error occurred:[/bold red] {e}")
                if not Confirm.ask("Continue session despite error?", default=True):
                    break # Exit inner loop

    def _force_refresh_targets(self, platforms: Dict[str, List[str]]):
        """Helper function to force refresh data for current targets."""
        total_targets = sum(len(u) for u in platforms.values())
        if total_targets == 0:
            self.console.print("[yellow]No targets to refresh.[/yellow]")
            return

        refresh_task = self.progress.add_task(
            "[yellow]Refreshing data...", total=total_targets
        )
        self.progress.start()
        failed_refreshes = []
        try:
            for platform, usernames in platforms.items():
                fetcher = getattr(self, f'fetch_{platform}', None)
                if not fetcher:
                    logger.warning("No fetcher for %s, cannot refresh.", platform)
                    # Advance progress for skipped users
                    if refresh_task in self.progress.task_ids:
                       self.progress.advance(refresh_task, advance=len(usernames))
                    continue

                for username in usernames:
                    user_prefix = {"twitter": "@", "reddit": "u/"}.get(platform, "")
                    display_name = f"{user_prefix}{username}"
                    if refresh_task in self.progress.task_ids:
                        self.progress.update(
                            refresh_task,
                            description=f"[yellow]Refreshing {platform}/{display_name}..."
                        )
                    try:
                        # Call fetcher with force_refresh=True
                        fetcher(username=username, force_refresh=True)
                    except Exception as e:
                        # Catch any error during refresh for a specific user
                        failed_refreshes.append((platform, display_name))
                        error_msg = (f"Refresh failed for {platform}/{display_name}: "
                                     f"{type(e).__name__}")
                        logger.error(error_msg + f" - {e}", exc_info=False)
                        self.console.print(f"[red]{error_msg}[/red]")
                    finally:
                        # Advance progress for this user
                        if refresh_task in self.progress.task_ids:
                            self.progress.advance(refresh_task)
        finally:
            # Ensure progress bar stops cleanly
            if refresh_task in self.progress.task_ids:
                # Mark task as complete even if some failed
                task_obj = self.progress._tasks.get(refresh_task) # Access internal task
                if task_obj:
                    self.progress.update(refresh_task, completed=task_obj.total,
                                     description="[green]Refresh finished.")
                self.progress.remove_task(refresh_task) # Clean up task
            if self.progress.live.is_started:
                self.progress.stop()

            if failed_refreshes:
                self.console.print(
                    f"[yellow]Refresh encountered issues for "
                    f"{len(failed_refreshes)} target(s). Check logs.[/yellow]"
                )
            else:
                self.console.print("[green]Data refresh attempt completed for all targets.[/green]")


    # --- Non-Interactive Mode (stdin processing) ---
    def process_stdin(self, output_format: str):
        """Processes analysis request from JSON input via stdin."""
        # Log status messages to stderr to avoid polluting stdout JSON output
        self.console.print(
            "[cyan]Processing analysis request from stdin...[/cyan]", file=sys.stderr
        )
        try:
            # --- Load and Validate Input JSON ---
            try:
                input_data = json.load(sys.stdin)
            except json.JSONDecodeError as jde:
                raise ValueError(f"Invalid JSON received via stdin: {jde}")

            platforms_input = input_data.get("platforms")
            query = input_data.get("query")

            # Basic validation of input structure
            if not isinstance(platforms_input, dict) or not platforms_input:
                raise ValueError("Invalid or missing 'platforms' dictionary in JSON input.")
            if not isinstance(query, str) or not query:
                raise ValueError("Invalid or missing 'query' string in JSON input.")

            # --- Filter and Validate Platforms/Usernames ---
            valid_platforms_to_analyze: Dict[str, List[str]] = {}
            # Get platforms that have credentials configured (+ HN)
            configured = self.get_available_platforms(check_creds=True)
            usable_platforms = set(configured)
            # HN is always usable if fetcher exists
            if hasattr(self, 'fetch_hackernews'):
                usable_platforms.add('hackernews')

            for platform, usernames in platforms_input.items():
                platform = platform.lower() # Normalize platform name
                if platform not in usable_platforms:
                    logger.warning(
                        "Platform '%s' requested via stdin is not supported or "
                        "configured. Skipping.", platform
                    )
                    continue

                # Normalize usernames input (string or list)
                usernames_list: List[str] = []
                if isinstance(usernames, str):
                    usernames_list = [usernames.strip()] if usernames.strip() else []
                elif isinstance(usernames, list):
                    usernames_list = [u.strip() for u in usernames
                                      if isinstance(u, str) and u.strip()]
                else:
                    logger.warning(
                        "Invalid username format provided for platform '%s' "
                        "in stdin JSON. Expected string or list of strings. Skipping.",
                        platform
                    )
                    continue
                if not usernames_list:
                    logger.warning(
                        "No valid usernames provided for platform '%s' in stdin JSON. "
                        "Skipping.", platform
                    )
                    continue

                # Apply platform-specific validation (like in interactive mode)
                validated_users = []
                if platform == 'mastodon':
                    for u in usernames_list:
                        if '@' in u and '.' in u.split('@', 1)[1]:
                            validated_users.append(u)
                        else:
                            logger.warning(
                                "Invalid Mastodon username format '%s' received via "
                                "stdin. Skipping.", u
                            )
                elif platform == 'bluesky':
                     for u in usernames_list:
                          if '.' in u:
                              validated_users.append(u)
                          else:
                              logger.warning(
                                  "Invalid Bluesky handle format '%s' received via "
                                  "stdin. Skipping.", u
                              )
                else:
                    validated_users = usernames_list # No specific format check needed

                # Add validated users for the platform
                if validated_users:
                    valid_platforms_to_analyze[platform] = validated_users
                elif usernames_list: # Log if users were provided but none validated
                    logger.warning(
                        "No valid users remained for platform '%s' after "
                        "validation (stdin input).", platform
                    )

            # Check if any valid targets remain
            if not valid_platforms_to_analyze:
                raise ValueError(
                    "No valid/usable platforms and usernames found in stdin JSON "
                    "after validation and configuration checks."
                )

            # --- Perform Analysis ---
            # Ensure progress bar is stopped if active (shouldn't be in stdin mode)
            if self.progress.live.is_started:
                self.progress.stop()
            # Call the core analysis function
            analysis_report = self.analyze(valid_platforms_to_analyze, query)

            # --- Handle Output ---
            if analysis_report:
                result_lower = analysis_report.strip().lower()
                is_error = result_lower.startswith(("[red]", "[yellow]", "error:",
                                                    "warning:", "analysis failed",
                                                    "analysis aborted"))

                if not is_error:
                    platform_names_list = sorted(valid_platforms_to_analyze.keys())
                    # If --no-auto-save, print raw markdown report to stdout
                    if self.args and self.args.no_auto_save:
                        print(analysis_report) # Output report directly
                        sys.exit(0) # Successful exit
                    else:
                        # Save report using specified format (or default)
                        output_format_to_use = self.args.format if self.args else 'markdown'
                        self._save_output(analysis_report, query,
                                          platform_names_list, output_format_to_use)
                        # Print success message to stderr
                        self.console.print(
                            f"[green]Analysis complete. Output saved as "
                            f"{output_format_to_use}.[/green]", file=sys.stderr
                        )
                        sys.exit(0) # Successful exit
                else:
                    # Analysis produced an error message
                    sys.stderr.write(
                        "[ERROR] Analysis failed or produced an error report:\n"
                    )
                    sys.stderr.write(analysis_report + "\n")
                    sys.exit(1) # Exit with error code
            else:
                # Analysis returned no result at all
                sys.stderr.write("[ERROR] Analysis returned no result.\n")
                sys.exit(1) # Exit with error code

        # --- Stdin Processing Error Handling ---
        except ValueError as ve:
            logger.error("Invalid stdin input: %s", ve)
            sys.stderr.write(f"Error: Invalid input - {ve}\n")
            sys.exit(1)
        except RateLimitExceededError as rle:
            logger.error("Rate limit error during stdin processing: %s", rle)
            sys.stderr.write(f"Error: Rate limit encountered - {rle}\n")
            sys.exit(2) # Use different exit code for rate limits?
        except RuntimeError as rte:
            # Catch runtime/config errors during stdin processing
            logger.error("Runtime error during stdin processing: %s", rte)
            sys.stderr.write(f"Error: Runtime error - {rte}\n")
            sys.exit(1)
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(
                "Unexpected error during stdin processing: %s", e, exc_info=True
            )
            sys.stderr.write(f"Error: Unexpected error - {e}\n")
            sys.exit(1)


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Social Media OSINT analyzer using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Environment Variables Required:
  OPENROUTER_API_KEY: Your OpenRouter API key.
  IMAGE_ANALYSIS_MODEL: Model name for image analysis (e.g., google/gemini-pro-vision).
  ANALYSIS_MODEL: Optional text analysis model override (default: mistralai/mixtral-8x7b-instruct).

Platform Credentials (at least one set + HackerNews potentially):
  TWITTER_BEARER_TOKEN: For Twitter v2 API.
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT: For Reddit API.
  BLUESKY_IDENTIFIER (handle), BLUESKY_APP_SECRET (app password): For Bluesky API.
  MASTODON_ACCESS_TOKEN, MASTODON_API_BASE_URL (e.g., https://mastodon.social): For Mastodon API.

See README.md for setup and usage details.
"""
    )
    parser.add_argument(
        '--stdin', action='store_true',
        help="Read analysis request from stdin as JSON instead of interactive mode."
    )
    parser.add_argument(
        '--format', choices=['json', 'markdown'], default='markdown',
        help="Output format for saved reports (default: markdown)."
    )
    parser.add_argument(
        '--no-auto-save', action='store_true',
        help="Disable auto-saving reports. In interactive mode, prompts to save. "
             "In stdin mode, prints report to stdout if successful."
    )
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING', help='Set logging level (default: WARNING).'
    )

    args = parser.parse_args()

    # --- Configure Logging ---
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.WARNING)
    # Set root logger level first
    logging.getLogger().setLevel(log_level_numeric)
    # Set specific logger level (optional, but good practice)
    logger.setLevel(log_level_numeric)
    # Ensure handlers also respect the level
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level_numeric)
    logger.info("Logging level set to %s", args.log_level)

    # --- Initialize and Run ---
    try:
        # Pass parsed args to the analyzer instance
        analyzer = SocialOSINTLM(args=args)
        if args.stdin:
            analyzer.process_stdin(args.format)
        else:
            analyzer.run() # Start interactive mode
    except RuntimeError as e:
        # Catch critical initialization errors (e.g., missing core env vars)
        # Log error specifically
        logging.getLogger('SocialOSINTLM').critical(
            "Initialization failed: %s", e, exc_info=False # No need for full traceback here
        )
        # Print user-friendly error to stderr
        error_console = Console(stderr=True, style="bold red")
        error_console.print(f"\nCRITICAL ERROR: {e}")
        error_console.print(
            "Please check your .env file for required API keys and "
            "valid platform credentials/URLs (like OPENROUTER_API_KEY)."
        )
        sys.exit(1) # Exit with error code
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        cancel_console = Console(stderr=True)
        cancel_console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        # Catch any other unexpected exceptions during setup or top-level run
        logging.getLogger('SocialOSINTLM').critical(
            "Unexpected critical error: %s", e, exc_info=True # Log traceback
        )
        error_console = Console(stderr=True, style="bold red")
        error_console.print(f"\nUNEXPECTED CRITICAL ERROR: {e}")
        sys.exit(1) # Exit with generic error code
import os
import sys
import json
import hashlib
import logging
import argparse
import threading
import httpx # Still used for media downloads and some platform fetches (e.g. HackerNews)
import tweepy
import praw
import prawcore
import shutil
import base64
from mastodon import (
    Mastodon,
    MastodonError,
    MastodonNotFoundError,
    MastodonRatelimitError,
    MastodonUnauthorizedError,
    MastodonVersionError,
)
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.text import Text # Added import
from urllib.parse import quote_plus, urlparse
from PIL import Image
from atproto import Client, exceptions as atproto_exceptions
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, cast
from functools import lru_cache

# OpenAI library imports
from openai import OpenAI, APIError, RateLimitError, AuthenticationError, BadRequestError, OpenAIError
from openai.types.chat import ChatCompletion # For type hinting completion object

load_dotenv()  # Load .env file if available

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("analyzer.log"), logging.StreamHandler()],
)
logger = logging.getLogger("SocialOSINTLM")

# Constants
CACHE_EXPIRY_HOURS = 24 # How long cache files are considered 'fresh'
MAX_CACHE_ITEMS = 200 # Max tweets/posts/submissions per user/platform in cache
REQUEST_TIMEOUT = 20.0 # Default timeout for HTTP requests
INITIAL_FETCH_LIMIT = 50 # How many items to fetch on first run or force_refresh
INCREMENTAL_FETCH_LIMIT = 50 # How many items to fetch during incremental updates
MASTODON_FETCH_LIMIT = 40 # Mastodon API max limit is often 40
SUPPORTED_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
] # Consolidated list for media analysis


class RateLimitExceededError(Exception): # Renamed for clarity if used internally
    """Custom exception for API rate limits."""
    pass


class UserNotFoundError(Exception):
    """Custom exception for when a user/profile cannot be found."""
    pass


class AccessForbiddenError(Exception):
    """Custom exception for access denied (e.g., private account)."""
    pass


# JSON Encoder
class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder to handle datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Helper Function (Moved to Global Scope)
def get_sort_key(item: Dict[str, Any], dt_key: str) -> datetime:
    """Safely gets and parses a datetime string or object/timestamp for sorting."""
    dt_val = item.get(dt_key)
    if isinstance(dt_val, str):
        try:
            # Handle ISO format strings (including 'Z' for UTC)
            if dt_val.endswith("Z"):
                dt_val = dt_val[:-1] + "+00:00"
            dt_obj = datetime.fromisoformat(dt_val)
            # Ensure timezone for comparison if needed (make it UTC if naive)
            return dt_obj if dt_obj.tzinfo else dt_obj.replace(tzinfo=timezone.utc)
        except ValueError:  # Handle cases where conversion might fail
            logger.warning(f"Could not parse datetime string: {dt_val}")
            return datetime.min.replace(tzinfo=timezone.utc) # Fallback to minimum time
    elif isinstance(dt_val, datetime):
        # Ensure timezone for comparison if needed (make it UTC if naive)
        return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    # Fallback for missing/invalid keys or other types (like timestamps)
    elif isinstance(dt_val, (int, float)):
        try:
            # Attempt to treat as UNIX timestamp
            dt_obj = datetime.fromtimestamp(dt_val, tz=timezone.utc)
            return dt_obj
        except (ValueError, OSError):  # OSError can occur for out-of-range timestamps
            logger.warning(f"Could not convert timestamp: {dt_val}")
            return datetime.min.replace(tzinfo=timezone.utc)

    logger.debug(
        f"Using fallback datetime for key '{dt_key}' with value type: {type(dt_val)}"
    )
    return datetime.min.replace(
        tzinfo=timezone.utc
    )  # Fallback for missing/invalid keys


# Main Class
class SocialOSINTLM:
    _llm_completion_object: Optional[ChatCompletion] = None
    _llm_api_exception: Optional[Exception] = None

    def __init__(self, args=None):
        self.console = Console()
        self.base_dir = Path("data")
        self._setup_directories()
        # self.progress is no longer a shared instance variable for all progress bars.
        # It will be instantiated locally where needed.
        self.args = args if args else argparse.Namespace()
        # Ensure 'offline' attribute exists, defaulting to False if not in args (e.g. direct instantiation)
        if not hasattr(self.args, 'offline'):
            self.args.offline = False # Default to online if 'offline' arg not present

        # Mastodon multi-instance attributes
        self.mastodon_config_file_path_str = os.getenv("MASTODON_CONFIG_FILE", "mastodon_instances.json")
        # self.mastodon_config_file_path will be resolved in _initialize_mastodon_clients
        # to check relative to base_dir and then as provided.
        self._mastodon_clients: Dict[str, Mastodon] = {} # Key: api_base_url, Value: Mastodon client
        self._default_mastodon_lookup_client: Optional[Mastodon] = None
        self._mastodon_clients_initialized: bool = False
            
        self._verify_env_vars() # This will now check LLM vars too

    def _verify_env_vars(self):
        # Core LLM settings
        required_llm = ["LLM_API_KEY", "LLM_API_BASE_URL", "IMAGE_ANALYSIS_MODEL", "ANALYSIS_MODEL"]
        missing_llm = [var for var in required_llm if not os.getenv(var)]
        if missing_llm:
            raise RuntimeError(
                f"Missing critical LLM-related environment variables: {', '.join(missing_llm)}. "
                "Please set LLM_API_KEY, LLM_API_BASE_URL, IMAGE_ANALYSIS_MODEL, and ANALYSIS_MODEL."
            )

        # Check for Mastodon configuration file existence first
        # self.mastodon_config_file_path_str is set in __init__
        path_relative_to_base_dir = self.base_dir / self.mastodon_config_file_path_str
        path_as_provided = Path(self.mastodon_config_file_path_str)
        
        mastodon_config_found = False
        if path_relative_to_base_dir.is_file():
            mastodon_config_found = True
            logger.info(f"Mastodon configuration file found at: {path_relative_to_base_dir}")
        elif path_as_provided.is_file():
            mastodon_config_found = True
            logger.info(f"Mastodon configuration file found at: {path_as_provided}")
        else:
            logger.debug(f"Mastodon configuration file '{self.mastodon_config_file_path_str}' not found. Mastodon availability depends on other platforms.")


        # Check for at least one platform credential set OR Mastodon config file
        platforms_configured = any(
            [
                all(os.getenv(k) for k in ["TWITTER_BEARER_TOKEN"]),
                all(
                    os.getenv(k)
                    for k in [
                        "REDDIT_CLIENT_ID",
                        "REDDIT_CLIENT_SECRET",
                        "REDDIT_USER_AGENT",
                    ]
                ),
                all(os.getenv(k) for k in ["BLUESKY_IDENTIFIER", "BLUESKY_APP_SECRET"]),
                mastodon_config_found, # Mastodon is considered "configured" if its JSON file exists
            ]
        )
        
        # HN needs no keys, considered configured if no other platform is.
        available_no_creds = self.get_available_platforms(check_creds=False) # Get all conceptually known platforms
        
        is_only_hackernews_conceptually = "hackernews" in available_no_creds and len(available_no_creds) == 1

        if not platforms_configured and not is_only_hackernews_conceptually:
            logger.warning(
                "No platform API credentials (Twitter, Reddit, Bluesky) found AND Mastodon configuration file is missing or not found."
                " Only HackerNews might work if it's the sole available platform concept. "
                "Please set credentials for at least one platform or provide a 'MASTODON_CONFIG_FILE' via environment variable (default: mastodon_instances.json)."
            )
        elif not platforms_configured and is_only_hackernews_conceptually:
            # This case means only HackerNews is conceptually available and nothing else is configured.
            pass # HackerNews alone is okay if no other platforms are configured

    def _setup_directories(self):
        """Ensures necessary directories exist."""
        for dir_name in ["cache", "media", "outputs"]:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def _handle_purge(self):
        """Handles the interactive purging of cached data."""
        self.console.print("\n[bold yellow]Select Data to Purge:[/bold yellow]")
        purge_options = {
            "1": (
                "All",
                [
                    self.base_dir / "cache",
                    self.base_dir / "media",
                    self.base_dir / "outputs",
                ],
            ),
            "2": ("Cache (Text/Metadata)", [self.base_dir / "cache"]),
            "3": ("Media Files", [self.base_dir / "media"]),
            "4": ("Output Reports", [self.base_dir / "outputs"]),
            "5": ("Cancel", []),
        }

        for key, (name, _) in purge_options.items():
            self.console.print(f" {key}. {name}")

        choice = Prompt.ask("Enter number", default="5").strip()

        if choice not in purge_options:
            self.console.print("[red]Invalid selection.[/red]")
            return

        selected_name, dirs_to_purge = purge_options[choice]

        if not dirs_to_purge: # Handle Cancel choice
            self.console.print("[cyan]Purge operation cancelled.[/cyan]")
            return

        dir_names = [d.name for d in dirs_to_purge]
        confirm_msg = f"This will PERMANENTLY delete all files in '{', '.join(dir_names)}'. Are you sure?"

        if Confirm.ask(f"[bold red]{confirm_msg}[/bold red]", default=False):
            self.console.print(f"[yellow]Purging {selected_name}...[/yellow]")
            success_count = 0
            fail_count = 0
            for dir_path in dirs_to_purge:
                try:
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
                        logger.info(f"Successfully purged directory: {dir_path}")
                        dir_path.mkdir(parents=True, exist_ok=True) # Recreate after purge
                        self.console.print(
                            f"[green]Successfully purged '{dir_path.name}'.[/green]"
                        )
                        success_count += 1
                    else:
                        self.console.print(
                            f"[yellow]Directory '{dir_path.name}' does not exist, skipping.[/yellow]"
                        )
                        dir_path.mkdir(parents=True, exist_ok=True) # Ensure it exists for future use
                        success_count += (
                            1 # Still a "success" in terms of desired state (dir exists and is empty)
                        )

                except OSError as e:
                    logger.error(
                        f"Failed to purge directory {dir_path}: {e}", exc_info=True
                    )
                    self.console.print(
                        f"[bold red]Error purging '{dir_path.name}': {e}[/bold red]"
                    )
                    fail_count += 1
                except Exception as e: # Catch any other unexpected errors
                    logger.error(
                        f"Unexpected error purging directory {dir_path}: {e}",
                        exc_info=True,
                    )
                    self.console.print(
                        f"[bold red]Unexpected error purging '{dir_path.name}': {e}[/bold red]"
                    )
                    fail_count += 1

            if fail_count > 0:
                self.console.print(
                    f"[yellow]Purge operation completed with {fail_count} error(s). Check logs.[/yellow]"
                )
            elif success_count > 0: # Only print full success if something was actually done
                self.console.print(
                    f"[green]Purge operation completed successfully.[/green]"
                )
            # else: No message if nothing was purged and no errors (e.g., all dirs already empty/missing)

        else:
            self.console.print("[cyan]Purge operation cancelled.[/cyan]")
        self.console.print("") # Add a newline for spacing


    @property
    def llm_client(self) -> OpenAI:
        """Initializes and returns the OpenAI client for LLM calls."""
        if not hasattr(self, "_llm_client_instance"):
            try:
                api_key = os.environ.get("LLM_API_KEY")
                base_url = os.environ.get("LLM_API_BASE_URL")

                if not api_key:
                    raise ValueError(
                        "LLM_API_KEY not found in environment variables. This is required."
                    )
                if not base_url:
                    raise ValueError(
                        "LLM_API_BASE_URL not found in environment variables. This is required (e.g., https://openrouter.ai/api/v1 or http://localhost:8000/v1)."
                    )

                current_default_headers: Dict[str, str] = {}
                # Conditionally add OpenRouter specific headers if configured
                if "openrouter.ai" in base_url.lower():
                    referer = os.getenv("OPENROUTER_REFERER", "http://localhost:3000") # Default if not set
                    x_title = os.getenv("OPENROUTER_X_TITLE", "SocialOSINTLM") # Default if not set
                    current_default_headers["HTTP-Referer"] = referer
                    current_default_headers["X-Title"] = x_title
                    # Content-Type is handled by the openai library

                self._llm_client_instance = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=httpx.Timeout(60.0, connect=10.0), # Standard timeout object
                    default_headers=current_default_headers if current_default_headers else None,
                )
                logger.info(f"LLM client initialized for base URL: {base_url}")
            except ValueError as e: # For missing env vars
                raise RuntimeError(f"LLM client configuration error: {e}")
            except Exception as e: # Catch-all for other init errors
                raise RuntimeError(f"Failed to initialize LLM client: {e}")
        return self._llm_client_instance

    @property
    def bluesky(self) -> Client:
        """Initializes and returns the Bluesky client."""
        if not hasattr(self, "_bluesky_client"):
            try:
                if not os.getenv("BLUESKY_IDENTIFIER") or not os.getenv(
                    "BLUESKY_APP_SECRET"
                ):
                    raise RuntimeError(
                        "Bluesky credentials (BLUESKY_IDENTIFIER, BLUESKY_APP_SECRET) not set in environment."
                    )
                client = Client()
                # In offline mode, we might not want to attempt login if it makes network calls
                # However, the atproto Client itself might need a session for some internal logic
                # or for constructing URLs correctly, even if not fetching.
                # Login is essential for media downloads even if URLs are public, sometimes they need session.
                if not self.args.offline: # Attempt login only if not in offline mode.
                    try:
                        # Login to get session token needed for some operations like media download
                        login_response = client.login(
                            os.environ["BLUESKY_IDENTIFIER"],
                            os.environ["BLUESKY_APP_SECRET"],
                        )
                        logger.debug(
                            f"Bluesky login successful for handle: {login_response.handle}"
                        )
                    except atproto_exceptions.AtProtocolError as login_err:
                        logger.error(f"Bluesky login failed: {login_err}")
                        if "invalid identifier or password" in str(login_err).lower():
                            raise RuntimeError(
                                "Bluesky login failed: Invalid identifier or password."
                            )
                        else:
                            raise RuntimeError(f"Bluesky login failed: {login_err}")
                else:
                    logger.info("Offline mode: Bluesky client instantiated without login attempt. Media downloads might fail if tokens are required and not cached.")


                self._bluesky_client = client

            except (KeyError, RuntimeError) as e:
                logger.error(f"Bluesky setup failed: {e}")
                raise RuntimeError(
                    f"Bluesky setup failed: {e}"
                ) # Re-raise as RuntimeError for caller to handle
        return self._bluesky_client

    # Method to extract instance domain (NEW)
    def _get_instance_domain_from_acct(self, acct: str) -> Optional[str]:
        """Extracts the instance domain from a 'user@instance.domain' string."""
        if "@" in acct:
            parts = acct.split("@", 1)
            if len(parts) == 2 and "." in parts[1] and parts[1].strip():
                # Further check to ensure the domain part is not just "." or empty
                domain_part = parts[1].strip()
                if domain_part and domain_part != ".":
                    return domain_part
        logger.debug(f"Could not extract valid instance domain from account string: '{acct}'")
        return None

    # Method to initialize Mastodon clients from config (NEW)
    def _initialize_mastodon_clients(self):
        """
        Initializes Mastodon clients from the JSON configuration file.
        Populates self._mastodon_clients and self._default_mastodon_lookup_client.
        """
        if self._mastodon_clients_initialized:
            logger.debug("Mastodon clients already attempted initialization.")
            return

        # Determine the correct path for the Mastodon config file
        # Option 1: Path relative to self.base_dir
        path_relative_to_base_dir = self.base_dir / self.mastodon_config_file_path_str
        # Option 2: Path as directly provided (could be absolute or relative to CWD)
        path_as_provided = Path(self.mastodon_config_file_path_str)

        actual_config_path: Optional[Path] = None

        if path_relative_to_base_dir.is_file():
            actual_config_path = path_relative_to_base_dir
            logger.debug(f"Found Mastodon config file at: {actual_config_path} (relative to base_dir)")
        elif path_as_provided.is_file():
            actual_config_path = path_as_provided
            logger.debug(f"Found Mastodon config file at: {actual_config_path} (as provided)")
        else:
            logger.warning(
                f"Mastodon config file '{self.mastodon_config_file_path_str}' not found at "
                f"'{path_relative_to_base_dir}' (relative to data dir) or "
                f"'{path_as_provided}' (as direct path). Mastodon functionality will be unavailable."
            )
            self._mastodon_clients_initialized = True # Mark as "initialized" (i.e., attempt made)
            return

        try:
            with open(actual_config_path, "r", encoding="utf-8") as f:
                instances_config = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding Mastodon config file '{actual_config_path}': {e}. Mastodon functionality impaired.")
            self._mastodon_clients_initialized = True
            return
        except OSError as e:
            logger.error(f"Error reading Mastodon config file '{actual_config_path}': {e}. Mastodon functionality impaired.")
            self._mastodon_clients_initialized = True
            return

        if not isinstance(instances_config, list):
            logger.error(f"Mastodon config file '{actual_config_path}' must contain a JSON list of instance objects. Mastodon functionality impaired.")
            self._mastodon_clients_initialized = True
            return

        default_candidates = []
        successful_clients_count = 0

        for instance_conf in instances_config:
            if not isinstance(instance_conf, dict):
                logger.warning(f"Skipping invalid entry in Mastodon config (not a dictionary): {str(instance_conf)[:100]}")
                continue

            api_base_url = instance_conf.get("api_base_url")
            access_token = instance_conf.get("access_token")
            instance_name = instance_conf.get("name", api_base_url) # Default name to URL if not provided

            if not api_base_url or not access_token:
                logger.warning(f"Skipping incomplete Mastodon instance config for '{instance_name}': missing api_base_url or access_token.")
                continue

            parsed_url = urlparse(api_base_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Skipping Mastodon instance '{instance_name}' due to invalid api_base_url format: {api_base_url}. Should be like 'https://mastodon.social'.")
                continue
            
            normalized_api_base_url = api_base_url.rstrip('/')


            try:
                client = Mastodon(
                    access_token=access_token,
                    api_base_url=normalized_api_base_url,
                    request_timeout=REQUEST_TIMEOUT,
                )

                if not self.args.offline:
                    try:
                        instance_info = client.instance()
                        fetched_instance_title = instance_info.get("title", "N/A")
                        logger.debug(f"Successfully connected to Mastodon instance: {fetched_instance_title} ({client.api_base_url}) for config entry '{instance_name}'")
                    except MastodonError as instance_err:
                        logger.error(f"Failed to connect/verify Mastodon instance {client.api_base_url} for '{instance_name}': {instance_err}")
                        if "unauthorized" in str(instance_err).lower() or "401" in str(instance_err):
                            logger.error(f" -> Suggestion: Check Access Token for {client.api_base_url}.")
                        elif "not found" in str(instance_err).lower() or "404" in str(instance_err):
                             logger.error(f" -> Suggestion: API base URL {client.api_base_url} seems incorrect.")
                        # Do not add this client if connectivity check fails in online mode
                        continue 
                else:
                    logger.info(f"Offline mode: Mastodon client for '{instance_name}' ({client.api_base_url}) initialized without connectivity check.")

                self._mastodon_clients[normalized_api_base_url] = client
                successful_clients_count += 1
                if instance_conf.get("is_default_lookup_instance", False) is True:
                    default_candidates.append(client)

            except Exception as e: # Catch any error during Mastodon() instantiation
                logger.error(f"Error initializing Mastodon client object for '{instance_name}' ({api_base_url}): {e}")

        if default_candidates:
            if len(default_candidates) > 1:
                logger.warning(f"Multiple Mastodon instances marked as 'is_default_lookup_instance' in '{actual_config_path}'. Using the first one found: {default_candidates[0].api_base_url}")
            self._default_mastodon_lookup_client = default_candidates[0]
            logger.info(f"Default Mastodon lookup client set to: {self._default_mastodon_lookup_client.api_base_url}")
        elif self._mastodon_clients: # No explicit default, pick first one available from successfully initialized clients
            self._default_mastodon_lookup_client = next(iter(self._mastodon_clients.values()))
            logger.info(f"No explicit default Mastodon lookup instance found. Using first successfully initialized client as default: {self._default_mastodon_lookup_client.api_base_url}")


        if not self._mastodon_clients:
            logger.warning(f"No Mastodon clients were successfully initialized from the configuration file: {actual_config_path}")
        else:
            logger.info(f"Successfully initialized {successful_clients_count} Mastodon client(s) from {actual_config_path}.")

        self._mastodon_clients_initialized = True

    # Method to get a specific or default Mastodon client (NEW)
    def _get_mastodon_client(self, target_user_acct: Optional[str] = None) -> Optional[Mastodon]:
        """
        Gets an appropriate Mastodon client.
        If target_user_acct (user@instance.domain) is provided, tries to find a specific client.
        Otherwise, returns the default lookup client.
        Initializes clients from config if not already done.
        """
        if not self._mastodon_clients_initialized:
            self._initialize_mastodon_clients() # Ensure clients are loaded

        # If clients still not initialized after attempt (e.g. config file error), bail.
        if not self._mastodon_clients_initialized: # Should not be strictly necessary due to above call, but defensive
            logger.error("Mastodon client initialization was not completed. Cannot get client.")
            return None


        if target_user_acct:
            instance_domain = self._get_instance_domain_from_acct(target_user_acct)
            if instance_domain:
                # Try to find a client matching this instance_domain
                # Common schemes are https and http. Check both for api_base_url matching.
                # The keys in self._mastodon_clients are already normalized (e.g. https://domain.tld)
                potential_urls_to_check = [
                    f"https://{instance_domain}",
                    f"http://{instance_domain}" # Less common for API but possible
                ]
                for p_url in potential_urls_to_check:
                    normalized_p_url = p_url.rstrip('/')
                    if normalized_p_url in self._mastodon_clients:
                        logger.debug(f"Using specific Mastodon client for instance '{normalized_p_url}' to handle '{target_user_acct}'.")
                        return self._mastodon_clients[normalized_p_url]
                logger.debug(f"No specifically configured client for instance '{instance_domain}' of user '{target_user_acct}'. Will use default lookup client if available.")

        if self._default_mastodon_lookup_client:
            logger.debug(f"Using default Mastodon lookup client: {self._default_mastodon_lookup_client.api_base_url}")
            return self._default_mastodon_lookup_client
        
        # Fallback: if no specific client for target_user_acct AND no default_lookup_client was set,
        # but there ARE some clients initialized, use the first one available.
        # This situation implies a config where no instance was marked as default,
        # and the target user's instance wasn't directly configured.
        if self._mastodon_clients: # pragma: no cover (should ideally be handled by default logic in _initialize_mastodon_clients)
            first_available_client = next(iter(self._mastodon_clients.values()))
            logger.warning(
                f"No specific client for '{target_user_acct}' and no default lookup client was designated. "
                f"Falling back to the first available Mastodon client: {first_available_client.api_base_url}"
            )
            return first_available_client

        logger.warning(f"No Mastodon clients are available (neither specific for '{target_user_acct}' nor a default). Cannot proceed with Mastodon operations.")
        return None

    @property
    def reddit(self) -> praw.Reddit:
        """Initializes and returns the Reddit client (PRAW)."""
        if not hasattr(self, "_reddit"):
            try:
                if not all(
                    os.getenv(k)
                    for k in [
                        "REDDIT_CLIENT_ID",
                        "REDDIT_CLIENT_SECRET",
                        "REDDIT_USER_AGENT",
                    ]
                ):
                    raise RuntimeError(
                        "Reddit credentials (ID, Secret, User-Agent) not fully set in environment."
                    )
                self._reddit = praw.Reddit(
                    client_id=os.environ["REDDIT_CLIENT_ID"],
                    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                    user_agent=os.environ["REDDIT_USER_AGENT"],
                    read_only=True, # We only need read access
                )
                # PRAW might do some lazy checks or basic validation on init.
                # In offline mode, we want to avoid network calls.
                # However, basic instantiation should be fine. PRAW's `read_only`
                # and subsequent calls like `redditor.id` are what trigger network.
                if not self.args.offline:
                    is_read_only = self._reddit.read_only
                    logger.debug(f"Reddit client initialized (Read-Only: {is_read_only}).")
                    # This check `self._reddit.user.me()` requires non-read-only mode and network.
                    # We are in read_only, so this specific check is not applicable.
                    # if (
                    #     not is_read_only and self._reddit.user.me() is None
                    # ): # pragma: no cover (depends on live API)
                    #     logger.warning(
                    #         "Reddit client may not be properly authenticated even though no error was raised."
                    #     )
                else:
                    logger.info("Offline mode: Reddit client (PRAW) instantiated.")


            except (
                KeyError,
                prawcore.exceptions.OAuthException,
                prawcore.exceptions.ResponseException,
                RuntimeError,
            ) as e:
                err_msg = str(e)
                if (
                    "401" in err_msg
                    or "invalid_client" in err_msg
                    or "unauthorized" in err_msg
                ):
                    raise RuntimeError(
                        f"Reddit authentication failed: Check Client ID/Secret. ({e})"
                    )
                else:
                    logger.error(f"Reddit setup failed: {e}")
                    raise RuntimeError(f"Reddit setup failed: {e}")
        return self._reddit

    @property
    def twitter(self) -> tweepy.Client:
        """Initializes and returns the Twitter client (tweepy)."""
        if not hasattr(self, "_twitter"):
            try:
                token = os.getenv("TWITTER_BEARER_TOKEN")
                if not token:
                    raise RuntimeError(
                        "Twitter Bearer Token (TWITTER_BEARER_TOKEN) not set."
                    )
                self._twitter = tweepy.Client(
                    bearer_token=token, wait_on_rate_limit=False
                )
                # Skip connection test in offline mode
                if not self.args.offline:
                    # Test connection by fetching a known, public user
                    # This raises Unauthorized or TweepyException if token is bad or API down
                    self._twitter.get_user(
                        username="twitterdev", user_fields=["id"]
                    ) # pragma: no cover (depends on live API)
                    logger.debug("Twitter client initialized and connection tested.")
                else:
                    logger.info("Offline mode: Twitter client initialized without connection test.")
            except (KeyError, RuntimeError) as e:
                logger.error(f"Twitter setup failed: {e}")
                raise RuntimeError(f"Twitter setup failed: {e}")
            except tweepy.errors.Unauthorized as e: # pragma: no cover
                logger.error(
                    f"Twitter authentication failed (Unauthorized): Check Bearer Token. {e}"
                )
                raise RuntimeError(
                    f"Twitter authentication failed: Invalid Bearer Token? ({e})"
                )
            except tweepy.errors.TweepyException as e: # pragma: no cover
                logger.error(f"Twitter API error during initialization: {e}")
                raise RuntimeError(f"Twitter client initialization failed: {e}")
        return self._twitter

    def _handle_rate_limit(self, platform_context: str, exception: Optional[Exception] = None):
        """Handles rate limit exceptions by logging and raising a standard error."""
        error_message = f"{platform_context} API rate limit exceeded."
        reset_info = ""
        wait_seconds = 900  # Default 15 minutes

        if isinstance(exception, tweepy.TooManyRequests): # pragma: no cover
            rate_limit_reset = exception.response.headers.get("x-rate-limit-reset")
            if rate_limit_reset:
                try:
                    reset_timestamp = int(rate_limit_reset)
                    reset_time = datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)
                    current_time = datetime.now(timezone.utc)
                    wait_seconds = max(int((reset_time - current_time).total_seconds()) + 5, 1) # Add 5s buffer
                    reset_info = f"Try again after: {reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ({wait_seconds}s)"
                except (ValueError, TypeError):
                    logger.warning("Could not parse Twitter rate limit reset time header.")
                    reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
            else:
                reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying (reset header missing)."
        elif (isinstance(exception, (prawcore.exceptions.RequestException, httpx.HTTPStatusError))
              and hasattr(exception, "response") and exception.response is not None
              and exception.response.status_code == 429): # pragma: no cover
            # Reddit and general HTTP 429 may use Retry-After or x-ratelimit-reset
            retry_after = exception.response.headers.get("Retry-After") or exception.response.headers.get("x-ratelimit-reset")
            if retry_after and retry_after.isdigit():
                wait_seconds = int(retry_after) + 5
                reset_info = f"Try again in {wait_seconds} seconds."
            else:
                reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
        elif isinstance(exception, MastodonRatelimitError): # pragma: no cover
            reset_header = getattr(exception, "headers", {}).get("X-RateLimit-Reset")
            if reset_header:
                try:
                    reset_time = datetime.fromisoformat(reset_header.replace("Z", "+00:00"))
                    current_time = datetime.now(timezone.utc)
                    wait_seconds = max(int((reset_time - current_time).total_seconds()) + 5, 1)
                    reset_info = f"Try again after: {reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ({wait_seconds}s)"
                except ValueError:
                    logger.warning("Could not parse Mastodon rate limit reset time header.")
                    reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
            else:
                reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
        elif isinstance(exception, RateLimitError): # Handles LLM rate limits (from openai library)
            error_message = f"LLM API ({platform_context}) rate limit exceeded."
            wait_seconds = 60 # Default 1 min for LLM if no specific header
            reset_info = f"Wait ~{wait_seconds // 60} minute(s) before retrying."
            
            # OpenAI library's RateLimitError might have response details
            if hasattr(exception, 'response') and exception.response:
                headers = exception.response.headers
                # Standard 'Retry-After' header (seconds or HTTP-date)
                retry_after_header = headers.get("retry-after") or headers.get("Retry-After")
                # OpenRouter specific rate limit headers (if LLM_API_BASE_URL points there)
                openrouter_reset_s = headers.get("x-ratelimit-reset") # float seconds until reset
                openrouter_retry_after_ms = headers.get("x-ratelimit-retry-after-ms") # int ms to wait

                if openrouter_retry_after_ms and openrouter_retry_after_ms.isdigit():
                    wait_seconds = (int(openrouter_retry_after_ms) // 1000) + 2 # ms to s, +2s buffer
                    reset_info = f"Try again in {wait_seconds} seconds (from x-ratelimit-retry-after-ms)."
                elif openrouter_reset_s:
                    try:
                        current_time_s = datetime.now(timezone.utc).timestamp()
                        reset_timestamp_s = float(openrouter_reset_s)
                        wait_seconds = max(int(reset_timestamp_s - current_time_s) + 5, 1) # +5s buffer
                        reset_time_dt = datetime.fromtimestamp(reset_timestamp_s, tz=timezone.utc)
                        reset_info = f"Try again after: {reset_time_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} ({wait_seconds}s from x-ratelimit-reset)."
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse OpenRouter x-ratelimit-reset: {openrouter_reset_s}")
                elif retry_after_header:
                    if retry_after_header.isdigit():
                        wait_seconds = int(retry_after_header) + 5
                        reset_info = f"Try again in {wait_seconds} seconds (from Retry-After header)."
                    else: # HTTP-date format
                        try:
                            reset_time_dt = datetime.strptime(retry_after_header, '%a, %d %b %Y %H:%M:%S GMT').replace(tzinfo=timezone.utc)
                            current_time = datetime.now(timezone.utc)
                            wait_seconds = max(int((reset_time_dt - current_time).total_seconds()) + 5, 1)
                            reset_info = f"Try again after: {reset_time_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} ({wait_seconds}s from Retry-After date)."
                        except ValueError:
                            logger.warning(f"Could not parse LLM rate limit Retry-After date header: {retry_after_header}")
            else: # Fallback if no response details on the exception object
                logger.debug("OpenAI RateLimitError did not have detailed response headers for retry timing.")
        elif (isinstance(exception, atproto_exceptions.AtProtocolError) and "rate limit" in str(exception).lower()): # pragma: no cover (hard to trigger)
            reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
            try:
                if hasattr(exception, "response") and exception.response:
                    err_data = json.loads(exception.response.content)
                    if err_data.get("error") == "RateLimitExceeded": pass # Bluesky specific check
            except (json.JSONDecodeError, AttributeError): pass # Stick to default
        else: # For other httpx 429s (e.g., media download)
            if (isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code == 429):
                retry_after_header = exception.response.headers.get("Retry-After")
                if retry_after_header and retry_after_header.isdigit():
                    wait_seconds = int(retry_after_header) + 5
                    reset_info = f"Try again in {wait_seconds} seconds."
                else:
                    reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."
                error_message = f"Media Download or other HTTP ({platform_context}) API rate limit exceeded."
            else:
                reset_info = f"Wait ~{wait_seconds // 60} minutes before retrying."

        self.console.print(
            Panel(
                f"[bold red]Rate Limit Blocked: {platform_context}[/bold red]\n"
                f"{error_message}\n"
                f"{reset_info}",
                title="🚫 Rate Limit",
                border_style="red",
            )
        )
        raise RateLimitExceededError(error_message + f" ({reset_info})") # Reraise for upstream handling

    def _get_media_path(self, url: str, platform: str, username: str) -> Path:
        """Generates a consistent local path for downloaded media."""
        # Username is not strictly needed here for hash, but kept for consistency
        # Using only URL hash for broader caching if same media is on different user profiles
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return (
            self.base_dir / "media" / f"{url_hash}.media"
        ) # Stub, extension added after download

    def _download_media(self, url: str, platform: str, username: str, headers: Optional[dict] = None) -> Optional[Path]: # pragma: no cover
        """Downloads a media file from a URL and caches it."""
        media_path_stub = self._get_media_path(url, platform, username)

        # Check if file with any valid extension exists
        existing_files = list(
            (self.base_dir / "media").glob(f"{media_path_stub.stem}.*")
        )
        if existing_files:
            # Prioritize common image/video formats if multiple versions exist (unlikely with current naming)
            preferred_exts = [".jpg", ".png", ".webp", ".gif", ".mp4", ".webm"] # etc.
            for ext in preferred_exts:
                found_path = self.base_dir / "media" / f"{media_path_stub.stem}{ext}"
                if found_path.exists() and found_path in existing_files:
                    logger.debug(f"Media cache hit (preferred ext): {found_path}")
                    return found_path
            # Fallback to first found if no preferred match (e.g. if .media was an old cache)
            logger.debug(f"Media cache hit (generic): {existing_files[0]}")
            return existing_files[0]

        # If not in cache and in offline mode, don't attempt to download
        if self.args.offline:
            logger.warning(f"Offline mode: Media {url} not in local cache. Skipping download.")
            return None

        # Define valid media types and their extensions
        valid_types = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
            # Add other types as needed
        }
        final_media_path = None

        try:
            # Prepare authentication headers if needed by platform for media
            auth_headers = {}
            try:
                if platform == "twitter": # Twitter media URLs might need auth sometimes
                    if not hasattr(self, "_twitter"): self.twitter # Ensure client exists
                    # Twitter's new API media might be directly accessible via CDN without bearer,
                    # but some older or specific media might need it.
                    # tweepy Client might handle this internally for its objects, but for direct URLs:
                    token = os.getenv("TWITTER_BEARER_TOKEN")
                    if token: auth_headers["Authorization"] = f"Bearer {token}"
                elif platform == "bluesky": # Bluesky CDN URLs might need auth
                    if not hasattr(self, "_bluesky_client"): self.bluesky
                    # Accessing the session token might be internal to atproto Client.
                    # For bsky.app/cdn.bsky.app, auth is sometimes needed.
                    # This is a simplified attempt; true auth might require more.
                    access_token = getattr(self.bluesky._session, "access_jwt", None) # type: ignore
                    if not access_token:
                        # Try to refresh/ensure login if token is missing?
                        # For now, just log a warning if it's not immediately available.
                        logger.warning(f"Bluesky access token not available for media download for URL: {url}. This might be okay for public CDNs.")
                    else:
                        auth_headers["Authorization"] = f"Bearer {access_token}"
                # Note: Mastodon media is generally public via CDN links, specific auth not typically added here.
                # If a Mastodon instance required auth for media and `_get_mastodon_client` returned a client for that specific instance,
                # we *could* try to use its `access_token` here, but it's complex as `_download_media` doesn't know which client to use.
                # For now, relies on public accessibility or generic user-agent.

            except RuntimeError as client_init_err:
                logger.warning(f"Cannot add auth headers for {platform} media download, client init failed: {client_init_err}")


            request_headers = headers.copy() if headers else {}
            request_headers.update(auth_headers)
            request_headers.setdefault("User-Agent", "SocialOSINTLM/1.0") # Generic UA

            with httpx.Client(follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
                resp = client.get(url, headers=request_headers)
                resp.raise_for_status() # Raise HTTPStatusError for 4xx/5xx responses

            content_type = resp.headers.get("content-type", "").lower().split(";")[0].strip()
            extension = valid_types.get(content_type)

            if not extension:
                # Try to guess from URL if content-type is generic (e.g. application/octet-stream)
                parsed_url_path = urlparse(url).path # Changed variable name
                path_ext = Path(parsed_url_path).suffix.lower()
                if path_ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".webm"]:
                    extension = path_ext
                    logger.debug(f"Guessed extension '{extension}' from URL for content type '{content_type}'.")
                else:
                    content_preview = resp.content[:64] # Log first 64 bytes for inspection
                    logger.warning(f"Unsupported or non-media type '{content_type}' for URL: {url}. Content preview (bytes): {content_preview}")
                    return None # Not a recognized media type we handle

            final_media_path = media_path_stub.with_suffix(extension)
            final_media_path.write_bytes(resp.content)
            logger.debug(f"Downloaded media to: {final_media_path}")
            return final_media_path

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._handle_rate_limit(f"{platform} Media Download", e)
            elif e.response.status_code in [404, 410]: # Not Found or Gone
                logger.warning(f"Media not found ({e.response.status_code}) for {url}. Skipping.")
            elif e.response.status_code in [403, 401]: # Forbidden or Unauthorized
                logger.warning(f"Media access forbidden/unauthorized ({e.response.status_code}) for {url}. Skipping.")
            else:
                logger.error(f"HTTP error {e.response.status_code} downloading {url}: {e}. Response: {e.response.text[:200]}")
            return None
        except httpx.RequestError as e: # Network errors
            logger.error(f"Network error downloading {url}: {e}")
            return None
        except Exception as e: # Catch-all for unexpected issues
            # Log minimally to avoid huge logs for common errors like bad certs on obscure instances
            logger.error(f"Media download failed unexpectedly for {url}: {str(e)}", exc_info=False)
            return None
  
    def _analyze_image(self, file_path: Path, context: str = "") -> Optional[str]: # pragma: no cover
        """Analyzes an image file using a vision-capable LLM."""
        # In offline mode, new image analysis is not performed.
        # Relies on previously cached `media_analysis` strings.
        if self.args.offline:
            logger.info(f"Offline mode: Skipping LLM image analysis for {file_path}.")
            return None 

        if not file_path or not file_path.exists():
            logger.warning(f"Image analysis skipped: file path invalid or missing ({file_path})")
            return None
        if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.debug(f"Skipping analysis for non-image file: {file_path}")
            return None

        temp_path = None # Path for temporary processed image
        analysis_file_path = file_path # Start with original path
        original_format = None

        try:
            with Image.open(file_path) as img:
                original_format = img.format.lower() if img.format else None
                # Double check format, PIL can open more but vision models are picky
                if original_format not in ["jpeg", "png", "webp", "gif"]: # Common vision model supported types
                    logger.warning(f"Unsupported image type detected by PIL: {original_format} at {file_path} for vision analysis.")
                    return None # Not suitable for typical vision models

                # Preprocessing for Vision Model
                # Max dimension (e.g., for Gemini Pro Vision, common is 1024-2048)
                # Using 1536 as a general large size, helps with detail.
                max_dimension = 1536
                target_format = "JPEG" # Most vision models prefer JPEG or PNG

                needs_resizing = max(img.size) > max_dimension
                needs_conversion = original_format != "jpeg" # If not already JPEG
                is_animated = getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1

                img_to_process = img
                if is_animated: # Use first frame of animated images (GIF, animated WebP)
                    img.seek(0) # Go to the first frame
                    img_to_process = img.copy() # Work on a copy of the first frame
                    logger.debug(f"Using first frame of animated image: {file_path}")
                    needs_conversion = True # Will need to save as non-animated

                # Handle color modes (convert to RGB if necessary)
                if img_to_process.mode != "RGB":
                    if img_to_process.mode == "P" and "transparency" in img_to_process.info:
                        # Palette mode with transparency, convert to RGBA first then RGB
                        img_to_process = img_to_process.convert("RGBA")
                    if img_to_process.mode == "RGBA":
                        # Create a white background and paste RGBA image onto it
                        bg = Image.new("RGB", img_to_process.size, (255, 255, 255))
                        bg.paste(img_to_process, mask=img_to_process.split()[3]) # 3 is the alpha channel
                        img_to_process = bg
                        logger.debug(f"Converted RGBA image to RGB with white background: {file_path}")
                    else: # For other modes like L, CMYK, etc.
                        img_to_process = img_to_process.convert("RGB")
                        logger.debug(f"Converted image mode {img.mode} to RGB: {file_path}")
                    needs_conversion = True


                if needs_resizing:
                    scale_factor = max_dimension / max(img_to_process.size)
                    new_size = (int(img_to_process.size[0] * scale_factor), int(img_to_process.size[1] * scale_factor))
                    img_to_process = img_to_process.resize(new_size, Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image to {new_size}: {file_path}")
                    needs_conversion = True # Resizing implies we should re-save

                if needs_conversion: # If any change was made or format isn't JPEG
                    temp_suffix = f".processed.{target_format.lower()}"
                    temp_path = file_path.with_suffix(temp_suffix)
                    img_to_process.save(temp_path, target_format, quality=85) # Save as JPEG with reasonable quality
                    analysis_file_path = temp_path # Use this processed file for analysis
                    logger.debug(f"Saved processed image for analysis: {analysis_file_path}")
                else: # Original is fine
                    analysis_file_path = file_path
                    logger.debug(f"Using original image file for analysis: {analysis_file_path}")


            # Read the (potentially processed) image file and encode to base64
            base64_image = base64.b64encode(analysis_file_path.read_bytes()).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{base64_image}" # Assuming JPEG after processing

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
            model_to_use = os.getenv("IMAGE_ANALYSIS_MODEL")
            if not model_to_use:
                logger.error("IMAGE_ANALYSIS_MODEL environment variable not set. Cannot analyze image.")
                return None

            logger.debug(f"Sending image analysis request for {file_path} to model {model_to_use} via {self.llm_client.base_url}") # type: ignore
            
            # Use the OpenAI client
            completion = self.llm_client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}},
                    ]}
                ],
                max_tokens=1024, # Reasonable token limit for description
            )
            if not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
                logger.error(f"Invalid image analysis API response structure for {file_path}: {completion.model_dump_json(indent=2)}")
                return None

            analysis_text = completion.choices[0].message.content
            logger.debug(f"Image analysis successful for: {file_path}")
            return analysis_text.strip()

        except (IOError, Image.DecompressionBombError, SyntaxError, ValueError) as img_err: # PIL errors
            logger.error(f"Image processing error for {file_path}: {str(img_err)}")
            return None
        except APIError as api_err: # Generic OpenAI API error
            model_used = os.getenv("IMAGE_ANALYSIS_MODEL")
            err_message = str(api_err)
            status_code = api_err.status_code if hasattr(api_err, 'status_code') else "N/A"
            logger.error(f"LLM API error during image analysis ({model_used}). Status: {status_code}. Error: {err_message}")
            if hasattr(api_err, 'response') and api_err.response and hasattr(api_err.response, 'text'):
                logger.error(f"API Response Snippet: {api_err.response.text[:500]}")

            if isinstance(api_err, RateLimitError):
                self._handle_rate_limit(f"LLM Image Analysis ({model_used})", api_err) # This will re-raise
            elif isinstance(api_err, AuthenticationError):
                logger.error(f"LLM API Authentication Error (401) for model {model_used}. Check your LLM_API_KEY.")
            elif isinstance(api_err, BadRequestError):
                logger.error(f"LLM API Bad Request (400) for model {model_used}. Often due to invalid input.")
            return None # For other API errors, just log and return None
        except Exception as e:
            logger.error(f"Unexpected error during image analysis for {file_path}: {str(e)}", exc_info=True)
            return None
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Deleted temporary analysis file: {temp_path}")
                except OSError as e: # pragma: no cover
                    logger.warning(f"Could not delete temporary analysis file {temp_path}: {e}")

    @lru_cache(maxsize=128) # Cache path generation
    def _get_cache_path(self, platform: str, username: str) -> Path:
        """Generates a consistent local path for cache files."""
        # Sanitize username for filesystem, allowing common chars like @ . - _
        safe_username = "".join(c if c.isalnum() or c in ["-", "_", ".", "@"] else "_" for c in username)
        safe_username = safe_username[:100] # Limit length
        return self.base_dir / "cache" / f"{platform}_{safe_username}.json"

    def _load_cache(self, platform: str, username: str) -> Optional[Dict[str, Any]]:
        """Loads data from a user's cache file if fresh or if in offline mode."""
        cache_path = self._get_cache_path(platform, username)
        if not cache_path.exists():
            logger.debug(f"Cache miss (file not found): {cache_path}")
            return None
        try:
            logger.debug(f"Attempting to load cache: {cache_path}")
            data = json.loads(cache_path.read_text(encoding="utf-8"))

            # Validate basic cache structure: timestamp must exist
            if "timestamp" not in data:
                logger.warning(f"Cache file for {platform}/{username} is missing timestamp. Discarding.")
                cache_path.unlink(missing_ok=True)
                return None

            # Parse timestamp, ensure it's timezone-aware (UTC)
            timestamp = datetime.min.replace(tzinfo=timezone.utc) # Default to min time if parsing fails
            try:
                timestamp_str = data["timestamp"]
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str)
                elif isinstance(timestamp_str, datetime): # Should not happen with JSON load, but defensive
                    timestamp = timestamp_str # pragma: no cover
                else:
                    raise ValueError("Invalid timestamp format in cache")

                if timestamp.tzinfo is None: # Ensure timezone for comparison
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError) as ts_err:
                logger.warning(f"Failed to parse timestamp in cache for {platform}/{username}: {ts_err}. Discarding cache.")
                cache_path.unlink(missing_ok=True)
                return None

            # Validate essential keys per platform for a "valid" cache structure
            required_keys = ["timestamp"] # Universal
            if platform == "mastodon": required_keys.extend(["posts", "user_info", "stats"])
            elif platform == "twitter": required_keys.extend(["tweets", "user_info"]) # No separate stats block for twitter
            elif platform == "reddit": required_keys.extend(["submissions", "comments", "stats"]) # user_profile is inside stats often
            elif platform == "bluesky": required_keys.extend(["posts", "stats"]) # profile_info is key
            elif platform == "hackernews": required_keys.extend(["items", "stats"]) # Modern key is 'items'
            
            # Handle legacy 'submissions' key for HackerNews cache during validation
            if platform == "hackernews" and "submissions" not in data and "items" in data: # Cache already uses "items"
                pass # `required_keys` already covers "items"
            elif platform == "hackernews" and "submissions" in data and "items" not in data: # Old cache with 'submissions'
                 required_keys = ["timestamp", "submissions", "stats"] # Temporarily use old key for validation


            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys: # pragma: no cover
                logger.warning(f"Cache file for {platform}/{username} is incomplete (missing: {missing_keys}). Discarding.")
                cache_path.unlink(missing_ok=True)
                return None
            
            # Now that it's structurally valid, perform actual data migration for HackerNews if needed
            if platform == "hackernews" and "submissions" in data and "items" not in data: # pragma: no cover
                data["items"] = data.pop("submissions")
                logger.debug(f"Migrated 'submissions' to 'items' for legacy HackerNews cache: {cache_path}")


            # Offline mode: use any structurally valid cache, regardless of age
            if self.args.offline:
                cache_age = datetime.now(timezone.utc) - timestamp
                logger.info(f"Offline mode: Using cache for {platform}/{username} (Cache age: {cache_age}).")
                return data

            # Online mode: check cache freshness
            is_fresh = (datetime.now(timezone.utc) - timestamp) < timedelta(hours=CACHE_EXPIRY_HOURS)
            if is_fresh:
                logger.info(f"Cache hit and valid (fresh) for {platform}/{username}")
                return data
            else:
                logger.info(f"Cache expired for {platform}/{username}. Returning stale data for incremental baseline.")
                return data # Return stale data for incremental fetches when online

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e: # FileNotFoundError for unlink race condition
            logger.warning(f"Failed to load or parse cache for {platform}/{username}: {e}. Discarding cache.")
            cache_path.unlink(missing_ok=True) # Attempt to clean up bad cache
            return None
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error loading cache for {platform}/{username}: {e}", exc_info=True)
            cache_path.unlink(missing_ok=True) # Clean up potentially corrupted cache
            return None

    def _save_cache(self, platform: str, username: str, data: Dict[str, Any]):
        """Saves data to a user's cache file."""
        cache_path = self._get_cache_path(platform, username)
        try:
            # Ensure items are sorted chronologically before saving (newest first)
            sort_key_map = {
                "twitter": [("tweets", "created_at")],
                "reddit": [("submissions", "created_utc"), ("comments", "created_utc")],
                "bluesky": [("posts", "created_at")],
                "hackernews": [("items", "created_at")], # Changed from submissions to items
                "mastodon": [("posts", "created_at")],
            }
            if platform in sort_key_map:
                for list_key, dt_key in sort_key_map[platform]:
                    if list_key in data and isinstance(data[list_key], list) and data[list_key]:
                        # Filter out items that might be missing the sort key (though unlikely for primary data)
                        # items_to_sort = [item for item in data[list_key] if dt_key in item] # Not needed if get_sort_key handles missing
                        # Sort the original list in place
                        data[list_key].sort(key=lambda x: get_sort_key(x, dt_key), reverse=True)
                        logger.debug(f"Sorted '{list_key}' for {platform}/{username} by '{dt_key}'.")

            data["timestamp"] = datetime.now(timezone.utc) # Update timestamp
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2, cls=DateTimeEncoder), encoding="utf-8")
            logger.info(f"Saved cache for {platform}/{username} to {cache_path}")
        except TypeError as e: # pragma: no cover
            logger.error(f"Failed to serialize data for {platform}/{username} cache (TypeError): {e}. Data snippet: {str(data)[:500]}...", exc_info=True)
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to save cache for {platform}/{username}: {e}", exc_info=True)

    # --- Platform Fetch Methods (Twitter, Reddit, Bluesky, Mastodon, HackerNews) ---
    # These methods will now check `self.args.offline` at the beginning.
    # If offline:
    #   - Try to load cache.
    #   - If cache exists, return it.
    #   - If no cache, log a warning and return a minimal empty structure or None.
    #   - Do NOT make any API calls.
    # If online (or force_refresh is True):
    #   - Proceed with existing logic (load cache for incremental, then API calls).

    def fetch_twitter(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches tweets and user info for a Twitter user."""
        cached_data = self._load_cache("twitter", username)

        # Offline mode: use cache if available, otherwise return empty structure
        if self.args.offline:
            if cached_data:
                logger.info(f"Offline mode: Using cached data for Twitter @{username}.")
                return cached_data
            else:
                logger.warning(f"Offline mode: No cache found for Twitter @{username}. No data will be fetched.")
                # Return an empty structure for consistency downstream
                return {"timestamp": datetime.now(timezone.utc).isoformat(), "user_info": {}, "tweets": [], "media_analysis": [], "media_paths": []}

        # Online mode: proceed with normal fetch logic
        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Twitter @{username}")
            return cached_data

        logger.info(f"Fetching Twitter data for @{username} (Force Refresh: {force_refresh})")
        since_id = None
        existing_tweets = []
        existing_media_analysis = []
        existing_media_paths = []
        user_info = None

        if not force_refresh and cached_data: # Stale cache exists, prepare for incremental
            logger.info(f"Attempting incremental fetch for Twitter @{username}")
            existing_tweets = cached_data.get("tweets", [])
            if existing_tweets: # Should already be sorted by _save_cache
                # existing_tweets.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True) # Re-sort just in case
                since_id = existing_tweets[0].get("id") # ID of the newest tweet in cache
                if since_id: logger.debug(f"Using since_id: {since_id}")
                else: logger.warning("Found existing tweets but couldn't get ID from the first one.")
            user_info = cached_data.get("user_info")
            existing_media_analysis = cached_data.get("media_analysis", [])
            existing_media_paths = cached_data.get("media_paths", [])


        try:
            if not user_info or force_refresh: # Fetch user info if not in cache or forced
                try:
                    tw_client = self.twitter
                    # Essential fields for profile
                    user_response = tw_client.get_user(username=username, user_fields=["created_at", "public_metrics", "profile_image_url", "verified", "description", "location"])
                    if not user_response or not user_response.data:
                        raise UserNotFoundError(f"Twitter user @{username} not found.")
                    user = user_response.data
                    created_at_iso = user.created_at.isoformat() if user.created_at else None
                    user_info = {"id": str(user.id), "name": user.name, "username": user.username, "created_at": created_at_iso, "public_metrics": user.public_metrics, "profile_image_url": user.profile_image_url, "verified": user.verified, "description": user.description, "location": user.location}
                    logger.debug(f"Fetched user info for @{username}")
                except tweepy.NotFound: # Tweepy's specific exception for 404
                    raise UserNotFoundError(f"Twitter user @{username} not found.")
                except tweepy.Forbidden as e: # For suspended or protected users if info cannot be fetched
                    if "suspended" in str(e).lower(): # pragma: no cover
                        raise AccessForbiddenError(f"Twitter user @{username} is suspended.")
                    else: # pragma: no cover
                        raise AccessForbiddenError(f"Access forbidden to Twitter user @{username}'s profile (protected/private?).")
                except (tweepy.errors.Unauthorized, tweepy.errors.TweepyException) as auth_err: # pragma: no cover
                    logger.error(f"Twitter API authentication/request error getting user @{username}: {auth_err}")
                    raise RuntimeError(f"Twitter API error: {auth_err}")


            user_id = user_info["id"] # type: ignore
            new_tweets_data = [] # Raw tweet objects from API
            new_media_includes: Dict[str, List[Any]] = {} # Store media objects from 'includes' # Cast Dict type
            fetch_limit = INITIAL_FETCH_LIMIT if (force_refresh or not since_id) else INCREMENTAL_FETCH_LIMIT
            pagination_token = None
            tweets_fetch_count = 0

            # Loop for pagination if needed
            while True:
                current_page_limit = min(fetch_limit - tweets_fetch_count, 100) # Max 100 per page for user tweets
                if current_page_limit <= 0: break
                try:
                    tw_client = self.twitter
                    tweets_response = tw_client.get_users_tweets(id=user_id, max_results=current_page_limit, since_id=since_id if not force_refresh else None, pagination_token=pagination_token, tweet_fields=["created_at", "public_metrics", "attachments", "entities", "conversation_id", "in_reply_to_user_id", "referenced_tweets"], expansions=["attachments.media_keys", "author_id"], media_fields=["url", "preview_image_url", "type", "media_key", "width", "height", "alt_text"])
                except tweepy.TooManyRequests as e: # Rate limit
                    self._handle_rate_limit("Twitter", exception=e)
                    return None # Abort fetch for this user
                except tweepy.NotFound: # User gone between profile check and tweet fetch?
                    raise UserNotFoundError(f"Tweets not found for user ID {user_id} (@{username}). User might be protected, suspended or deleted after profile check.")
                except tweepy.Forbidden as e: # Protected tweets
                    if "protected" in str(e).lower(): # pragma: no cover
                        raise AccessForbiddenError(f"Access forbidden to @{username}'s tweets (account is protected).")
                    else: # pragma: no cover
                        raise AccessForbiddenError(f"Access forbidden to @{username}'s tweets (reason: {e}).")
                except (tweepy.errors.Unauthorized, tweepy.errors.TweepyException) as auth_err: # pragma: no cover
                    logger.error(f"Twitter API authentication/request error getting tweets for @{username}: {auth_err}")
                    raise RuntimeError(f"Twitter API error: {auth_err}")


                if tweets_response.data:
                    page_count = len(tweets_response.data)
                    new_tweets_data.extend(tweets_response.data)
                    tweets_fetch_count += page_count
                    logger.debug(f"Fetched {page_count} new tweets page (Total this run: {tweets_fetch_count}).")
                if tweets_response.includes: # Collate media from all pages
                    for key, items in tweets_response.includes.items():
                        if key not in new_media_includes: new_media_includes[key] = []
                        # Avoid duplicates if media appears in multiple pages' includes (unlikely for media_keys)
                        existing_keys = {item["media_key"] for item in new_media_includes[key] if "media_key" in item}
                        for item in items:
                            if "media_key" in item and item["media_key"] not in existing_keys:
                                new_media_includes[key].append(item)
                                existing_keys.add(item["media_key"])


                pagination_token = tweets_response.meta.get("next_token")
                if not pagination_token or tweets_fetch_count >= fetch_limit:
                    if pagination_token and tweets_fetch_count >= fetch_limit: logger.info(f"Reached fetch limit ({fetch_limit}) for Twitter @{username}.")
                    elif not pagination_token: logger.debug("No more pages found for Twitter tweets.")
                    break
            logger.info(f"Fetched {tweets_fetch_count} total new tweets for @{username}.")

            # Process new tweets and media
            processed_new_tweets = []
            newly_added_media_analysis = [] # Store analysis strings
            newly_added_media_paths = set()    # Store paths of downloaded media this run
            all_media_objects = {m.media_key: m for m in new_media_includes.get("media", [])} # type: ignore


            for tweet in new_tweets_data:
                media_items_for_tweet = []
                if tweet.attachments and "media_keys" in tweet.attachments:
                    for media_key in tweet.attachments["media_keys"]:
                        media = all_media_objects.get(media_key)
                        if media:
                            # Prefer full URL for photos/gifs, preview for videos if no direct video URL
                            url = media.url if media.type in ["photo", "gif"] and media.url else media.preview_image_url # type: ignore
                            if url:
                                media_path = self._download_media(url=url, platform="twitter", username=username)
                                if media_path:
                                    analysis = None
                                    if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS: # Analyze only if it's an image type we support
                                        analysis = self._analyze_image(media_path, f"Twitter user @{username}'s tweet (ID: {tweet.id})")
                                    media_items_for_tweet.append({"type": media.type, "analysis": analysis, "url": url, "alt_text": media.alt_text, "local_path": str(media_path)}) # type: ignore
                                    if analysis: newly_added_media_analysis.append(analysis)
                                    newly_added_media_paths.add(str(media_path))
                                else: logger.warning(f"Failed to download media {media.type} from {url} for tweet {tweet.id}") # type: ignore
                # Store referenced tweet info more simply
                referenced_tweets_info = []
                if tweet.referenced_tweets:
                    for ref in tweet.referenced_tweets: referenced_tweets_info.append({"type": ref.type, "id": str(ref.id)})

                tweet_data = {"id": str(tweet.id), "text": tweet.text, "created_at": tweet.created_at.isoformat(), "metrics": tweet.public_metrics, "entities": tweet.entities, # entities can be complex, store as is
                              "conversation_id": str(tweet.conversation_id), "in_reply_to_user_id": str(tweet.in_reply_to_user_id) if tweet.in_reply_to_user_id else None, "referenced_tweets": referenced_tweets_info, "media": media_items_for_tweet}
                processed_new_tweets.append(tweet_data)

            # Combine new and existing, sort, and limit
            processed_new_tweets.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True)
            existing_ids = {t["id"] for t in existing_tweets}
            unique_new_tweets = [t for t in processed_new_tweets if t["id"] not in existing_ids]

            combined_tweets = unique_new_tweets + existing_tweets
            combined_tweets.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True) # Ensure final sort
            final_tweets = combined_tweets[:MAX_CACHE_ITEMS]

            # Combine media analysis and paths, keep unique, limit
            valid_new_analyses = [a for a in newly_added_media_analysis if a] # Filter out None analyses
            final_media_analysis = sorted(list(set(valid_new_analyses + [m for m in existing_media_analysis if m]))) # Unique sorted analyses
            final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))[:MAX_CACHE_ITEMS*2] # Limit paths too

            # Note: Twitter API v2 user object doesn't directly expose total likes/retweets received
            # Stats would require fetching all tweets and summing metrics, which can be extensive.
            # Omitting complex stats for now, rely on user_info public_metrics.

            final_data = {"timestamp": datetime.now(timezone.utc).isoformat(), # Handled by _save_cache
                          "user_info": user_info, "tweets": final_tweets, "media_analysis": final_media_analysis, "media_paths": final_media_paths}
            self._save_cache("twitter", username, final_data)
            logger.info(f"Successfully updated Twitter cache for @{username}. Total tweets cached: {len(final_tweets)}")
            return final_data

        except RateLimitExceededError: # pragma: no cover
            logger.warning(f"Twitter fetch for @{username} aborted due to rate limit.")
            return None # Or re-raise if main loop should handle it
        except (UserNotFoundError, AccessForbiddenError) as user_err:
            logger.error(f"Twitter fetch failed for @{username}: {user_err}")
            # self.console.print(f"[yellow]Skipping Twitter user @{username}: {user_err}[/yellow]") # Caller will print
            return None # Don't save cache if user not found or forbidden
        except RuntimeError as e: # From self.twitter or other setup issues
            logger.error(f"Runtime error during Twitter fetch for @{username}: {e}")
            return None
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error fetching Twitter data for @{username}: {str(e)}", exc_info=True)
            return None
    
    def fetch_reddit(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches submissions, comments, and user profile for a Reddit user."""
        cached_data = self._load_cache("reddit", username)

        # Offline mode: use cache if available, otherwise return empty structure
        if self.args.offline:
            if cached_data:
                logger.info(f"Offline mode: Using cached data for Reddit u/{username}.")
                return cached_data
            else:
                logger.warning(f"Offline mode: No cache found for Reddit u/{username}. No data will be fetched.")
                return {"timestamp": datetime.now(timezone.utc).isoformat(), "user_profile": {}, "submissions": [], "comments": [], "media_analysis": [], "media_paths": [], "stats": {}}

        # Online mode: proceed with normal fetch logic
        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Reddit u/{username}")
            return cached_data

        logger.info(f"Fetching Reddit data for u/{username} (Force Refresh: {force_refresh})")
        latest_submission_fullname = None # t3_xxxxxx
        latest_comment_fullname = None    # t1_xxxxxx
        existing_submissions = []
        existing_comments = []
        existing_media_analysis = []
        existing_media_paths = []
        cached_profile_info = None # Will hold profile from cache if available

        if not force_refresh and cached_data: # Stale cache exists
            logger.info(f"Attempting incremental fetch for Reddit u/{username}")
            existing_submissions = cached_data.get("submissions", [])
            existing_comments = cached_data.get("comments", [])
            if existing_submissions: # Assumed sorted
                latest_submission_fullname = existing_submissions[0].get("fullname")
                if latest_submission_fullname: logger.debug(f"Using latest submission fullname: {latest_submission_fullname}")
            if existing_comments: # Assumed sorted
                latest_comment_fullname = existing_comments[0].get("fullname")
                if latest_comment_fullname: logger.debug(f"Using latest comment fullname: {latest_comment_fullname}")
            
            cached_profile_info = cached_data.get("user_profile") # Load profile from cache
            existing_media_analysis = cached_data.get("media_analysis", [])
            existing_media_paths = cached_data.get("media_paths", [])

        redditor_info: Dict[str, Any] = {} # This will hold the profile info for the current operation

        try:
            reddit_client = self.reddit
            redditor_praw_obj = reddit_client.redditor(username) # Get the PRAW Redditor object

            # Determine if we need to fetch live profile data
            should_fetch_live_profile = True

            if self.args.offline: # Should have been caught by the initial offline check, but defensive
                if cached_profile_info:
                    redditor_info = cached_profile_info
                    logger.info(f"Offline mode (redundant check): Using cached profile for Reddit u/{username}.")
                    should_fetch_live_profile = False
                else:
                    logger.warning(f"Offline mode (redundant check): No cached profile found for Reddit u/{username}. Profile details will be unavailable.")
                    should_fetch_live_profile = False 
            elif not force_refresh and cached_profile_info: # Online, not forcing refresh, and have cached profile
                redditor_info = cached_profile_info # Use cached profile
                logger.info(f"Using cached profile for Reddit u/{username}.")
                should_fetch_live_profile = False
            
            # If we still need to fetch live (online AND (force_refresh OR no cached_profile_info available))
            if should_fetch_live_profile:
                logger.info(f"Fetching live profile data for Reddit u/{username}.")
                try:
                    # These attributes will trigger API calls
                    # Ensure redditor_praw_obj is not None before accessing attributes
                    if redditor_praw_obj is None: # Should not happen if reddit_client.redditor() was successful
                        raise RuntimeError("PRAW Redditor object is None, cannot fetch profile.")

                    live_redditor_id = redditor_praw_obj.id 
                    live_redditor_created_utc = redditor_praw_obj.created_utc
                    
                    # Populate redditor_info with live data
                    redditor_info = {
                        "id": live_redditor_id, 
                        "name": redditor_praw_obj.name, # Should match input 'username'
                        "created_utc": datetime.fromtimestamp(live_redditor_created_utc, tz=timezone.utc).isoformat(),
                        "link_karma": getattr(redditor_praw_obj, "link_karma", 0),
                        "comment_karma": getattr(redditor_praw_obj, "comment_karma", 0),
                        "icon_img": getattr(redditor_praw_obj, "icon_img", None),
                        "is_suspended": getattr(redditor_praw_obj, "is_suspended", False)
                    }
                    if redditor_info.get("is_suspended"):
                        logger.warning(f"Reddit user u/{username} (live check) is suspended.")
                    logger.debug(f"Successfully fetched live profile for Reddit u/{username}.")

                except prawcore.exceptions.NotFound:
                    raise UserNotFoundError(f"Reddit user u/{username} not found (live profile fetch).")
                except prawcore.exceptions.Forbidden: 
                    # This can happen if user is suspended/shadowbanned, or if our access is blocked
                    logger.warning(f"Access forbidden to Reddit user u/{username}'s profile (live fetch). User might be suspended, shadowbanned, or access blocked.")
                    is_suspended_attr = getattr(redditor_praw_obj, 'is_suspended', None) if redditor_praw_obj else None
                    if is_suspended_attr is True: 
                         redditor_info = {"name": username, "is_suspended": True, "id": None} 
                         logger.warning(f"Marking user u/{username} as suspended based on PRAW attribute after Forbidden error.")
                    else: 
                         raise AccessForbiddenError(f"Access forbidden to Reddit user u/{username}'s profile (live fetch).")

                except (prawcore.exceptions.PrawcoreException, RuntimeError) as client_err: 
                    logger.error(f"Reddit API/client error accessing user u/{username} (live profile fetch): {client_err}")
                    raise RuntimeError(f"Reddit API error (live profile fetch): {client_err}")

            # --- Fetch Submissions ---
            new_submissions_data = []
            newly_added_media_analysis = [] # For this fetch run only
            newly_added_media_paths = set()    # For this fetch run only
            fetch_limit_val = INCREMENTAL_FETCH_LIMIT 
            count_subs = 0 
            processed_submission_ids = {s["id"] for s in existing_submissions if "id" in s} # Ensure 'id' exists

            logger.debug("Fetching new submissions...")
            try:
                # Ensure redditor_praw_obj is not None before using it
                if redditor_praw_obj is None:
                     raise RuntimeError("PRAW Redditor object is None, cannot fetch submissions.")
                params_subs: Dict[str, Union[int, str]] = {"limit": fetch_limit_val}
                if not force_refresh and latest_submission_fullname:
                    params_subs["before"] = latest_submission_fullname 
                    logger.debug(f"Fetching submissions before {latest_submission_fullname}")
                
                for submission in redditor_praw_obj.submissions.new(limit=fetch_limit_val, params=params_subs):
                    count_subs +=1
                    submission_id = submission.id
                    if submission_id in processed_submission_ids: 
                        logger.debug(f"Skipping already processed submission ID: {submission_id}")
                        continue

                    media_items_for_submission = []
                    submission_url = getattr(submission, "url", None)
                    if submission_url:
                        is_direct_media_link = any(submission_url.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS + [".mp4", ".webm", ".mov"])
                        is_reddit_media = any(host in urlparse(submission_url).netloc for host in ["i.redd.it", "v.redd.it", "preview.redd.it"])

                        if is_direct_media_link or is_reddit_media:
                            media_path = self._download_media(url=submission_url, platform="reddit", username=username)
                            if media_path:
                                analysis = None
                                if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                    analysis = self._analyze_image(media_path, f"Reddit user u/{username}'s post in r/{submission.subreddit.display_name} (ID: {submission_id})")
                                media_items_for_submission.append({"type": "image" if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else "video", "analysis": analysis, "url": submission_url, "local_path": str(media_path)})
                                if analysis: newly_added_media_analysis.append(analysis)
                                newly_added_media_paths.add(str(media_path))

                    is_gallery = getattr(submission, "is_gallery", False)
                    media_metadata = getattr(submission, "media_metadata", None)
                    if not media_items_for_submission and is_gallery and media_metadata: 
                        for media_id_key, media_item_val in media_metadata.items(): 
                            source = media_item_val.get("s")
                            preview_data = media_item_val.get("p", []) 
                            image_url = None
                            if source: image_url = source.get("u") or source.get("gif") 
                            if not image_url and preview_data: image_url = preview_data[-1].get("u") 

                            if image_url:
                                image_url_clean = image_url.replace("&amp;", "&") 
                                media_path = self._download_media(url=image_url_clean, platform="reddit", username=username)
                                if media_path:
                                    analysis = None
                                    if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                        analysis = self._analyze_image(media_path, f"Reddit user u/{username}'s gallery post in r/{submission.subreddit.display_name} (ID: {submission_id}, Media: {media_id_key})")
                                    media_items_for_submission.append({"type": "gallery_image", "analysis": analysis, "url": image_url_clean, "alt_text": media_item_val.get("caption") or media_item_val.get("title"), "local_path": str(media_path)})
                                    if analysis: newly_added_media_analysis.append(analysis)
                                    newly_added_media_paths.add(str(media_path))
                    
                    submission_data = {"id": submission_id, "fullname": submission.fullname, "title": submission.title, 
                                       "text": submission.selftext[:2000] if hasattr(submission, "selftext") else "",
                                       "score": submission.score, "upvote_ratio": getattr(submission, "upvote_ratio", None), 
                                       "subreddit": submission.subreddit.display_name, "permalink": f"https://www.reddit.com{submission.permalink}", 
                                       "created_utc": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
                                       "url": submission.url if submission.is_self else None, 
                                       "link_url": submission.url if not submission.is_self else None,
                                       "is_self": submission.is_self, "is_gallery": is_gallery, 
                                       "num_comments": submission.num_comments, "stickied": submission.stickied, 
                                       "over_18": submission.over_18, "spoiler": submission.spoiler, "media": media_items_for_submission}
                    new_submissions_data.append(submission_data)
                    processed_submission_ids.add(submission_id) 
            except prawcore.exceptions.Forbidden: 
                logger.warning(f"Access forbidden while fetching submissions for u/{username} (possibly subreddit restriction).")
            except prawcore.exceptions.RequestException as req_err:
                if hasattr(req_err, "response") and req_err.response is not None and req_err.response.status_code == 429: 
                    self._handle_rate_limit("Reddit", exception=req_err)
                    return None
                else: 
                    logger.error(f"Reddit request failed fetching submissions for u/{username}: {req_err}")
            except RuntimeError as rt_err: # Catch if redditor_praw_obj was None
                logger.error(f"Runtime error during Reddit submission fetch for u/{username}: {rt_err}")
                # Potentially return None or re-raise depending on desired handling
                return None
            logger.info(f"Fetched {len(new_submissions_data)} new submissions for u/{username} (scanned approx {count_subs}).")

            # --- Fetch Comments ---
            new_comments_data = []
            count_comments = 0
            processed_comment_ids = {c["id"] for c in existing_comments if "id" in c} # Ensure 'id' exists
            logger.debug("Fetching new comments...")
            try:
                # Ensure redditor_praw_obj is not None
                if redditor_praw_obj is None:
                    raise RuntimeError("PRAW Redditor object is None, cannot fetch comments.")
                params_comments: Dict[str, Union[int, str]] = {"limit": fetch_limit_val}
                if not force_refresh and latest_comment_fullname:
                    params_comments["before"] = latest_comment_fullname
                    logger.debug(f"Fetching comments before {latest_comment_fullname}")
                for comment in redditor_praw_obj.comments.new(limit=fetch_limit_val, params=params_comments):
                    count_comments += 1
                    comment_id = comment.id
                    if comment_id in processed_comment_ids:
                        logger.debug(f"Skipping already processed comment ID: {comment_id}")
                        continue
                    new_comments_data.append({"id": comment_id, "fullname": comment.fullname, 
                                              "text": comment.body[:2000], 
                                              "score": comment.score, "subreddit": comment.subreddit.display_name, 
                                              "permalink": f"https://www.reddit.com{comment.permalink}", 
                                              "created_utc": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                                              "is_submitter": comment.is_submitter, "stickied": comment.stickied, 
                                              "parent_id": comment.parent_id, "submission_id": comment.submission.id})
                    processed_comment_ids.add(comment_id)
            except prawcore.exceptions.Forbidden: 
                 logger.warning(f"Access forbidden while fetching comments for u/{username}.")
            except prawcore.exceptions.RequestException as req_err: 
                if hasattr(req_err, "response") and req_err.response is not None and req_err.response.status_code == 429:
                    self._handle_rate_limit("Reddit", exception=req_err)
                    return None
                else: 
                    logger.error(f"Reddit request failed fetching comments for u/{username}: {req_err}")
            except RuntimeError as rt_err: # Catch if redditor_praw_obj was None
                logger.error(f"Runtime error during Reddit comment fetch for u/{username}: {rt_err}")
                return None
            logger.info(f"Fetched {len(new_comments_data)} new comments for u/{username} (scanned approx {count_comments}).")


            # Combine, sort, limit
            combined_submissions = new_submissions_data + existing_submissions
            combined_comments = new_comments_data + existing_comments
            combined_submissions.sort(key=lambda x: get_sort_key(x, "created_utc"), reverse=True)
            combined_comments.sort(key=lambda x: get_sort_key(x, "created_utc"), reverse=True)
            final_submissions = combined_submissions[:MAX_CACHE_ITEMS]
            final_comments = combined_comments[:MAX_CACHE_ITEMS]

            valid_new_analyses = [a for a in newly_added_media_analysis if a]
            final_media_analysis = sorted(list(set(valid_new_analyses + [m for m in existing_media_analysis if m])))
            final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))[:MAX_CACHE_ITEMS*2]

            # Calculate some basic stats
            total_submissions = len(final_submissions)
            total_comments = len(final_comments)
            submissions_with_media = len([s for s in final_submissions if s.get("media")])
            avg_sub_score = sum(s.get("score",0) for s in final_submissions) / max(total_submissions, 1)
            avg_comment_score = sum(c.get("score",0) for c in final_comments) / max(total_comments, 1)
            avg_sub_upvote_ratio = sum(s.get("upvote_ratio", 0.0) or 0.0 for s in final_submissions if s.get("upvote_ratio") is not None) / max(len([s for s in final_submissions if s.get("upvote_ratio") is not None]), 1)

            stats = {"total_submissions_cached": total_submissions, "total_comments_cached": total_comments, 
                     "submissions_with_media": submissions_with_media, "total_media_items_processed": len(final_media_paths), 
                     "avg_submission_score": round(avg_sub_score,2), "avg_comment_score": round(avg_comment_score,2), 
                     "avg_submission_upvote_ratio": round(avg_sub_upvote_ratio,3)}

            final_data = {"timestamp": datetime.now(timezone.utc).isoformat(), 
                          "user_profile": redditor_info, # Use the correctly populated redditor_info
                          "submissions": final_submissions, "comments": final_comments, 
                          "media_analysis": final_media_analysis, "media_paths": final_media_paths, "stats": stats}
            self._save_cache("reddit", username, final_data)
            logger.info(f"Successfully updated Reddit cache for u/{username}. Cached submissions: {total_submissions}, comments: {total_comments}")
            return final_data

        except RateLimitExceededError: 
            logger.warning(f"Reddit fetch for u/{username} aborted due to rate limit.")
            return None
        except (UserNotFoundError, AccessForbiddenError) as user_err:
            logger.error(f"Reddit fetch failed for u/{username}: {user_err}")
             # If user_profile could not be fetched and we have no cached profile, we might still have submissions/comments.
             # Decide if we want to save partial data or not. Currently, it returns None, so no save.
            return None
        except RuntimeError as e: 
            logger.error(f"Runtime error during Reddit fetch for u/{username}: {e}")
            return None
        except Exception as e: 
            logger.error(f"Unexpected error fetching Reddit data for u/{username}: {str(e)}", exc_info=True)
            return None

    def fetch_bluesky(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches posts and user profile for a Bluesky user."""
        cached_data = self._load_cache("bluesky", username)

        # Offline mode: use cache if available, otherwise return empty structure
        if self.args.offline:
            if cached_data:
                logger.info(f"Offline mode: Using cached data for Bluesky user {username}.")
                return cached_data
            else:
                logger.warning(f"Offline mode: No cache found for Bluesky user {username}. No data will be fetched.")
                return {"timestamp": datetime.now(timezone.utc).isoformat(), "profile_info": {}, "posts": [], "media_analysis": [], "media_paths": [], "stats": {}}

        # Online mode: proceed with normal fetch logic
        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Bluesky user {username}")
            return cached_data

        logger.info(f"Fetching Bluesky data for {username} (Force Refresh: {force_refresh})")
        latest_post_datetime = None # Store actual datetime for comparison
        existing_posts = []
        existing_media_analysis = []
        existing_media_paths = []
        profile_info = None # Store user profile


        if not force_refresh and cached_data: # Stale cache exists
            logger.info(f"Attempting incremental fetch for Bluesky {username}")
            existing_posts = cached_data.get("posts", [])
            if existing_posts: # Assumed sorted
                # existing_posts.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True)
                # Get datetime of the newest post in cache
                latest_post_datetime = get_sort_key(existing_posts[0], "created_at")
                logger.debug(f"Latest known post datetime: {latest_post_datetime.isoformat() if latest_post_datetime else 'N/A'}")
            profile_info = cached_data.get("profile_info") # User profile from cache
            existing_media_analysis = cached_data.get("media_analysis", [])
            existing_media_paths = cached_data.get("media_paths", [])

        try:
            bsky_client = self.bluesky
            if not profile_info or force_refresh: # Fetch profile if not cached or forced
                try:
                    profile = bsky_client.get_profile(actor=username) # `username` is actor handle or DID
                    labels_list = []
                    if profile.labels: labels_list = [{"value": lbl.val, "timestamp": lbl.cts} for lbl in profile.labels] # type: ignore
                    profile_info = {"did": profile.did, "handle": profile.handle, "display_name": profile.display_name, "description": profile.description, "avatar": profile.avatar, "banner": profile.banner, "followers_count": profile.followers_count, "follows_count": profile.follows_count, "posts_count": profile.posts_count, "labels": labels_list}
                    logger.debug(f"Fetched Bluesky profile info for {username}")
                except atproto_exceptions.AtProtocolError as e: # More specific error handling
                    err_str = str(e).lower()
                    if isinstance(e, atproto_exceptions.BadRequestError) and \
                       ("profile not found" in err_str or "could not resolve handle" in err_str):
                        raise UserNotFoundError(f"Bluesky user {username} not found or handle invalid.")
                    elif isinstance(e, atproto_exceptions.NetworkError) and \
                         ("blocked by actor" in err_str or "blocking actor" in err_str): # pragma: no cover (hard to test blocking)
                        raise AccessForbiddenError(f"Blocked from accessing Bluesky profile for {username}.")
                    elif isinstance(e, atproto_exceptions.UnauthorizedError): # pragma: no cover
                        logger.error(f"Bluesky authentication error fetching profile for {username}: {e}")
                        raise RuntimeError(f"Bluesky authentication failed: {e}")
                    else: # pragma: no cover
                        logger.error(f"Unexpected error fetching Bluesky profile for {username}: {e}")
                        raise AccessForbiddenError(f"Error fetching Bluesky profile for {username}: {e}")


            new_posts_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            cursor = None
            processed_uris = set(p["uri"] for p in existing_posts if "uri" in p) # Ensure 'uri' exists # Keep track of URIs already in cache
            fetch_limit_per_page = min(INCREMENTAL_FETCH_LIMIT, 100) # API max is 100
            total_fetched_this_run = 0
            # Determine overall fetch limit for this run
            max_fetches = INITIAL_FETCH_LIMIT if (force_refresh or not latest_post_datetime) else INCREMENTAL_FETCH_LIMIT
            reached_old_post = False

            logger.debug(f"Fetching new Bluesky posts for {username}...")
            while total_fetched_this_run < max_fetches and not reached_old_post:
                try:
                    response = bsky_client.get_author_feed(actor=username, cursor=cursor, limit=fetch_limit_per_page)
                except atproto_exceptions.RateLimitExceededError as rle: # pragma: no cover
                    self._handle_rate_limit("Bluesky", exception=rle)
                    return None
                except atproto_exceptions.AtProtocolError as e: # pragma: no cover (various errors can occur here)
                    err_str = str(e).lower()
                    if isinstance(e, atproto_exceptions.BadRequestError) and \
                       ("could not resolve handle" in err_str or "profile not found" in err_str): # Handle might be deactivated
                        raise UserNotFoundError(f"Bluesky user {username} not found or handle cannot be resolved during feed fetch.")
                    if isinstance(e, atproto_exceptions.NetworkError) and \
                       ("blocked by actor" in err_str or "blocking actor" in err_str):
                        raise AccessForbiddenError(f"Access to Bluesky user {username}'s feed is blocked.")
                    elif isinstance(e, atproto_exceptions.UnauthorizedError):
                        logger.error(f"Bluesky authentication error fetching feed for {username}: {e}")
                        raise RuntimeError(f"Bluesky authentication failed: {e}")

                    logger.error(f"Bluesky API error fetching feed for {username}: {e}")
                    return None # Abort fetch for this user


                if not response or not response.feed:
                    logger.debug("No more posts found in feed.")
                    break
                logger.debug(f"Processing feed page with {len(response.feed)} items. Cursor: {response.cursor}")

                for feed_item in response.feed:
                    # Ensure feed_item.post exists and has a record
                    if not hasattr(feed_item, "post"): # Skip non-post items (e.g. list item notifications if they appeared)
                        logger.debug(f"Skipping feed item without 'post' attribute: {feed_item}")
                        continue
                    post = feed_item.post
                    post_uri = post.uri
                    if post_uri in processed_uris: continue # Already have this post

                    record = getattr(post, "record", None)
                    if not record: continue # Skip if no record data

                    # Compare creation time for incremental fetch
                    created_at_dt = get_sort_key({"created_at": getattr(record, "created_at", None)}, "created_at")
                    if not force_refresh and latest_post_datetime and created_at_dt <= latest_post_datetime:
                        logger.info(f"Reached post ({post_uri} at {created_at_dt.isoformat()}) older than or same as latest known post ({latest_post_datetime.isoformat() if latest_post_datetime else 'N/A'}). Stopping incremental fetch.")
                        reached_old_post = True
                        break

                    media_items_for_post = []
                    embed = getattr(record, "embed", None)
                    image_embeds_to_process = [] # Collect all image embeds
                    embed_type_str = getattr(embed, "$type", "unknown") if embed else None # type: ignore

                    if embed:
                        # Direct images embed
                        if hasattr(embed, "images"): image_embeds_to_process.extend(embed.images) # type: ignore
                        # Images in a media embed (e.g., within a record embed)
                        media_embed = getattr(embed, "media", None) # Could be app.bsky.embed.record#viewRecord
                        record_embed = getattr(embed, "record", None) # Could be app.bsky.embed.record#viewRecord

                        if media_embed and hasattr(media_embed, "images"): image_embeds_to_process.extend(media_embed.images) # type: ignore
                        # Handle images within a quoted/embedded record (e.g. a post quoting another post with images)
                        if record_embed: # This is a ViewRecord
                            # The actual content of the embedded record is in record.record.value or record.value
                            # For ViewRecord, it's record.value or record.embeds[0].record (if it's a recordWithMedia)
                            # Let's check for nested embeds within the `record_embed` (which is a ViewRecord)
                            # A ViewRecord can itself have embeds
                            nested_record_value = getattr(record_embed, "record", None) or getattr(record_embed, "value", None) # Adjusted for ViewRecord structure
                            if nested_record_value: # This is now the actual record data (e.g. PostRecord)
                                nested_embed = getattr(nested_record_value, "embed", None)
                                if nested_embed and hasattr(nested_embed, "images"):
                                    image_embeds_to_process.extend(nested_embed.images) # type: ignore
                    
                    for image_info in image_embeds_to_process: # ImageInfo object
                        img_blob = getattr(image_info, "image", None) # This is the actual BlobRef # Renamed img
                        alt_text = getattr(image_info, "alt", "")
                        if img_blob:
                            # CID can be in blob.cid or blob.ref.link (older format)
                            cid = getattr(img_blob, "cid", None) or getattr(getattr(img_blob, "ref", None), "link", None)
                            if cid:
                                author_did = post.author.did # DID of the post author for CDN URL
                                img_mime_type = getattr(img_blob, "mime_type", "image/jpeg").split("/")[-1] # Get extension
                                # Construct CDN URL (example, actual format might vary or use API)
                                # Standard is: https://{PDS_HOST}/xrpc/com.atproto.sync.getBlob?did={USER_DID}&cid={IMAGE_CID}
                                # But public CDNs are often: cdn.bsky.app/img/{type}/plain/{did}/{cid}@{ext}
                                safe_author_did = quote_plus(str(author_did))
                                safe_cid = quote_plus(str(cid))
                                cdn_url = f"https://cdn.bsky.app/img/feed_fullsize/plain/{safe_author_did}/{safe_cid}@{img_mime_type}"

                                media_path = self._download_media(url=cdn_url, platform="bluesky", username=username)
                                if media_path:
                                    analysis = None
                                    if media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                        analysis = self._analyze_image(media_path, f"Bluesky user {username}'s post ({post.uri})")
                                    media_items_for_post.append({"type": "image", "analysis": analysis, "url": cdn_url, "alt_text": alt_text, "local_path": str(media_path)})
                                    if analysis: newly_added_media_analysis.append(analysis)
                                    newly_added_media_paths.add(str(media_path))
                                else: logger.warning(f"Failed to download Bluesky image from {cdn_url} for post {post_uri}")
                            else: logger.warning(f"Could not find image CID/link in image blob for post {post_uri}")
                        else: logger.warning(f"Image embed structure missing 'image' blob for post {post_uri}")


                    reply_ref = getattr(record, "reply", None)
                    reply_parent_uri = None
                    reply_root_uri = None
                    if reply_ref:
                        parent_ref = getattr(reply_ref, "parent", None)
                        if parent_ref and hasattr(parent_ref, "uri"): reply_parent_uri = parent_ref.uri
                        root_ref = getattr(reply_ref, "root", None)
                        if root_ref and hasattr(root_ref, "uri"): reply_root_uri = root_ref.uri

                    post_langs = getattr(record, "langs", [])

                    post_data = {"uri": post_uri, "cid": post.cid, "author_did": post.author.did, "text": getattr(record, "text", "")[:3000], # Limit text
                                 "created_at": created_at_dt.isoformat(), "langs": post_langs,
                                 "reply_parent": reply_parent_uri, "reply_root": reply_root_uri,
                                 "likes": getattr(post, "like_count", 0), "reposts": getattr(post, "repost_count", 0), "reply_count": getattr(post, "reply_count",0),
                                 "embed_type": embed_type_str, "media": media_items_for_post}
                    new_posts_data.append(post_data)
                    processed_uris.add(post_uri)
                    total_fetched_this_run += 1
                    if total_fetched_this_run >= max_fetches:
                        logger.info(f"Reached fetch limit ({max_fetches}) for Bluesky {username}.")
                        break # Break from inner loop (feed items)

                if reached_old_post: break # Break from outer while loop
                cursor = response.cursor
                if not cursor:
                    logger.debug("Reached end of feed (no cursor).")
                    break # No more pages
                if total_fetched_this_run >= max_fetches: break # Break from outer while loop

            logger.info(f"Fetched {len(new_posts_data)} new posts for Bluesky user {username}.")

            # Combine, sort, limit (newest first)
            existing_uris_set = {p["uri"] for p in existing_posts if "uri" in p} # Rebuild to be sure, ensure 'uri' exists
            unique_new_posts = [p for p in new_posts_data if p.get("uri") not in existing_uris_set] # Use .get for safety

            combined_posts = unique_new_posts + existing_posts
            combined_posts.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True)
            final_posts = combined_posts[:MAX_CACHE_ITEMS]

            valid_new_analyses = [a for a in newly_added_media_analysis if a]
            final_media_analysis = sorted(list(set(valid_new_analyses + [m for m in existing_media_analysis if m])))
            final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))[:MAX_CACHE_ITEMS*2]

            # Stats
            total_posts = len(final_posts)
            posts_with_media = len([p for p in final_posts if p.get("media")])
            reply_posts = len([p for p in final_posts if p.get("reply_parent")]) # Simple count of replies
            repost_count_sum = sum(p.get("reposts",0) for p in final_posts)
            like_count_sum = sum(p.get("likes",0) for p in final_posts)

            stats = {"total_posts_cached": total_posts, "posts_with_media": posts_with_media, "reply_posts_cached": reply_posts,
                     "total_media_items_processed": len(final_media_paths),
                     "avg_likes": round(like_count_sum / max(total_posts,1), 2),
                     "avg_reposts": round(repost_count_sum / max(total_posts,1), 2),
                     "avg_replies": round(sum(p.get("reply_count",0) for p in final_posts) / max(total_posts,1), 2)
                     }

            final_data = {"timestamp": datetime.now(timezone.utc).isoformat(), # Handled by _save_cache
                          "profile_info": profile_info, "posts": final_posts, "media_analysis": final_media_analysis, "media_paths": final_media_paths, "stats": stats}
            self._save_cache("bluesky", username, final_data)
            logger.info(f"Successfully updated Bluesky cache for {username}. Total posts cached: {total_posts}")
            return final_data

        except RateLimitExceededError: # pragma: no cover
            logger.warning(f"Bluesky fetch for {username} aborted due to rate limit.")
            return None
        except (UserNotFoundError, AccessForbiddenError) as user_err:
            logger.error(f"Bluesky fetch failed for {username}: {user_err}")
            return None
        except RuntimeError as e: # From self.bluesky setup
            logger.error(f"Runtime error during Bluesky fetch for {username}: {e}")
            return None
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error fetching Bluesky data for {username}: {str(e)}", exc_info=True)
            return None

    def fetch_mastodon(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches statuses (posts/boosts) and user info for a Mastodon user."""
        # Username MUST be in format user@instance.domain for Mastodon.py account_lookup
        # and for consistent caching key.
        cache_key_username = username # Use the full user@instance for cache key
        # Use _get_instance_domain_from_acct for a more robust check of the instance part
        if "@" not in cache_key_username or not self._get_instance_domain_from_acct(cache_key_username):
            logger.error(f"Invalid Mastodon username format for fetch: '{cache_key_username}'. Needs 'user@instance.domain'.")
            # This should ideally be caught earlier during input, but double check here.
            raise ValueError(f"Invalid Mastodon username format: '{cache_key_username}'. Must be 'user@instance.domain'.")


        cached_data = self._load_cache("mastodon", cache_key_username)

        # Offline mode: use cache if available, otherwise return empty structure
        if self.args.offline:
            if cached_data:
                logger.info(f"Offline mode: Using cached data for Mastodon user {cache_key_username}.")
                return cached_data
            else:
                logger.warning(f"Offline mode: No cache found for Mastodon user {cache_key_username}. No data will be fetched.")
                return {"timestamp": datetime.now(timezone.utc).isoformat(), "user_info": {}, "posts": [], "media_analysis": [], "media_paths": [], "stats": {}}

        # Online mode: proceed with normal fetch logic
        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for Mastodon user {cache_key_username}")
            return cached_data

        logger.info(f"Fetching Mastodon data for {cache_key_username} (Force Refresh: {force_refresh})")

        # Get the appropriate Mastodon client
        masto_client = self._get_mastodon_client(target_user_acct=username)
        if not masto_client:
            logger.error(f"No suitable Mastodon client found to fetch data for {username}. Check Mastodon configuration (e.g., mastodon_instances.json).")
            # Returning None will be handled by the caller as a failed fetch.
            return None

        logger.info(f"Using Mastodon client for instance: {masto_client.api_base_url} to process user {username}")

        since_id = None # ID of the newest status in cache to fetch newer ones
        existing_posts = []
        existing_media_analysis = []
        existing_media_paths = []
        user_info = None # Store full user profile


        if not force_refresh and cached_data: # Stale cache exists
            logger.info(f"Attempting incremental fetch for Mastodon {cache_key_username}")
            existing_posts = cached_data.get("posts", [])
            if existing_posts: # Assumed sorted by _save_cache
                # existing_posts.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True) # Re-sort just in case
                since_id = existing_posts[0].get("id") # Mastodon status ID
                if since_id: logger.debug(f"Using since_id: {since_id}")
            user_info = cached_data.get("user_info")
            existing_media_analysis = cached_data.get("media_analysis", [])
            existing_media_paths = cached_data.get("media_paths", [])


        try:
            # masto_client is now sourced from _get_mastodon_client()
            if not user_info or force_refresh:
                try:
                    # account_lookup resolves user@otherinstance.domain from configured instance
                    logger.debug(f"Looking up Mastodon account: {username} using client for {masto_client.api_base_url}")
                    account = masto_client.account_lookup(acct=username) # `username` is user@instance.domain
                    # Parse and store comprehensive user info
                    created_at_dt = account.get("created_at") # This is a datetime object from Mastodon.py
                    created_at_iso = created_at_dt.isoformat() if isinstance(created_at_dt, datetime) else str(created_at_dt)
                    custom_fields = []
                    if account.get("fields"): custom_fields = [{"name": f.get("name"), "value": f.get("value")} for f in account["fields"]]
                    user_info = {"id": str(account["id"]), "username": account["username"], # local username on their instance
                                 "acct": account["acct"], # full user@instance
                                 "display_name": account["display_name"],
                                 "note_html": account.get("note", ""), # Bio as HTML
                                 "note_text": BeautifulSoup(account.get("note",""), "html.parser").get_text(separator=" ", strip=True), # Cleaned bio text
                                 "url": account["url"], "avatar": account["avatar"], "header": account["header"],
                                 "locked": account.get("locked", False), "bot": account.get("bot", False),
                                 "discoverable": account.get("discoverable"), "group": account.get("group", False),
                                 "followers_count": account["followers_count"], "following_count": account["following_count"],
                                 "statuses_count": account["statuses_count"],
                                 "last_status_at": account.get("last_status_at"), # Can be str or datetime
                                 "created_at": created_at_iso, "custom_fields": custom_fields}
                    logger.info(f"Fetched Mastodon user info for {cache_key_username}")
                    if user_info["locked"]: logger.warning(f"Mastodon user {cache_key_username} has a locked/private account. Status fetch might be limited or fail.")
                except MastodonNotFoundError:
                    raise UserNotFoundError(f"Mastodon user {username} not found via {masto_client.api_base_url}.")
                except MastodonUnauthorizedError: # pragma: no cover (instance might disallow lookup, or account locked + no follow)
                    raise AccessForbiddenError(f"Unauthorized access to Mastodon user {username}'s info (locked account / instance policy?).")
                except (MastodonVersionError, MastodonError) as e: # pragma: no cover
                    logger.error(f"Mastodon API error looking up {username}: {e}")
                    err_str = str(e).lower()
                    if "blocked" in err_str or "forbidden" in err_str or "federation" in err_str:
                         raise AccessForbiddenError(f"Forbidden/Blocked from accessing Mastodon user {username}'s info (possibly blocked by user/instance or federation issue).")
                    else: raise RuntimeError(f"Mastodon API error during user lookup: {e}")


            user_id_val = user_info["id"] # type: ignore # Renamed user_id
            new_posts_data = []
            newly_added_media_analysis = []
            newly_added_media_paths = set()
            fetch_limit_val = INITIAL_FETCH_LIMIT if (force_refresh or not since_id) else INCREMENTAL_FETCH_LIMIT
            api_limit_val = min(fetch_limit_val, MASTODON_FETCH_LIMIT) # Renamed api_limit
            processed_status_ids = {p["id"] for p in existing_posts if "id" in p} # Ensure 'id' exists
            new_statuses = []

            logger.debug(f"Fetching new statuses for user ID {user_id_val} ({cache_key_username}) (since_id: {since_id})")
            try:
                # Fetch statuses (includes own posts and boosts by the user)
                new_statuses = masto_client.account_statuses(id=user_id_val, limit=api_limit_val, since_id=since_id if not force_refresh else None, exclude_replies=False, exclude_reblogs=False)
                if len(new_statuses) == api_limit_val: # pragma: no cover
                    logger.warning(f"Reached Mastodon API limit ({api_limit_val}) in a single fetch for {cache_key_username}. Some newer posts might be missed if >{api_limit_val} were posted since last check.")
            except MastodonRatelimitError as e: # pragma: no cover
                self._handle_rate_limit("Mastodon", exception=e)
                return None
            except MastodonNotFoundError: # Should not happen if user_info was fetched, but possible race
                raise UserNotFoundError(f"Mastodon user ID {user_id_val} (handle: {username}) not found during status fetch.")
            except (MastodonUnauthorizedError, MastodonVersionError, MastodonError) as e: # pragma: no cover
                err_str = str(e).lower()
                # If account is locked and we are not following, status fetch will be unauthorized
                if user_info and user_info.get("locked") and ("unauthorized" in err_str or "forbidden" in err_str) : # type: ignore
                    logger.warning(f"Cannot fetch statuses for locked Mastodon account {username}.")
                    new_statuses = [] # Treat as no new statuses
                elif "blocked" in err_str or "forbidden" in err_str or "federation" in err_str:
                    raise AccessForbiddenError(f"Access forbidden/blocked fetching Mastodon statuses for {username}.")
                else:
                    logger.error(f"Error fetching statuses for {username}: {e}")
                    raise RuntimeError(f"Mastodon API error during status fetch: {e}")

            logger.info(f"Fetched {len(new_statuses)} new raw statuses for Mastodon user {cache_key_username}.")
            count_added = 0
            for status in new_statuses: # These are already sorted newest first by API
                status_id = str(status["id"])
                if status_id in processed_status_ids: # Should not happen with since_id, but defensive
                    logger.debug(f"Skipping already processed Mastodon status ID: {status_id}")
                    continue

                # Clean HTML content
                cleaned_text = ""
                is_content_warning = bool(status.get("spoiler_text"))
                if is_content_warning:
                    try:
                        cw_soup = BeautifulSoup(status["spoiler_text"], "html.parser")
                        cleaned_cw_text = cw_soup.get_text(separator=" ", strip=True)
                    except Exception as parse_err: # pragma: no cover (should not fail on simple text)
                        logger.warning(f"HTML parsing failed for CW text of status {status_id}: {parse_err}.")
                        cleaned_cw_text = status["spoiler_text"] # Fallback
                    cleaned_text = f"[CW: {cleaned_cw_text}] [Content Hidden]" # LLM should understand this
                else:
                    try:
                        soup = BeautifulSoup(status["content"], "html.parser")
                        # Remove script/style tags, convert <br> to \n, add \n after <p>
                        for script_or_style in soup(["script", "style"]): script_or_style.decompose()
                        for br_tag in soup.find_all("br"): br_tag.replace_with("\n") # Renamed br
                        for p_tag_item in soup.find_all("p"): p_tag_item.append("\n") # Renamed p_tag
                        # Get text, handling multiple lines from <p> and <br> correctly
                        lines = (line.strip() for line in soup.get_text().splitlines())
                        cleaned_text = "\n".join(line for line in lines if line) # Rejoin non-empty lines
                    except Exception as parse_err: # pragma: no cover
                        logger.warning(f"HTML parsing failed for status {status_id}: {parse_err}. Using raw content snippet.")
                        cleaned_text = status["content"][:500] + "..." # Fallback

                media_items_for_post = []
                for attachment in status.get("media_attachments", []):
                    media_url_val = attachment.get("url") # Original URL # Renamed media_url
                    preview_url_val = attachment.get("preview_url") # Preview/thumbnail # Renamed preview_url
                    media_type_val = attachment.get("type", "unknown") # image, video, gifv, audio # Renamed media_type
                    description_val = attachment.get("description") # Alt text # Renamed description
                    remote_url_val = attachment.get("remote_url") # If media is remote # Renamed remote_url

                    url_to_download = media_url_val or preview_url_val # Prefer original if available # Renamed url_to_download
                    if url_to_download:
                        # Pass the specific masto_client if media downloads require auth from that instance
                        # For now, _download_media doesn't have a client pass-through, relies on public URLs or generic auth.
                        # This could be an enhancement if specific instance tokens are needed for media.
                        media_path = self._download_media(url=url_to_download, platform="mastodon", username=cache_key_username)
                        if media_path:
                            analysis = None
                            if media_type_val == "image" and media_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                                image_context = f"Mastodon user {cache_key_username}'s post ({status.get('url', status_id)})"
                                analysis = self._analyze_image(media_path, image_context)
                            media_items_for_post.append({"id": str(attachment.get("id")), "type": media_type_val, "analysis": analysis, "url": media_url_val, "preview_url": preview_url_val, "remote_url": remote_url_val, "description": description_val, "local_path": str(media_path)})
                            if analysis: newly_added_media_analysis.append(analysis)
                            newly_added_media_paths.add(str(media_path))
                        else: logger.warning(f"Failed to download Mastodon media {media_type_val} from {url_to_download} for status {status_id}")

                is_reblog = status.get("reblog") is not None
                reblog_info = status.get("reblog") if is_reblog else None
                reblog_original_author_acct = None
                reblog_original_url = None
                if reblog_info: # reblog_info is a Status dict itself
                    reblog_acct_info = reblog_info.get("account")
                    if reblog_acct_info: reblog_original_author_acct = reblog_acct_info.get("acct")
                    reblog_original_url = reblog_info.get("url")


                tags_list_val = [{"name": tag["name"], "url": tag["url"]} for tag in status.get("tags", [])] # Renamed tags
                mentions_list_val = [{"acct": mention["acct"], "url": mention["url"]} for mention in status.get("mentions", [])] # Renamed mentions
                emojis_list_val = [{"shortcode": emoji["shortcode"], "url": emoji["url"]} for emoji in status.get("emojis", [])] # Renamed emojis

                poll_data = None
                if status.get("poll"):
                    poll_item_val = status["poll"] # Renamed poll
                    poll_options = [{"title": opt["title"], "votes_count": opt.get("votes_count")} for opt in poll_item_val.get("options", [])]
                    poll_data = {"id": str(poll_item_val.get("id")), "expires_at": poll_item_val.get("expires_at"), "expired": poll_item_val.get("expired"), "multiple": poll_item_val.get("multiple"), "votes_count": poll_item_val.get("votes_count"), "voters_count": poll_item_val.get("voters_count"), "options": poll_options}


                post_data = {"id": status_id, "created_at": status["created_at"].isoformat(), # created_at is datetime
                             "url": status["url"],
                             "in_reply_to_id": str(status.get("in_reply_to_id")) if status.get("in_reply_to_id") else None,
                             "in_reply_to_account_id": str(status.get("in_reply_to_account_id")) if status.get("in_reply_to_account_id") else None,
                             "text_html": status["content"], "text_cleaned": cleaned_text[:3000], # Limit cleaned text
                             "spoiler_text": status.get("spoiler_text", ""), "visibility": status.get("visibility"),
                             "sensitive": status.get("sensitive", False), "language": status.get("language"),
                             "reblogs_count": status.get("reblogs_count",0), "favourites_count": status.get("favourites_count",0), "replies_count": status.get("replies_count",0),
                             "is_reblog": is_reblog, "reblog_original_author": reblog_original_author_acct, "reblog_original_url": reblog_original_url,
                             "tags": tags_list_val, "mentions": mentions_list_val, "emojis": emojis_list_val, "poll": poll_data,
                             "media": media_items_for_post}
                new_posts_data.append(post_data)
                processed_status_ids.add(status_id)
                count_added += 1
            logger.info(f"Processed {count_added} new unique statuses for Mastodon user {cache_key_username}.")


            # Combine, sort, limit
            combined_posts = new_posts_data + existing_posts # Newest are first from API
            combined_posts.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True)
            final_posts = combined_posts[:MAX_CACHE_ITEMS]

            valid_new_analyses = [a for a in newly_added_media_analysis if a]
            final_media_analysis = sorted(list(set(valid_new_analyses + [m for m in existing_media_analysis if m])))
            final_media_paths = sorted(list(newly_added_media_paths.union(existing_media_paths)))[:MAX_CACHE_ITEMS*2]

            # Stats
            total_posts = len(final_posts)
            original_posts_list = [p for p in final_posts if not p.get("is_reblog")] # Renamed original_posts
            total_original_posts = len(original_posts_list)
            total_reblogs = total_posts - total_original_posts
            posts_with_media = len([p for p in final_posts if p.get("media")])
            reply_posts_count = len([p for p in final_posts if p.get("in_reply_to_id")])
            avg_favs = sum(p["favourites_count"] for p in original_posts_list) / max(total_original_posts, 1)
            avg_reblogs_calc = sum(p["reblogs_count"] for p in original_posts_list) / max(total_original_posts, 1) # Renamed avg_reblogs
            avg_replies = sum(p["replies_count"] for p in original_posts_list) / max(total_original_posts, 1)

            stats = {"total_posts_cached": total_posts, "total_original_posts_cached": total_original_posts, "total_reblogs_cached": total_reblogs, "total_replies_cached": reply_posts_count,
                     "posts_with_media": posts_with_media, "total_media_items_processed": len(final_media_paths),
                     "avg_favourites_on_originals": round(avg_favs,2), "avg_reblogs_on_originals": round(avg_reblogs_calc,2), "avg_replies_on_originals": round(avg_replies,2)}


            final_data = {"timestamp": datetime.now(timezone.utc).isoformat(), # Handled by _save_cache
                          "user_info": user_info, "posts": final_posts, "media_analysis": final_media_analysis, "media_paths": final_media_paths, "stats": stats}
            self._save_cache("mastodon", cache_key_username, final_data)
            logger.info(f"Successfully updated Mastodon cache for {cache_key_username}. Total posts cached: {total_posts}")
            return final_data

        except ValueError as ve: # From our own validation, e.g. username format
            logger.error(f"Mastodon fetch failed for {username}: {ve}")
            return None
        except RateLimitExceededError: # pragma: no cover
            logger.warning(f"Mastodon fetch for {username} aborted due to rate limit.")
            return None
        except (UserNotFoundError, AccessForbiddenError) as user_err:
            logger.error(f"Mastodon fetch failed for {username}: {user_err}")
            return None
        except RuntimeError as e: # From self.mastodon setup (though less likely now) or other issues
            logger.error(f"Runtime error during Mastodon fetch for {username}: {e}")
            return None
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error fetching Mastodon data for {username}: {str(e)}", exc_info=True)
            return None

    def fetch_hackernews(self, username: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches user activity (stories, comments) from HackerNews via Algolia API."""
        cached_data = self._load_cache("hackernews", username)

        # Offline mode: use cache if available, otherwise return empty structure
        if self.args.offline:
            if cached_data:
                logger.info(f"Offline mode: Using cached data for HackerNews user {username}.")
                # Ensure 'items' key for consistency if loading old cache
                if "submissions" in cached_data and "items" not in cached_data: # pragma: no cover
                    cached_data["items"] = cached_data.pop("submissions")
                return cached_data
            else:
                logger.warning(f"Offline mode: No cache found for HackerNews user {username}. No data will be fetched.")
                return {"timestamp": datetime.now(timezone.utc).isoformat(), "items": [], "stats": {}}

        # Online mode: proceed with normal fetch logic
        if not force_refresh and cached_data and \
           (datetime.now(timezone.utc) - get_sort_key(cached_data, "timestamp")) < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Using recent cache for HackerNews user {username}")
            # Ensure 'items' key from old cache data labeled 'submissions'
            if "submissions" in cached_data and "items" not in cached_data: # pragma: no cover (legacy cache)
                cached_data["items"] = cached_data.pop("submissions")
            return cached_data

        logger.info(f"Fetching HackerNews data for {username} (Force Refresh: {force_refresh})")
        latest_timestamp_i = 0 # Algolia uses `created_at_i` (integer timestamp)
        existing_items = [] # Changed from existing_submissions

        if not force_refresh and cached_data: # Stale cache exists
            logger.info(f"Attempting incremental fetch for HackerNews {username}")
            # Use 'items' key, fallback to 'submissions' for backward compatibility with old cache
            existing_items = cached_data.get("items", cached_data.get("submissions", []))
            if existing_items: # Assumed sorted
                # existing_items.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True)
                try:
                    # Get the created_at_i of the newest item in cache
                    latest_timestamp_i = max(item.get("created_at_i", 0) for item in existing_items if item.get("created_at_i") is not None) # Added None check
                    logger.debug(f"Using latest timestamp_i: {latest_timestamp_i}")
                except ValueError: # pragma: no cover (if items list is empty or no created_at_i)
                    logger.debug("No valid existing items found to determine latest timestamp_i.")
                    latest_timestamp_i = 0


        try:
            # HN Algolia API: https://hn.algolia.com/api
            base_url = "https://hn.algolia.com/api/v1/search" # Not search_by_date
            # Max items to fetch in one go. Algolia default is 20, max seems to be 1000.
            # For incremental, a smaller number is fine. For initial, larger.
            hits_per_page = INITIAL_FETCH_LIMIT if (force_refresh or not latest_timestamp_i) else INCREMENTAL_FETCH_LIMIT

            safe_username = quote_plus(username) # URL-encode username
            params: Dict[str, Any] = {"tags": f"author_{safe_username}", "hitsPerPage": hits_per_page, "typoTolerance": False} # Avoid fuzzy matching on username # Added type hint

            if not force_refresh and latest_timestamp_i > 0:
                # Fetch items created *after* the latest one we have
                params["numericFilters"] = f"created_at_i>{latest_timestamp_i}"
                logger.debug(f"Applying numeric filter: created_at_i > {latest_timestamp_i}")
            # Note: Algolia sorts by relevance by default if no sort order specified.
            # For fetching newest, `search_by_date` endpoint is better: `https://hn.algolia.com/api/v1/search_by_date`
            # and use `tags=story,author_USERNAME` or `tags=comment,author_USERNAME`
            # Let's stick to the general search and rely on numericFilters for incremental.
            # The default sort for `search` can be an issue if user has > hitsPerPage items since last check.
            # However, for typical incremental updates, `created_at_i` filter should work.

            new_items_data = []
            processed_ids = set(item["objectID"] for item in existing_items if "objectID" in item) # Use objectID for uniqueness # Ensure objectID exists

            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.get(base_url, params=params)
                # Check for 400 Bad Request specifically for "invalid tag name"
                if response.status_code == 400: # pragma: no cover
                    try:
                        err_data = response.json()
                        if "message" in err_data:
                            err_msg = err_data["message"]
                            if "invalid tag name" in err_msg.lower():
                                logger.error(f"HN Algolia API Bad Request (400): Invalid tag name likely due to username format '{username}'. Error: {err_msg}")
                                raise UserNotFoundError(f"HackerNews username '{username}' resulted in an invalid API tag (check format).")
                            else:
                                logger.error(f"HN Algolia API Bad Request (400) for {username}: {err_msg}")
                                raise httpx.HTTPStatusError(err_msg, request=response.request, response=response)
                        else: response.raise_for_status() # Re-raise generic 400
                    except json.JSONDecodeError: response.raise_for_status() # Re-raise if not JSON
                else: response.raise_for_status() # For other HTTP errors
                data = response.json()


            if "hits" not in data: # pragma: no cover (API should always return hits array)
                logger.warning(f"No 'hits' found in HN Algolia response for {username}")
                data["hits"] = [] # Ensure hits is an empty list to avoid errors

            if not data["hits"]:
                 # This is common if user has no (new) activity or doesn't exist.
                 # Algolia doesn't throw 404 for non-existent user tags, just 0 hits.
                 logger.info(f"HN Algolia query for '{username}' returned 0 new items. User might have no recent activity or might not exist.")


            logger.info(f"Fetched {len(data.get('hits', []))} potential new items for HN user {username}.")

            for hit in data.get("hits", []):
                object_id = hit.get("objectID")
                if not object_id or not hit.get("created_at_i"): # Essential fields
                    logger.warning(f"Skipping invalid HN hit (missing ID or timestamp): {hit.get('title', object_id)}")
                    continue
                if object_id in processed_ids: # Should be rare if created_at_i filter works
                    logger.debug(f"Skipping already processed HN item ID: {object_id}")
                    continue

                created_at_ts = hit["created_at_i"]
                tags = hit.get("_tags", [])
                item_type = "unknown"
                # Determine item type based on tags or fields
                if "story" in tags and "comment" not in tags: item_type = "story"
                elif "comment" in tags: item_type = "comment"
                elif "poll" in tags and "pollopt" not in tags: item_type = "poll"
                elif "pollopt" in tags: item_type = "pollopt"
                else: # Fallback deduction if tags are minimal (e.g. only author tag)
                    if hit.get("title") and hit.get("url") and "comment" not in tags: item_type = "story"
                    elif hit.get("title") and not hit.get("url") and "comment" not in tags : item_type = "ask_hn_or_job" # Ask HN, Show HN, Job
                    elif hit.get("comment_text") and "comment" not in tags : item_type = "comment" # comment_text implies comment

                # Clean HTML from text fields
                raw_text = hit.get("story_text") or hit.get("comment_text") or ""
                cleaned_text = ""
                if raw_text:
                    try:
                        soup = BeautifulSoup(raw_text, "html.parser")
                        cleaned_text = soup.get_text(separator=" ", strip=True)
                    except Exception as parse_err: # pragma: no cover
                        logger.warning(f"HTML parsing failed for HN item {object_id}: {parse_err}. Using raw snippet.")
                        cleaned_text = raw_text[:500] + "..."


                item_data = {"objectID": object_id, "type": item_type,
                             "title": hit.get("title"), "url": hit.get("url"),
                             "points": hit.get("points"), "num_comments": hit.get("num_comments"),
                             "story_id": hit.get("story_id"), # For comments, refers to parent story
                             "parent_id": hit.get("parent_id"), # For comments, refers to parent item (comment or story)
                             "created_at_i": created_at_ts,
                             "created_at": datetime.fromtimestamp(created_at_ts, tz=timezone.utc).isoformat(),
                             "text": cleaned_text}
                new_items_data.append(item_data)
                processed_ids.add(object_id)


            # Combine, sort, limit
            combined_items = new_items_data + existing_items
            combined_items.sort(key=lambda x: get_sort_key(x, "created_at"), reverse=True)
            final_items = combined_items[:MAX_CACHE_ITEMS]

            # Basic Stats
            story_items = [s for s in final_items if s.get("type") == "story" or s.get("type") == "ask_hn_or_job"] # Use .get
            comment_items = [c for c in final_items if c.get("type") == "comment"] # Use .get
            poll_items = [p for p in final_items if p.get("type") == "poll"] # Excludes pollopts for now, use .get

            total_items = len(final_items)
            total_stories = len(story_items)
            total_comments = len(comment_items)
            total_polls = len(poll_items)

            avg_story_pts = sum(s.get("points", 0) or 0 for s in story_items) / max(total_stories, 1)
            avg_story_num_comments = sum(s.get("num_comments", 0) or 0 for s in story_items) / max(total_stories, 1)
            avg_comment_pts = sum(c.get("points", 0) or 0 for c in comment_items) / max(total_comments, 1)


            stats = {"total_items_cached": total_items, "total_stories_cached": total_stories, "total_comments_cached": total_comments, "total_polls_cached": total_polls,
                     "average_story_points": round(avg_story_pts,2), "average_story_num_comments": round(avg_story_num_comments,2), "average_comment_points": round(avg_comment_pts,2)}

            final_data = {"timestamp": datetime.now(timezone.utc).isoformat(), # Handled by _save_cache
                          "items": final_items, # Changed from 'submissions'
                          "stats": stats}
            self._save_cache("hackernews", username, final_data)
            logger.info(f"Successfully updated HackerNews cache for {username}. Total items cached: {total_items}")
            return final_data

        except httpx.HTTPStatusError as e: # For non-400 Algolia errors
            if e.response.status_code == 429: self._handle_rate_limit("HackerNews (Algolia)", e) # pragma: no cover
            elif e.response.status_code >= 500: # pragma: no cover
                logger.error(f"HN Algolia API server error ({e.response.status_code}) for {username}: {e.response.text[:200]}")
            else: # pragma: no cover
                logger.error(f"HN Algolia API HTTP error for {username}: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except httpx.RequestError as e: # Network errors
            logger.error(f"HN Algolia API network error for {username}: {str(e)}")
            return None
        except UserNotFoundError as e: # From our "invalid tag name" check
            logger.error(f"HN fetch failed for {username}: {e}")
            return None
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error fetching HackerNews data for {username}: {str(e)}", exc_info=True)
            return None

    def analyze(self, platforms: Dict[str, Union[str, List[str]]], query: str) -> str:
        """Collects data from specified platforms and uses LLM to analyze it."""
        collected_text_summaries = []
        all_media_analyzes = [] # Collects all image analysis strings
        failed_fetches = [] # Tuples of (platform, username, reason)
        analysis_start_time = datetime.now(timezone.utc)
        platform_targets: Dict[str, List[str]] = {} # Stores successfully processed platform: [usernames]
        targets_to_process: List[tuple[str, str, str]] = [] # (platform, username, display_name)

        # Prepare list of targets
        for platform, usernames in platforms.items():
            if isinstance(usernames, str): usernames = [usernames]
            if not usernames: continue
            for username in usernames:
                display_name = username
                if platform == "twitter": display_name = f"@{username}"
                elif platform == "reddit": display_name = f"u/{username}"
                # Mastodon and Bluesky use full handles like user@instance or user.bsky.social
                targets_to_process.append((platform, username, display_name))

        if not targets_to_process:
            return "[yellow]No valid platforms or users specified for analysis.[/yellow]"

        total_targets = len(targets_to_process)
        
        # Create a Progress instance specifically for data collection
        collection_progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            # BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", # Optional bar
            transient=True, # Clears when `with` block exits
            console=self.console,
            refresh_per_second=10, # Smooth updates for spinner
        )
        collect_task_id: Optional[TaskID] = None # Explicitly TaskID

        try: # Overall try for the analysis process
            with collection_progress:
                collect_task_id = collection_progress.add_task( # Renamed collect_task
                    f"[cyan]Collecting data for {total_targets} target(s){' (OFFLINE MODE)' if self.args.offline else ''}...",
                    total=total_targets,
                )
                for platform, username, display_name in targets_to_process:
                    fetcher = getattr(self, f"fetch_{platform}", None)
                    if not fetcher:
                        logger.warning(f"No fetcher method found for platform: {platform}")
                        failed_fetches.append((platform, display_name, "Fetcher not implemented"))
                        if collect_task_id is not None and collect_task_id in collection_progress.task_ids:
                             collection_progress.advance(collect_task_id)
                        continue
                    
                    task_desc = f"[cyan]Fetching {platform} for {display_name}{' (cache only)' if self.args.offline else ''}..."
                    if collect_task_id is not None and collect_task_id in collection_progress.task_ids:
                        collection_progress.update(collect_task_id, description=task_desc)
                    
                    data = None
                    try:
                        # force_refresh is False here; if user wants refresh, they use the 'refresh' command in interactive mode
                        # The fetcher itself handles offline logic (i.e., will only use cache if self.args.offline is True)
                        data = fetcher(username=username, force_refresh=False) 
                        if data:
                            # Handle cases where fetcher in offline mode returns a minimal empty structure if no cache
                            is_empty_offline_data = False
                            if self.args.offline:
                                # Check if data seems like the empty placeholder from offline fetchers
                                if platform == "twitter" and data.get("tweets") == [] and not data.get("user_info", {}).get("id"): is_empty_offline_data = True
                                elif platform == "reddit" and data.get("submissions") == [] and data.get("comments") == [] and not data.get("user_profile",{}).get("id"): is_empty_offline_data = True
                                elif platform == "bluesky" and data.get("posts") == [] and not data.get("profile_info",{}).get("did"): is_empty_offline_data = True
                                elif platform == "mastodon" and data.get("posts") == [] and not data.get("user_info",{}).get("id"): is_empty_offline_data = True
                                elif platform == "hackernews" and data.get("items") == [] and not data.get("stats"): is_empty_offline_data = True # HN placeholder is simple
                            
                            if is_empty_offline_data:
                                logger.info(f"Offline mode: No cached data found for {platform}/{display_name}. Skipping summary.")
                                # Ensure this target is added to failed_fetches if not already there from an exception
                                if not any(f[0] == platform and f[1] == display_name for f in failed_fetches): # Avoid double-listing
                                    failed_fetches.append((platform, display_name, "No cached data (Offline Mode)"))
                            else:
                                summary = self._format_text_data(platform, username, data)
                                if summary:
                                    collected_text_summaries.append(summary)
                                    if platform not in platform_targets: platform_targets[platform] = []
                                    platform_targets[platform].append(username) # Store original username
                                else: # pragma: no cover (should always format if data, unless it was an empty placeholder handled above)
                                    logger.warning(f"Failed to format data summary for {platform}/{display_name}. Skipping.")
                                    failed_fetches.append((platform, display_name, "Data formatting failed"))

                                # Collect media analysis if present (this comes from cache, image analysis itself is skipped offline)
                                media_analyses_list = [ma for ma in data.get("media_analysis", []) if isinstance(ma, str) and ma.strip()] # Renamed media_analyses
                                if media_analyses_list: all_media_analyzes.extend(media_analyses_list)
                                logger.info(f"Successfully collected and formatted data for {platform}/{display_name}")
                        else: # Data is None (fetcher failed or offline with no cache and returned None)
                            # If fetcher returned None, and it wasn't due to a caught exception below
                            # that already added to failed_fetches.
                            if not any(f[0] == platform and f[1] == display_name for f in failed_fetches):
                                reason = "No cached data (Offline Mode)" if self.args.offline else "Data fetch failed (check logs)"
                                failed_fetches.append((platform, display_name, reason))
                            logger.warning(f"Data fetch returned None for {platform}/{display_name}")

                    except RateLimitExceededError: # pragma: no cover (now raised by _handle_rate_limit)
                        # This error is now raised by _handle_rate_limit
                        failed_fetches.append((platform, display_name, f"Rate Limited"))
                        # No console print here, _handle_rate_limit does it.
                    except (UserNotFoundError, AccessForbiddenError) as afe:
                        failed_fetches.append((platform, display_name, f"Access Error ({type(afe).__name__})"))
                        self.console.print(f"[yellow]Skipping {platform}/{display_name}: {afe}[/yellow]", highlight=False)
                    except ValueError as ve: # Catch specific ValueErrors like invalid Mastodon username format from fetch_mastodon
                        failed_fetches.append((platform, display_name, f"Input Error ({ve})"))
                        self.console.print(f"[yellow]Skipping {platform}/{display_name}: {ve}[/yellow]", highlight=False)
                    except RuntimeError as rte: # Client init errors, etc.
                        failed_fetches.append((platform, display_name, f"Runtime Error ({rte})"))
                        logger.error(f"Runtime error during fetch for {platform}/{display_name}: {rte}", exc_info=False) # No full stack trace for common runtime issues
                        self.console.print(f"[red]Runtime Error fetching {platform}/{display_name}: {rte}[/red]", highlight=False)
                    except Exception as e: # pragma: no cover (unexpected)
                        fetch_error_msg = f"Unexpected error during fetch for {platform}/{display_name}: {e}"
                        logger.error(fetch_error_msg, exc_info=True)
                        failed_fetches.append((platform, display_name, "Unexpected fetch error"))
                        self.console.print(f"[red]Error fetching {platform}/{display_name}: {e}[/red]", highlight=False)
                    finally:
                        if collect_task_id is not None and collect_task_id in collection_progress.task_ids:
                            collection_progress.advance(collect_task_id)
            
            # After collection progress finishes
            if failed_fetches:
                self.console.print("\n[bold yellow]Data Collection Issues:[/bold yellow]")
                failures_by_reason: Dict[str, List[str]] = {}
                for pf, user, reason in failed_fetches:
                    if reason not in failures_by_reason: failures_by_reason[reason] = []
                    failures_by_reason[reason].append(f"{pf}/{user}")
                for reason, targets in failures_by_reason.items():
                    self.console.print(f"- {reason}: {', '.join(targets)}")
                self.console.print("[yellow]Analysis will proceed with available data.[/yellow]\n")


            if not collected_text_summaries and not all_media_analyzes:
                # If nothing was collected at all (e.g. all targets failed)
                if not platform_targets and failed_fetches: # platform_targets empty implies no successful fetches
                    return "[red]Data collection failed for all targets. Analysis cannot proceed.[/red]"
                elif not platform_targets: # No targets were even attempted or all failed silently
                    return "[red]No data successfully collected or formatted. Analysis cannot proceed.[/red]"
                else: # Some data might exist but formatting failed or was empty # pragma: no cover
                     logger.warning("Proceeding to analysis with potentially limited data (no text summaries or no media analyzes).")


            # Deduplicate media analysis strings
            unique_media_analyzes = sorted(list(set(all_media_analyzes)))

            analysis_components = []
            image_model = os.getenv("IMAGE_ANALYSIS_MODEL")
            text_model = os.getenv("ANALYSIS_MODEL")


            if unique_media_analyzes:
                media_summary = f"## Consolidated Media Analysis (using {image_model}):\n\n*Note: The following are objective descriptions based on visual content analysis. Some image analysis may have been skipped if operating in offline mode or if images were unavailable.*\n\n"
                media_summary += "\n\n".join(f"### Image Analysis {i + 1}\n{analysis.strip()}" for i, analysis in enumerate(unique_media_analyzes))
                analysis_components.append(media_summary)
                logger.debug(f"Added {len(unique_media_analyzes)} unique media analyzes to prompt.")
            else: logger.debug("No media analysis results to add to prompt.")


            if collected_text_summaries:
                text_summary = f"## Collected Textual & Activity Data Summary:\n\n"
                text_summary += "\n\n---\n\n".join([s.strip() for s in collected_text_summaries]) # Ensure each summary is stripped
                analysis_components.append(text_summary)
                logger.debug(f"Added {len(collected_text_summaries)} platform text summaries to prompt.")
            else: logger.debug("No text summaries to add to prompt.")


            if not analysis_components: # Should be caught by earlier checks, but defensive
                return "[yellow]No text or media data available to send for analysis after collection phase.[/yellow]"

            # System prompt for the LLM
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
*   **Clarity & Structure:** Use clear language. Employ formatting (markdown headings, bullet points) to organize the response logically, often starting with a direct answer to the query followed by supporting evidence/analysis. If data collection failed for some targets, mention this early on.

**Output:** A structured analytical report that directly addresses the user's query, rigorously supported by evidence from the provided text and image data, adhering to all constraints. Start with a summary answer, then elaborate with details structured using relevant analysis domains. If data collection failed for some targets, mention this early on.
"""
            # Add note to system prompt if running in offline mode
            if self.args.offline:
                system_prompt += "\n\n**Operational Note:** You are operating in OFFLINE mode. All provided data comes from a local cache. No new information has been fetched from social media platforms for this analysis run. Image analysis for *any newly encountered* images was skipped. Interpret missing data or lack of very recent data accordingly."

            user_prompt = f"**Analysis Query:** {query}\n\n**Provided Data:**\n\n" + "\n\n===\n\n".join(analysis_components)
            llm_api_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            llm_progress = Progress( # New Progress instance for LLM call
                SpinnerColumn(),
                "[progress.description]{task.description}",
                transient=True,
                console=self.console,
                refresh_per_second=10,
            )
            analysis_task_llm_id: Optional[TaskID] = None # Renamed analysis_task
            with llm_progress:
                analysis_task_llm_id = llm_progress.add_task(f"[magenta]Analyzing with {text_model}...", total=None) # Indeterminate
                try:
                    self._llm_completion_object = None # Reset before call
                    self._llm_api_exception = None

                    # Use threading for non-blocking spinner
                    api_thread = threading.Thread(target=self._call_llm_api, kwargs={
                        "model_name": cast(str, text_model), # Cast model_name to str # Added cast
                        "messages": llm_api_messages, # type: ignore
                        "max_tokens": 3500, # Set a high max_tokens for the response
                        "temperature": 0.5 # Use a moderate temperature
                    })
                    api_thread.start()
                    while api_thread.is_alive():
                        # Check if task still exists before updating (can be removed on error)
                        if analysis_task_llm_id is not None and analysis_task_llm_id in llm_progress.task_ids:
                             llm_progress.update(analysis_task_llm_id, description=f"[magenta]Analyzing with {text_model} (waiting for API)...")
                        api_thread.join(0.1) # Check every 100ms, allows spinner to update
                    
                    if self._llm_api_exception: # Exception was set in _call_llm_api
                        err_details = str(self._llm_api_exception)
                        original_exception = self._llm_api_exception # Store original exception
                        if isinstance(self._llm_api_exception, APIError): # More detailed handling for OpenAI specific errors
                            api_err = self._llm_api_exception
                            status_code = api_err.status_code if hasattr(api_err, 'status_code') else "N/A"
                            err_name = type(api_err).__name__
                            err_msg_parts = [f"LLM API Error ({err_name}, Status: {status_code})"]
                            # Try to extract more detailed error messages from the error object or response body
                            msg_from_err_obj = getattr(api_err, 'message', None)
                            if msg_from_err_obj: err_msg_parts.append(str(msg_from_err_obj))
                            
                            # Check response body for error message
                            if hasattr(api_err, 'body') and isinstance(api_err.body, dict):
                                body_err = api_err.body.get('error')
                                if isinstance(body_err, dict) and 'message' in body_err:
                                    err_msg_parts.append(str(body_err['message']))
                                elif isinstance(body_err, str): # Sometimes error is just a string
                                     err_msg_parts.append(body_err)
                                elif 'message' in api_err.body: # Root level message
                                     err_msg_parts.append(str(api_err.body['message']))

                            err_details = ": ".join(list(dict.fromkeys(filter(None, err_msg_parts)))) # Unique, ordered parts
                            logger.error(f"Analysis LLM API Error: {err_details}")
                            model_used = text_model # For consistency in logging error messages
                            if isinstance(api_err, RateLimitError):
                                 self._handle_rate_limit(f"LLM Text Analysis ({text_model})", api_err) # This will re-raise
                            elif isinstance(api_err, AuthenticationError):
                                logger.error(f"LLM API Authentication Error (401) for model {model_used}. Check your LLM_API_KEY.")
                            elif isinstance(api_err, BadRequestError):
                                logger.error(f"LLM API Bad Request (400) for model {model_used}. Often due to invalid input (e.g. prompt too long).")
                            # Other APIError types will fall through to the general RuntimeError raise below
                        else:
                             logger.error(f"Analysis LLM API request failed with general exception: {err_details}", exc_info=self._llm_api_exception)
                        # Raise a standard RuntimeError with helpful message, chaining the original exception
                        raise RuntimeError(f"Analysis LLM API request failed: {err_details}") from original_exception
                    
                    if not self._llm_completion_object: # Should not happen if thread completed successfully
                        raise RuntimeError("LLM API call completed but no completion object was captured.")

                    completion = self._llm_completion_object
                    if not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
                        logger.error(f"Invalid analysis API response format or empty content: {completion.model_dump_json(indent=2)}")
                        return "[red]Analysis failed: Invalid response format or empty content from API.[/red]"

                    analysis_content = completion.choices[0].message.content
                    # Construct final report with metadata
                    targets_str = ", ".join(sorted(platform_targets.keys()))
                    report_timestamp = analysis_start_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                    final_report = (
                        f"# OSINT Analysis Report\n\n"
                        f"**Query:** {query}\n"
                        f"**Targets Queried:** {targets_str}\n" # Use successfully fetched platform names
                        f"**Report Generated:** {report_timestamp}\n"
                        f"**Mode:** {'Offline (Cached Data Only)' if self.args.offline else 'Online'}\n" # Add mode to report
                        f"**Models Used:**\n- Text Analysis: `{text_model}`\n- Image Analysis: `{image_model}`\n\n"
                    )
                    if failed_fetches: # Add a note if some data collection failed
                        final_report += f"**Data Collection Issues:** Data collection failed or was limited for some targets (see logs or previous messages). This is expected for some targets in offline mode if no cache existed.\n\n"
                    final_report += "---\n\n" + analysis_content.strip()
                    return final_report
                finally: # Ensure progress bar is cleaned up
                    if analysis_task_llm_id is not None and analysis_task_llm_id in llm_progress.task_ids:
                        llm_progress.update(analysis_task_llm_id, visible=False) # Hide before removing
                        llm_progress.remove_task(analysis_task_llm_id)
                    # Clear shared state for next potential call
                    self._llm_completion_object = None
                    self._llm_api_exception = None

        except RateLimitExceededError as rle: # pragma: no cover (if rate limit happens during collection and is re-raised)
            # This is if _handle_rate_limit was called by a fetcher
            return f"[red]Analysis aborted due to rate limiting during data collection: {rle}[/red]"
        except RuntimeError as run_err: # From API call failure or other programmatic issues
            # self.console.print(f"[bold red]Analysis Failed:[/bold red] {run_err}") # Already printed or will be by caller
            return f"[red]Analysis failed: {run_err}[/red]"
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error during analysis phase: {str(e)}", exc_info=True)
            return f"[red]Analysis failed due to unexpected error: {str(e)}[/red]"

    def _format_text_data(self, platform: str, username: str, data: dict) -> str: # username is original input
        """Formats fetched data into a concise text summary for the LLM."""
        MAX_ITEMS_PER_TYPE = 25 # Max items (tweets, posts, comments) to include in summary
        TEXT_SNIPPET_LENGTH = 750 # Max characters for text snippets
        if not data: return ""
        # Check for minimal placeholder data returned by offline fetchers when cache is empty
        # This prevents trying to format an empty shell as if it were real data.
        if self.args.offline:
            # These checks look for the specific empty structures returned by fetchers in offline mode when no cache is found.
            if platform == "twitter" and data.get("tweets") == [] and not data.get("user_info", {}).get("id"): return ""
            if platform == "reddit" and data.get("submissions") == [] and data.get("comments") == [] and not data.get("user_profile",{}).get("id"): return ""
            if platform == "bluesky" and data.get("posts") == [] and not data.get("profile_info",{}).get("did"): return ""
            if platform == "mastodon" and data.get("posts") == [] and not data.get("user_info",{}).get("id"): return ""
            if platform == "hackernews" and data.get("items") == [] and not data.get("stats"): return "" # HN placeholder is simpler

        output_lines = []

        platform_display_name = platform.capitalize()
        user_prefix = ""
        display_username = username # Default to original input username

        # Get user info from various possible keys (handles variations across platforms)
        user_info = data.get("user_info") or data.get("profile_info") or data.get("user_profile")

        # Set prefix and display username based on platform and available info
        if platform == "twitter":
            user_prefix = "@"
            display_username = user_info.get("username", username) if user_info else username
        elif platform == "reddit":
            user_prefix = "u/"
            display_username = user_info.get("name", username) if user_info else username
        elif platform == "mastodon": # acct is user@instance
            display_username = user_info.get("acct", username) if user_info else username
        elif platform == "bluesky": # handle is user.bsky.social
            display_username = user_info.get("handle", username) if user_info else username
        # HackerNews uses just the username

        output_lines.append(f"### {platform_display_name} Data Summary for: {user_prefix}{display_username}")

        # Add user profile information if available and not empty
        if user_info and user_info.get("id", user_info.get("did")): # Check for a core identifier
            output_lines.append("\n**User Profile:**")
            # Safely get and format creation date, handling different key names and types
            created_at_dt = get_sort_key(user_info, "created_at") or get_sort_key(user_info, "created_utc")
            created_at_str = created_at_dt.strftime("%Y-%m-%d") if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else "N/A"

            output_lines.append(f"- Handle/Username: `{user_prefix}{display_username}`")
            if "id" in user_info: output_lines.append(f"- ID: `{user_info['id']}`")
            # For Bluesky, DID is the primary ID, handle is secondary
            if "did" in user_info and "id" not in user_info : output_lines.append(f"- DID: `{user_info['did']}`")
            if "display_name" in user_info and user_info["display_name"] != display_username:
                 output_lines.append(f"- Display Name: '{user_info['display_name']}'")
            output_lines.append(f"- Account Created: {created_at_str}")

            if platform == "twitter":
                output_lines.append(f"- Verified: {user_info.get('verified', 'N/A')}")
                if user_info.get("location"): output_lines.append(f"- Location: {user_info['location']}")
                if user_info.get("description"): output_lines.append(f"- Description: {user_info['description'][:200] + ('...' if len(user_info['description']) > 200 else '')}") # Truncate long descriptions
                pm = user_info.get("public_metrics", {})
                output_lines.append(f"- Stats: Followers={pm.get('followers_count','N/A')}, Following={pm.get('following_count','N/A')}, Tweets={pm.get('tweet_count','N/A')}")
            elif platform == "reddit":
                output_lines.append(f"- Karma: Link={user_info.get('link_karma','N/A')}, Comment={user_info.get('comment_karma','N/A')}")
                output_lines.append(f"- Suspended: {user_info.get('is_suspended','N/A')}")
                if user_info.get("icon_img"): output_lines.append(f"- Icon URL: {user_info['icon_img']}")
            elif platform == "bluesky":
                if user_info.get("description"): output_lines.append(f"- Bio: {user_info['description'][:200] + ('...' if len(user_info['description']) > 200 else '')}") # Truncate long bios
                pm = user_info # Bluesky profile info is flat, use user_info directly for counts
                output_lines.append(f"- Stats: Posts={pm.get('posts_count','N/A')}, Following={pm.get('follows_count','N/A')}, Followers={pm.get('followers_count','N/A')}")
                if user_info.get("labels"): output_lines.append(f"- Labels: {', '.join(l['value'] for l in user_info['labels'])}")
            elif platform == "mastodon":
                output_lines.append(f"- Locked Account: {user_info.get('locked', 'N/A')}")
                output_lines.append(f"- Bot Account: {user_info.get('bot', 'N/A')}")
                if user_info.get("note_text"): output_lines.append(f"- Bio: {user_info['note_text'][:200] + ('...' if len(user_info['note_text']) > 200 else '')}") # Truncate long bios
                pm = user_info # Mastodon user_info contains counts
                output_lines.append(f"- Stats: Followers={pm.get('followers_count','N/A')}, Following={pm.get('following_count','N/A')}, Posts={pm.get('statuses_count','N/A')}")
                if user_info.get("custom_fields"):
                    fields_str = ", ".join([f"{f['name']}: {f['value'][:50]}" for f in user_info["custom_fields"]]) # Truncate field values
                    output_lines.append(f"- Profile Metadata: {fields_str}")
        elif not user_info and platform != "hackernews": # HackerNews has no specific user profile block from its API
            output_lines.append("\n**User Profile:**\n- Profile information not available in cache.")


        # Add aggregated stats if available and not empty
        stats = data.get("stats", {})
        if stats: # Only add this section if stats dict is present and non-empty
            output_lines.append("\n**Cached Activity Overview:**")
            stat_items = []
            if platform == "reddit": stat_items.extend([f"Subs={stats.get('total_submissions_cached',0)}", f"Comments={stats.get('total_comments_cached',0)}", f"Media Posts={stats.get('submissions_with_media',0)}", f"Avg Sub Score={stats.get('avg_submission_score',0):.1f}", f"Avg Comment Score={stats.get('avg_comment_score',0):.1f}"])
            elif platform == "bluesky": stat_items.extend([f"Posts={stats.get('total_posts_cached',0)}", f"Media Posts={stats.get('posts_with_media',0)}", f"Replies={stats.get('reply_posts_cached',0)}", f"Avg Likes={stats.get('avg_likes',0):.1f}", f"Avg Reposts={stats.get('avg_reposts',0):.1f}"])
            elif platform == "mastodon": stat_items.extend([f"Posts={stats.get('total_posts_cached',0)}", f"Originals={stats.get('total_original_posts_cached',0)}", f"Boosts={stats.get('total_reblogs_cached',0)}", f"Replies={stats.get('total_replies_cached',0)}", f"Media Posts={stats.get('posts_with_media',0)}", f"Avg Favs (Orig)={stats.get('avg_favourites_on_originals',0):.1f}", f"Avg Boosts (Orig)={stats.get('avg_reblogs_on_originals',0):.1f}"])
            elif platform == "hackernews": stat_items.extend([f"Items={stats.get('total_items_cached',0)}", f"Stories={stats.get('total_stories_cached',0)}", f"Comments={stats.get('total_comments_cached',0)}", f"Avg Story Pts={stats.get('average_story_points',0):.1f}", f"Avg Comment Pts={stats.get('average_comment_points',0):.1f}"])
            if stat_items: output_lines.append(f"- {', '.join(stat_items)}")
            if stats.get("total_media_items_processed") is not None: output_lines.append(f"- Total Media Items Processed (in cache): {stats.get('total_media_items_processed')}")
        elif platform != "twitter": # Twitter user_info contains public_metrics directly, no separate 'stats' block
             output_lines.append("\n**Cached Activity Overview:**\n- Aggregated stats not available in cache.")


        # Add recent activity items, truncated
        if platform == "twitter":
            tweets = data.get("tweets", [])
            output_lines.append(f"\n**Recent Tweets (up to {MAX_ITEMS_PER_TYPE}):**")
            if not tweets: output_lines.append("- No tweets found in cached data.")
            else:
                for i, t_item in enumerate(tweets[:MAX_ITEMS_PER_TYPE]): # Renamed t
                    created_at_dt = get_sort_key(t_item, "created_at")
                    created_at_str = created_at_dt.strftime("%Y-%m-%d %H:%M") if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else "N/A"
                    media_count = len(t_item.get("media", [])); media_info = f" (Media: {media_count})" if media_count > 0 else ""
                    ref_info = "" # Info about replies/quotes
                    if t_item.get("in_reply_to_user_id"): ref_info += " (Reply)"
                    if any(ref["type"] == "quoted" for ref in t_item.get("referenced_tweets",[])): ref_info += " (Quote Tweet)"
                    text = t_item.get("text", "[No Text]"); text_snippet = text[:TEXT_SNIPPET_LENGTH] + ("..." if len(text) > TEXT_SNIPPET_LENGTH else "") # Truncate text
                    metrics = t_item.get("metrics", {})
                    output_lines.append(f"- Tweet {i+1} ({created_at_str}){ref_info}{media_info}:\n  Content: {text_snippet}\n  Metrics: Likes={metrics.get('like_count',0)}, RTs={metrics.get('retweet_count',0)}, Replies={metrics.get('reply_count',0)}, Quotes={metrics.get('quote_count',0)}")
        elif platform == "reddit":
            submissions = data.get("submissions", [])
            output_lines.append(f"\n**Recent Submissions (up to {MAX_ITEMS_PER_TYPE}):**")
            if not submissions: output_lines.append("- No submissions found.")
            else:
                for i, s_item in enumerate(submissions[:MAX_ITEMS_PER_TYPE]): # Renamed s
                    created_at_dt = get_sort_key(s_item, "created_utc")
                    created_at_str = created_at_dt.strftime("%Y-%m-%d %H:%M") if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else "N/A"
                    media_count = len(s_item.get("media", [])); media_info = f" (Media: {media_count})" if media_count > 0 else ""
                    nsfw_info = " (NSFW)" if s_item.get("over_18") else ""; spoiler_info = " (Spoiler)" if s_item.get("spoiler") else ""
                    text = s_item.get("text", ""); text_snippet = text[:TEXT_SNIPPET_LENGTH] + ("..." if len(text) > TEXT_SNIPPET_LENGTH else "") # Truncate selftext
                    text_info = f"\n  Self-Text: {text_snippet}" if text_snippet else ""
                    link_info = f"\n  Link URL: {s_item.get('link_url')}" if s_item.get("link_url") else ""
                    output_lines.append(f"- Submission {i+1} in r/{s_item.get('subreddit','N/A')} ({created_at_str}):{media_info}{nsfw_info}{spoiler_info}\n  Title: {s_item.get('title', '[No Title]')}\n  Score: {s_item.get('score',0)} (Ratio: {s_item.get('upvote_ratio','N/A')}), Comments: {s_item.get('num_comments','N/A')}{link_info}{text_info}")

            comments = data.get("comments", [])
            output_lines.append(f"\n**Recent Comments (up to {MAX_ITEMS_PER_TYPE}):**")
            if not comments: output_lines.append("- No comments found.")
            else:
                for i, c_item in enumerate(comments[:MAX_ITEMS_PER_TYPE]): # Renamed c
                    created_at_dt = get_sort_key(c_item, "created_utc")
                    created_at_str = created_at_dt.strftime("%Y-%m-%d %H:%M") if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else "N/A"
                    text = c_item.get("text", "[No Text]"); text_snippet = text[:TEXT_SNIPPET_LENGTH] + ("..." if len(text) > TEXT_SNIPPET_LENGTH else "") # Truncate comment text
                    submitter_info = " (OP)" if c_item.get("is_submitter") else ""
                    output_lines.append(f"- Comment {i+1} in r/{c_item.get('subreddit','N/A')} ({created_at_str}){submitter_info}:\n  Content: {text_snippet}\n  Score: {c_item.get('score',0)}, Link: {c_item.get('permalink','N/A')}")
        elif platform == "hackernews":
            items = data.get("items", []) # Use 'items' key now
            output_lines.append(f"\n**Recent Activity (Stories & Comments, up to {MAX_ITEMS_PER_TYPE}):**")
            if not items: output_lines.append("- No activity found.")
            else:
                for i, item_val in enumerate(items[:MAX_ITEMS_PER_TYPE]): # Renamed item
                    created_at_dt = get_sort_key(item_val, "created_at")
                    created_at_str = created_at_dt.strftime("%Y-%m-%d %H:%M") if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else "N/A"
                    item_type = item_val.get("type", "unknown").capitalize()
                    title = item_val.get("title"); text = item_val.get("text", ""); text_snippet = text[:TEXT_SNIPPET_LENGTH] + ("..." if len(text) > TEXT_SNIPPET_LENGTH else "") # Truncate text
                    hn_link = f"https://news.ycombinator.com/item?id={item_val.get('objectID')}" # Direct link

                    output_lines.append(f"- Item {i+1} ({item_type}, {created_at_str}):")
                    if title: output_lines.append(f"  Title: {title}")
                    if item_val.get("url"): output_lines.append(f"  URL: {item_val.get('url')}")
                    if text_snippet: output_lines.append(f"  Text: {text_snippet}")
                    points = item_val.get("points"); num_comments = item_val.get("num_comments"); stats_parts = []
                    if points is not None: stats_parts.append(f"Pts={points}")
                    if item_type == "Story" and num_comments is not None: stats_parts.append(f"Comments={num_comments}")
                    if stats_parts: output_lines.append(f"  Stats: {', '.join(stats_parts)}")
                    output_lines.append(f"  HN Link: {hn_link}")
                    if item_type == "Comment" and item_val.get("story_id"):
                         output_lines.append(f"  Parent Story: https://news.ycombinator.com/item?id={item_val['story_id']}")
        elif platform == "bluesky":
            posts = data.get("posts", [])
            output_lines.append(f"\n**Recent Posts (up to {MAX_ITEMS_PER_TYPE}):**")
            if not posts: output_lines.append("- No posts found.")
            else:
                for i, p_item in enumerate(posts[:MAX_ITEMS_PER_TYPE]): # Renamed p
                    created_at_dt = get_sort_key(p_item, "created_at")
                    created_at_str = created_at_dt.strftime("%Y-%m-%d %H:%M") if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else "N/A"
                    media_count = len(p_item.get("media", [])); media_info = f" (Media: {media_count})" if media_count > 0 else ""
                    text = p_item.get("text", "[No Text]"); text_snippet = text[:TEXT_SNIPPET_LENGTH] + ("..." if len(text) > TEXT_SNIPPET_LENGTH else "") # Truncate text
                    embed_type = p_item.get("embed_type"); embed_desc = f" (Embed: {embed_type.split('.')[-1]})" if embed_type and isinstance(embed_type, str) else "" # Show last part of type # Added isinstance check
                    reply_info = " (Reply)" if p_item.get("reply_parent") else ""
                    langs_info = f" (Lang: {','.join(p_item.get('langs',[]))})" if p_item.get("langs") else ""
                    output_lines.append(f"- Post {i+1} ({created_at_str}):{reply_info}{media_info}{embed_desc}{langs_info}\n  Content: {text_snippet}\n  Stats: Likes={p_item.get('likes',0)}, Reposts={p_item.get('reposts',0)}, Replies={p_item.get('reply_count',0)}\n  URI: {p_item.get('uri','N/A')}")
        elif platform == "mastodon":
            posts = data.get("posts", [])
            output_lines.append(f"\n**Recent Posts (Toots & Boosts, up to {MAX_ITEMS_PER_TYPE}):**")
            if not posts: output_lines.append("- No posts found in fetched data.")
            else:
                for i, p_item in enumerate(posts[:MAX_ITEMS_PER_TYPE]): # Renamed p
                    created_at_dt = get_sort_key(p_item, "created_at")
                    created_at_str = created_at_dt.strftime("%Y-%m-%d %H:%M") if created_at_dt > datetime.min.replace(tzinfo=timezone.utc) else "N/A"
                    media_count = len(p_item.get("media", [])); media_info = f" (Media: {media_count})" if media_count > 0 else ""
                    cw_text = p_item.get("spoiler_text", ""); cw_info = f" (CW: {cw_text[:50]}{'...' if len(cw_text)>50 else ''})" if cw_text else "" # Truncate CW text in info
                    is_boost = p_item.get("is_reblog", False); boost_info = f" (Boost of {p_item.get('reblog_original_author', 'unknown')})" if is_boost else ""
                    reply_info = " (Reply)" if p_item.get("in_reply_to_id") else ""
                    visibility = p_item.get("visibility", "public"); vis_info = f" ({visibility.capitalize()})" if visibility != "public" else ""
                    lang_info = f" (Lang: {p_item.get('language')})" if p_item.get("language") else ""

                    text_snippet = p_item.get("text_cleaned", "") # Use pre-cleaned text
                    # Determine what to display for content based on CW/Boost
                    text_display = text_snippet[:TEXT_SNIPPET_LENGTH] + ("..." if len(text_snippet) > TEXT_SNIPPET_LENGTH else "") # Truncate content text
                    if cw_text and "[Content Hidden]" in text_snippet: # If it's a CW post, text_cleaned might just be "[CW...] [Content Hidden]"
                        text_display = text_snippet # Show the CW marker
                    elif is_boost and not text_snippet.strip(): # Boosts might have no local text_cleaned if it was just a pure boost
                        text_display = "[Boost Content Only - See Original]"
                    elif not text_display.strip(): # If no text, CW, or boost with content
                        text_display = "[No Text Content]"


                    output_lines.append(f"- Post {i+1} ({created_at_str}):{boost_info}{reply_info}{vis_info}{cw_info}{media_info}{lang_info}\n  Content: {text_display}\n  Stats: Favs={p_item.get('favourites_count',0)}, Boosts={p_item.get('reblogs_count',0)}, Replies={p_item.get('replies_count',0)}\n  Link: {p_item.get('url','N/A')}")
                    if p_item.get("tags"): output_lines.append(f"  Tags: {', '.join(['#' + t['name'] for t in p_item['tags']])}")
                    if p_item.get("poll"):
                        poll = p_item["poll"]
                        options_str = ", ".join([f"'{opt['title']}' ({opt.get('votes_count','?')})" for opt in poll.get("options",[])])
                        output_lines.append(f"  Poll ({poll.get('votes_count','?')} votes): {options_str}")
                    if is_boost and p_item.get("reblog_original_url"):
                        output_lines.append(f"  Original Post: {p_item['reblog_original_url']}")


        else: # Fallback for unknown platform types (should not happen with current structure) # pragma: no cover
            output_lines.append(f"\n**Raw Data Preview (Unknown Platform Type):**")
            output_lines.append(f"- {str(data)[:TEXT_SNIPPET_LENGTH]}...") # Truncate raw data preview

        return "\n".join(output_lines)

    def _call_llm_api(self, model_name: str, messages: list, max_tokens: int, temperature: float):
        """Helper method to make the LLM API call, designed to be run in a thread."""
        self._llm_completion_object = None
        self._llm_api_exception = None
        try:
            client = self.llm_client
            logger.debug(f"Sending LLM API request to model: {model_name} via {client.base_url}") # type: ignore
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages, # type: ignore # messages is List[ChatCompletionMessageParam]
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.debug(f"Received successful response from LLM API for model {model_name}.")
            self._llm_completion_object = completion
        except APIError as e: # Catch specific OpenAI API errors
            logger.error(f"LLM API call error in thread for model {model_name}: {str(e)}")
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'text'): # pragma: no cover
                response_text = e.response.text[:500]
                status_code_val = e.status_code if hasattr(e, 'status_code') else 'N/A' # Renamed status_code
                logger.error(f"Response status: {status_code_val}. Response body snippet: {response_text}")
            self._llm_api_exception = e
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error in LLM API call thread for model {model_name}: {str(e)}", exc_info=True)
            self._llm_api_exception = e

    def _save_output(self, content: str, query: str, platforms_analyzed: List[str], format_type: str = "markdown"):
        """Saves the analysis report to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize query and platforms for filename
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:30]).strip("_") or "query"
        safe_platforms = "_".join(sorted(platforms_analyzed))[:30].strip("_") or "platforms"
        offline_suffix = "_offline" if self.args.offline else ""
        filename_base = f"analysis_{timestamp}_{safe_platforms}_{safe_query}{offline_suffix}"


        # Extract metadata from the report content if possible (Markdown format)
        report_lines = content.splitlines()
        metadata = { # Default metadata
            "query": query,
            "platforms_analyzed": sorted(platforms_analyzed),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "text_model": os.getenv("ANALYSIS_MODEL"),
            "image_model": os.getenv("IMAGE_ANALYSIS_MODEL"),
            "output_format": format_type,
            "mode": "Offline (Cached Data Only)" if self.args.offline else "Online", # Add mode to metadata
        }
        report_content_md = content # Default to full content

        try:
            if report_lines and report_lines[0].strip() == "# OSINT Analysis Report": # Check if report_lines is not empty
                header_lines = []
                content_start_index = 1 # After the main title
                for i, line in enumerate(report_lines[1:], 1): # Start from second line
                    if line.strip() == "---": # End of metadata block
                        content_start_index = i + 1
                        break
                    header_lines.append(line)
                
                # Parse header lines for metadata
                for line in header_lines:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        key_clean = key.strip().lower().replace(" ", "_").replace(":", "") # Sanitize key # Renamed key
                        val_clean = val.strip() # Renamed val
                        if key_clean == "query": metadata["query"] = val_clean
                        elif key_clean == "targets_queried": metadata["platforms_analyzed"] = sorted([p.strip() for p in val_clean.split(",")])
                        elif key_clean == "report_generated": metadata["timestamp_utc"] = val_clean # Assuming it's UTC parseable
                        elif key_clean == "text_analysis": metadata["text_model"] = val_clean.strip("`")
                        elif key_clean == "image_analysis": metadata["image_model"] = val_clean.strip("`")
                        elif key_clean == "mode": metadata["mode"] = val_clean # Capture mode from report too
                # The actual LLM content starts after the "---"
                report_content_md = "\n".join(report_lines[content_start_index:]).strip()
        except IndexError: # pragma: no cover (if report is too short or malformed for this parsing)
            logger.warning("Could not parse metadata from report header, using defaults.")
            report_content_md = content # Use full content as report body

        try:
            filename: Optional[Path] = None # Initialize filename variable
            if format_type == "json":
                filename = output_dir / f"{filename_base}.json"
                data_to_save = {
                    "analysis_metadata": metadata,
                    "analysis_report_markdown": report_content_md # Store the LLM output as markdown
                }
                filename.write_text(json.dumps(data_to_save, indent=2, cls=DateTimeEncoder), encoding="utf-8")
            else: # markdown
                filename = output_dir / f"{filename_base}.md"
                # Reconstruct Markdown with YAML frontmatter
                md_metadata = "---\n"
                md_metadata += f"Query: {metadata['query']}\n"
                md_metadata += f"Platforms: {', '.join(metadata['platforms_analyzed'])}\n"
                md_metadata += f"Timestamp: {metadata['timestamp_utc']}\n"
                md_metadata += f"Mode: {metadata['mode']}\n" # Include mode in Markdown frontmatter
                md_metadata += f"Text Model: {metadata['text_model']}\n"
                md_metadata += f"Image Model: {metadata['image_model']}\n"
                md_metadata += "---\n\n"
                full_content = md_metadata + report_content_md
                filename.write_text(full_content, encoding="utf-8")

            if filename: # Check if filename was set
              self.console.print(f"[green]Analysis saved to: {filename}[/green]")
            else: # Should not happen if format_type is one of the choices
              self.console.print(f"[bold red]Failed to determine filename for saving output.[/bold red]")

        except Exception as e: # pragma: no cover
            self.console.print(f"[bold red]Failed to save output: {str(e)}[/bold red]")
            logger.error(f"Failed to save output file {filename_base}: {e}", exc_info=True)

    def get_available_platforms(self, check_creds=True) -> List[str]:
        """Determines which platforms are available based on environment variables and config files."""
        available = []
        # Twitter
        if not check_creds or all(os.getenv(k) for k in ["TWITTER_BEARER_TOKEN"]):
            available.append("twitter")
        # Reddit
        if not check_creds or all(os.getenv(k) for k in ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]):
            available.append("reddit")
        # Bluesky
        if not check_creds or all(os.getenv(k) for k in ["BLUESKY_IDENTIFIER", "BLUESKY_APP_SECRET"]):
            available.append("bluesky")
        
        # Mastodon - check for config file existence and basic validity
        # self.mastodon_config_file_path_str is set in __init__
        path_relative_to_base_dir = self.base_dir / self.mastodon_config_file_path_str
        path_as_provided = Path(self.mastodon_config_file_path_str)
        
        actual_config_path: Optional[Path] = None
        if path_relative_to_base_dir.is_file():
            actual_config_path = path_relative_to_base_dir
        elif path_as_provided.is_file():
            actual_config_path = path_as_provided

        if not check_creds: # If not checking credentials, assume Mastodon is conceptually available
             available.append("mastodon")
        elif actual_config_path: # Config file exists, now check its content if creds check is on
            try:
                # Basic check: can we load it as JSON and is it a list with at least one valid-looking entry?
                with open(actual_config_path, "r", encoding="utf-8") as f:
                    conf_data = json.load(f)
                if isinstance(conf_data, list) and any(
                    isinstance(item, dict) and item.get("api_base_url") and item.get("access_token")
                    for item in conf_data if isinstance(item, dict) # Ensure item is dict before .get
                ):
                    available.append("mastodon")
                elif check_creds: # Log issue if creds check is on and file content is bad
                    logger.warning(f"Mastodon config file '{actual_config_path}' found but appears malformed or empty of valid instances. Mastodon might be unavailable for operations.")
            except (json.JSONDecodeError, OSError) as e:
                if check_creds: # Log issue if creds check is on and file is unreadable
                    logger.warning(f"Could not read or parse Mastodon config file '{actual_config_path}': {e}. Mastodon might be unavailable for operations.")
        elif check_creds: # File not found and checking creds
            logger.debug(f"Mastodon config file '{self.mastodon_config_file_path_str}' not found. Mastodon unavailable for credentialed check.")


        # HackerNews (no creds needed, always available conceptually)
        # This logic means it's always added if not check_creds, or always added if check_creds (as it has no creds to fail)
        if not check_creds or True: 
            available.append("hackernews")
        return sorted(list(set(available))) # Unique and sorted

    def run(self): # pragma: no cover
        """Runs the interactive mode of the analyzer."""
        self.console.print(Panel("[bold blue]SocialOSINTLM[/bold blue]\nCollects and analyzes user activity across multiple platforms using vision and LLMs.\nEnsure API keys and identifiers are set in your `.env` file or Mastodon JSON config.", title="Welcome", border_style="blue"))
        
        # Display offline mode status prominently
        if self.args.offline:
            self.console.print(Panel("[bold yellow]OFFLINE MODE ENABLED[/bold yellow]\nData will be sourced only from local cache. No new data will be fetched from platforms, and new media will not be downloaded or analyzed by vision models.", title_align="center", border_style="yellow"))

        # Initial check for LLM client initialization (critical for analysis)
        try:
            _ = self.llm_client # Accessing property triggers initialization
        except RuntimeError as core_err:
            self.console.print(f"[bold red]Critical LLM Configuration Error:[/bold red] {core_err}")
            self.console.print("Cannot proceed without LLM configuration (LLM_API_KEY, LLM_API_BASE_URL etc.).")
            return # Exit if LLM is not configured

        while True:
            self.console.print("\n[bold cyan]Select Platform(s) for Analysis:[/bold cyan]")
            current_available = self.get_available_platforms(check_creds=True)

            if not current_available:
                all_conceptual = self.get_available_platforms(check_creds=False)
                # If only HackerNews is conceptually available and no others are configured
                # Also check if Mastodon is not even conceptually available (e.g. config file doesn't exist at all)
                mastodon_conceptually_available = "mastodon" in all_conceptual
                is_only_hackernews = "hackernews" in all_conceptual and len(all_conceptual) == 1 \
                                   or ("hackernews" in all_conceptual and "mastodon" in all_conceptual and len(all_conceptual) == 2 and not mastodon_conceptually_available)


                if is_only_hackernews :
                    self.console.print("[yellow]Only HackerNews seems to be available (no other platform credentials or Mastodon config found).[/yellow]")
                    current_available = ["hackernews"] # Allow proceeding with just HN
                else: # Could be other conceptual platforms but none configured
                    self.console.print("[bold red]Error: No platforms seem to be configured correctly for use.[/bold red]")
                    self.console.print("Please set credentials in a `.env` file (for Twitter, Reddit, Bluesky) and/or ensure a valid Mastodon JSON configuration file is present and readable.")
                    self.console.print("Check `analyzer.log` for detailed errors during startup or platform availability checks.")
                    break # Exit main loop

            # Sort available platforms for consistent display (e.g., Twitter first)
            platform_priority = {"twitter": 1, "bluesky": 2, "mastodon": 3, "reddit": 4, "hackernews": 5}
            current_available.sort(key=lambda x: (platform_priority.get(x, 999), x))

            platform_options = {str(i + 1): p for i, p in enumerate(current_available)}
            num_platforms = len(current_available)
            next_key_val = num_platforms + 1 # Renamed next_key

            cross_platform_key = None
            if num_platforms > 1: # Only offer cross-platform if more than one is available
                cross_platform_key = str(next_key_val)
                platform_options[cross_platform_key] = "cross-platform"
                next_key_val += 1
            
            purge_key = str(next_key_val); platform_options[purge_key] = "purge data"; next_key_val +=1
            exit_key = str(next_key_val); platform_options[exit_key] = "exit"


            for key_opt, name_opt in platform_options.items(): # Renamed key, name
                display_name_opt = name_opt.replace("-", " ").capitalize() # Renamed display_name
                self.console.print(f" {key_opt}. {display_name_opt}")
            
            choice = Prompt.ask(f"Enter number(s) (e.g., 1 or 1,3 or {cross_platform_key if cross_platform_key else 'N/A'})", default=exit_key).strip().lower()

            if choice == exit_key or choice == "exit": break
            if choice == purge_key or choice == "purge data":
                self._handle_purge()
                continue


            selected_platform_keys = []
            selected_names = [] # For display
            special_keys_for_selection = {exit_key, purge_key} # Keys not for platform selection
            if cross_platform_key: special_keys_for_selection.add(cross_platform_key)


            if cross_platform_key and (choice == cross_platform_key or choice == "cross-platform"):
                selected_platform_keys = [k for k, v in platform_options.items() if k not in special_keys_for_selection]
                selected_names = [platform_options[k] for k in selected_platform_keys]
                self.console.print(f"Selected: Cross-Platform Analysis ({', '.join(name.capitalize() for name in selected_names)})")
            else:
                raw_keys = [k.strip() for k in choice.split(",")]
                valid_keys_found = []
                invalid_inputs = []
                for k_choice in raw_keys: # Renamed k to k_choice
                    if k_choice in platform_options and k_choice not in special_keys_for_selection:
                        valid_keys_found.append(k_choice)
                    else: invalid_inputs.append(k_choice)

                if not valid_keys_found:
                    if any(k_choice in special_keys_for_selection for k_choice in raw_keys): # User picked exit/purge etc. with comma
                         self.console.print(f"[yellow]Invalid selection for analysis: '{choice}'. Please use platform numbers.[/yellow]")
                    else: self.console.print(f"[yellow]Invalid selection: '{choice}'. Please enter numbers corresponding to the platform options.[/yellow]")
                    continue
                if invalid_inputs: self.console.print(f"[yellow]Ignoring invalid input(s): {', '.join(invalid_inputs)}[/yellow]")

                selected_platform_keys = sorted(list(set(valid_keys_found))) # Unique, sorted keys
                selected_names = [platform_options[k].capitalize() for k in selected_platform_keys]
                self.console.print(f"Selected: {', '.join(selected_names)}")


            platforms_to_query: Dict[str, List[str]] = {}
            abort_selection = False # Flag to break outer loop if needed
            try:
                for key_sel in selected_platform_keys: # Renamed key
                    if abort_selection: break
                    if key_sel not in platform_options: continue # Should not happen due to prior validation
                    platform_name_sel = platform_options[key_sel] # Renamed platform_name

                    # Build prompt message with platform-specific examples/guidance
                    prompt_message = f"Enter {platform_name_sel.capitalize()} username(s) (comma-separated"
                    if platform_name_sel == "twitter": prompt_message += ", no '@')"
                    elif platform_name_sel == "reddit": prompt_message += ", no 'u/')"
                    elif platform_name_sel == "bluesky": prompt_message += ", e.g., 'handle.bsky.social')"
                    elif platform_name_sel == "mastodon": prompt_message += ", format: 'user@instance.domain')"
                    # else: prompt_message += ")" # Covers hackernews and default - moved below
                    
                    # Add offline mode notice to prompt
                    if self.args.offline:
                        prompt_message += " - OFFLINE, cached data only)"
                    else:
                        prompt_message += ")" # Close parenthesis if not offline

                    user_input = Prompt.ask(prompt_message, default="").strip()
                    if not user_input:
                        self.console.print(f"[yellow]No usernames entered for {platform_name_sel.capitalize()}. Skipping.[/yellow]")
                        continue
                    
                    usernames_list_raw = [u.strip() for u in user_input.split(",") if u.strip()] # Split and strip # Renamed usernames
                    if not usernames_list_raw: # If all inputs were empty or just commas
                        self.console.print(f"[yellow]No valid usernames provided for {platform_name_sel.capitalize()} after stripping. Skipping.[/yellow]")
                        continue

                    # Platform-specific validation/normalization
                    validated_users = []
                    if platform_name_sel == "mastodon":
                        # Initialize Mastodon clients if not already, to get default instance info for prompt
                        if not self._mastodon_clients_initialized:
                            self._initialize_mastodon_clients()
                        
                        default_instance_domain_for_prompt = None
                        if self._default_mastodon_lookup_client and self._default_mastodon_lookup_client.api_base_url:
                            # Extract domain like 'mastodon.social' from 'https://mastodon.social'
                            default_instance_domain_for_prompt = urlparse(self._default_mastodon_lookup_client.api_base_url).netloc
                        
                        for u_val in usernames_list_raw: # Renamed u
                            if "@" in u_val and self._get_instance_domain_from_acct(u_val): # Basic check for user@instance.domain
                                validated_users.append(u_val)
                            else: # Username lacks instance
                                if default_instance_domain_for_prompt: # If we have a default from config
                                    assumed_user = f"{u_val}@{default_instance_domain_for_prompt}"
                                    confirm_text = f"[yellow]Username '{u_val}' for Mastodon lacks an instance. Assume '{assumed_user}' (derived from default instance in your config)? [/yellow]"
                                    if Confirm.ask(Text.from_markup(confirm_text), default=True):
                                        validated_users.append(assumed_user)
                                    else: self.console.print(f"[yellow]Skipping Mastodon username '{u_val}' due to missing instance and no confirmation.[/yellow]")
                                else: # No default instance, cannot assume
                                    self.console.print(f"[bold red]Invalid Mastodon username: '{u_val}'. Must be 'user@instance.domain'. No default instance configured to make an assumption. Skipping.[/bold red]")
                        # usernames_list_raw = validated_users # This should be assigned to validated_users and then used
                    elif platform_name_sel == "twitter": 
                        validated_users = [u.lstrip("@") for u in usernames_list_raw]
                    elif platform_name_sel == "reddit": 
                        validated_users = [u.replace("u/", "").replace("/u/", "") for u in usernames_list_raw]
                    else: # For Bluesky handle, HackerNews, etc.
                        validated_users = usernames_list_raw
                    
                    if validated_users:
                        if platform_name_sel not in platforms_to_query: platforms_to_query[platform_name_sel] = []
                        # Add only unique usernames per platform
                        current_users_on_platform = set(platforms_to_query[platform_name_sel])
                        added_this_round_count = 0
                        for user_to_add_val in validated_users: # Renamed user_val
                            if user_to_add_val not in current_users_on_platform:
                                platforms_to_query[platform_name_sel].append(user_to_add_val)
                                current_users_on_platform.add(user_to_add_val)
                                added_this_round_count +=1
                        if added_this_round_count < len(validated_users): 
                            logger.debug(f"Excluded {len(validated_users) - added_this_round_count} duplicate username(s) for {platform_name_sel}.")
                    else: 
                        logger.warning(f"No valid usernames remained for {platform_name_sel} after validation/confirmation.")


                if not platforms_to_query: # No valid users for any selected platform
                    self.console.print("[yellow]No valid usernames entered or confirmed for any selected platform. Returning to menu.[/yellow]")
                    continue

                # Initialize clients for selected platforms (catches config issues early)
                self.console.print(f"[cyan]Initializing API client systems{' (Offline mode checks)' if self.args.offline else ''}...")
                
                client_init_progress = Progress(
                    SpinnerColumn(),
                    "[progress.description]{task.description}",
                    transient=True,
                    console=self.console,
                    refresh_per_second=10,
                )
                with client_init_progress:
                    init_task_id = client_init_progress.add_task("Initializing...", total=len(platforms_to_query)) # Renamed init_task
                    for platform_name_iter_init in list(platforms_to_query.keys()): # Renamed platform_name, iterate over copy of keys
                        if init_task_id is not None and init_task_id in client_init_progress.task_ids:
                             client_init_progress.update(init_task_id, description=f"Initializing {platform_name_iter_init.capitalize()} system...")
                        try:
                            # Accessing the property will trigger initialization and validation
                            # HackerNews has no client property on `self` (uses httpx directly).
                            if platform_name_iter_init == "mastodon":
                                # Explicitly trigger Mastodon client initialization from config
                                # This ensures _mastodon_clients and _default_mastodon_lookup_client are populated.
                                if not self._mastodon_clients_initialized: # Check if already done
                                    self._initialize_mastodon_clients()
                                # After initialization, check if any clients were actually loaded for Mastodon
                                if not self._mastodon_clients:
                                    raise RuntimeError("Mastodon client initialization failed: No clients loaded from configuration. Check config file and logs.")
                                logger.info(f"Mastodon client system (clients: {len(self._mastodon_clients)}) initialized/checked successfully.")
                            elif platform_name_iter_init != "hackernews": 
                                _ = getattr(self, platform_name_iter_init) # e.g., self.twitter
                            # If no exception, log success for the platform system
                            logger.info(f"{platform_name_iter_init.capitalize()} client system initialized successfully.")
                        except (RuntimeError, ValueError, MastodonError, tweepy.errors.TweepyException, prawcore.exceptions.PrawcoreException, atproto_exceptions.AtProtocolError, OpenAIError) as client_err: # Added OpenAIError
                            self.console.print(f"[bold red]Error initializing {platform_name_iter_init.capitalize()} client system:[/bold red] {client_err}")
                            self.console.print(f"[yellow]Cannot analyze {platform_name_iter_init.capitalize()}. Check credentials/config and logs.[/yellow]")
                            del platforms_to_query[platform_name_iter_init] # Remove from this round
                        finally:
                            if init_task_id is not None and init_task_id in client_init_progress.task_ids:
                                client_init_progress.advance(init_task_id)
                
                if not platforms_to_query: # If all selected platforms failed client init
                    self.console.print("[bold red]No client systems could be initialized successfully for selected platforms. Returning to menu.[/bold red]")
                    continue

                # Proceed to analysis loop with successfully initialized platforms
                self._run_analysis_loop(platforms_to_query)

            except (KeyboardInterrupt, EOFError): # Ctrl+C or Ctrl+D during platform/user input
                self.console.print("\n[yellow]Operation cancelled.[/yellow]")
                if Confirm.ask("Exit program?", default=False): abort_selection = True; break
                else: continue # To platform selection
            except Exception as e: # Catch-all for unexpected issues in this input phase
                logger.error(f"Unexpected error in main interactive loop: {e}", exc_info=True)
                self.console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
                if Confirm.ask("Try again?", default=False): continue
                else: abort_selection = True; break
            if abort_selection: break # Break from main while True if exit confirmed

        self.console.print("\n[blue]Exiting SocialOSINTLM analyzer.[/blue]")

    def _run_analysis_loop(self, platforms: Dict[str, List[str]]): # pragma: no cover
        """Runs the analysis query loop after platforms and users are selected."""
        # Prepare display string for current targets
        platform_labels = []
        platform_names_list_for_save = sorted(platforms.keys()) # For saving output # Renamed platform_names_list
        for pf_display, users_list_display in platforms.items(): # Renamed users
            display_users_list_items = []
            user_prefix_display = ""
            if pf_display == "twitter": user_prefix_display = "@"
            elif pf_display == "reddit": user_prefix_display = "u/"
            # Mastodon/Bluesky/HN handles are usually complete
            for u_item_display in users_list_display: display_users_list_items.append(f"{user_prefix_display}{u_item_display}" if user_prefix_display else u_item_display) # Renamed u
            platform_labels.append(f"{pf_display.capitalize()}: {', '.join(display_users_list_items)}")
        platform_info_str = " | ".join(platform_labels) # Renamed platform_info
        current_targets_str_display = f"Targets: {platform_info_str}" # Renamed current_targets_str
        
        offline_msg_display = " (OFFLINE MODE - Cache Only)" if self.args.offline else "" # Renamed offline_msg
        self.console.print(Panel(f"{current_targets_str_display}{offline_msg_display}\nEnter your analysis query below.\nCommands: `exit` (return to menu), `refresh` (force full data fetch{'' if not self.args.offline else ', N/A in offline'}), `help`", title="🔎 Analysis Session", border_style="cyan", expand=False))
        last_query_val = "" # Store the last successful query # Renamed last_query

        while True:
            try:
                query_input = Prompt.ask("\n[bold green]Analysis Query>[/bold green]", default=last_query_val).strip() # Renamed query
                if not query_input: continue # User just pressed Enter without input
                last_query_val = query_input # Store for re-use
                cmd_input = query_input.lower() # Renamed cmd

                if cmd_input == "exit":
                    self.console.print("[yellow]Exiting analysis session, returning to platform selection.[/yellow]")
                    break
                if cmd_input == "help":
                    self.console.print(Panel(f"**Available Commands:**\n- `exit`: Return to the platform selection menu.\n- `refresh`: Force a full data fetch for all current targets, ignoring cache{' (N/A in offline mode)' if self.args.offline else ''}.\n- `help`: Show this help message.\n\n**To analyze:**\nSimply type your analysis question (e.g., 'What are the main topics discussed?', 'Identify potential location clues from images and text.')\nPressing Enter with no input repeats the last query.", title="Help", border_style="blue", expand=False))
                    continue
                if cmd_input == "refresh":
                    if self.args.offline:
                        self.console.print("[yellow]'refresh' command is not applicable in offline mode as no new data is fetched from platforms.[/yellow]")
                        continue

                    # Confirm before forcing refresh as it uses more API calls
                    if Confirm.ask("Force refresh data for all current targets? This ignores cache and uses more API calls.", default=False):
                        total_targets_to_refresh = sum(len(u_list_for_refresh) for u_list_for_refresh in platforms.values()) # Corrected to u_list # Renamed u_list # Renamed total_targets_refresh
                        failed_refreshes_list = [] # Renamed failed_refreshes
                        
                        # Create a Progress instance specifically for refresh
                        refresh_progress_bar = Progress( # Renamed refresh_progress
                            SpinnerColumn(),
                            "[progress.description]{task.description}",
                            transient=True,
                            console=self.console,
                            refresh_per_second=10,
                        )
                        with refresh_progress_bar:
                            refresh_task_id_val = refresh_progress_bar.add_task("[yellow]Refreshing data...", total=total_targets_to_refresh) # Renamed refresh_task # Renamed refresh_task_id
                            for platform_for_refresh, usernames_for_refresh in platforms.items(): # Renamed platform, usernames # Renamed platform_ref, usernames_ref
                                fetcher_for_refresh = getattr(self, f"fetch_{platform_for_refresh}", None) # Renamed fetcher # Renamed fetcher_ref
                                if not fetcher_for_refresh: continue # Should not happen if platform is in `platforms`
                                for username_item_refresh in usernames_for_refresh: # Renamed username # Renamed username_ref
                                    display_name_for_refresh = username_item_refresh # Simple display for progress # Renamed display_name # Renamed display_name_ref
                                    if platform_for_refresh == "twitter": display_name_for_refresh = f"@{username_item_refresh}"
                                    elif platform_for_refresh == "reddit": display_name_for_refresh = f"u/{username_item_refresh}"
                                    
                                    if refresh_task_id_val is not None and refresh_task_id_val in refresh_progress_bar.task_ids:
                                        refresh_progress_bar.update(refresh_task_id_val, description=f"[yellow]Refreshing {platform_for_refresh}/{display_name_for_refresh}...")
                                    try:
                                        result_from_fetch = fetcher_for_refresh(username=username_item_refresh, force_refresh=True) # Renamed result
                                        if result_from_fetch is None: # Fetcher indicated failure, already logged by fetcher
                                            # Add to local list if not already (e.g. from specific exceptions)
                                            if not any(f[0] == platform_for_refresh and f[1] == display_name_for_refresh for f in failed_refreshes_list):
                                                failed_refreshes_list.append((platform_for_refresh, display_name_for_refresh))
                                    except Exception as e_refresh_loop: # Catch-all for unexpected issues during forced refresh # Renamed e # Renamed e_ref
                                        logger.error(f"Unexpected error during refresh for {platform_for_refresh}/{display_name_for_refresh}: {e_refresh_loop}", exc_info=True)
                                        self.console.print(f"[red]Unexpected Refresh failed for {platform_for_refresh}/{display_name_for_refresh}: {e_refresh_loop}[/red]")
                                        if not any(f[0] == platform_for_refresh and f[1] == display_name_for_refresh for f in failed_refreshes_list):
                                             failed_refreshes_list.append((platform_for_refresh, display_name_for_refresh))
                                    finally:
                                        if refresh_task_id_val is not None and refresh_task_id_val in refresh_progress_bar.task_ids:
                                             refresh_progress_bar.advance(refresh_task_id_val)
                        if failed_refreshes_list: self.console.print(f"[yellow]Data refresh attempted, but issues encountered for {len(failed_refreshes_list)} target(s) (see logs/previous messages).[/yellow]")
                        else: self.console.print("[green]Data refresh attempt completed for all targets.[/green]")
                    continue # Back to query prompt

                # Actual analysis call
                self.console.print(f"\n[cyan]Starting analysis for query:[/cyan] '{query_input}'", highlight=False)
                analysis_result_text = self.analyze(platforms, query_input) # Renamed analysis_result

                if analysis_result_text:
                    # Check if the result is an error/warning message string or a report
                    result_lower_stripped_text = analysis_result_text.strip().lower() # Renamed result_lower_stripped
                    is_error_result = any(result_lower_stripped_text.startswith(prefix) for prefix in ["[red]", "error:", "analysis failed", "analysis aborted"]) # Renamed is_error
                    is_warning_result = result_lower_stripped_text.startswith("[yellow]") or result_lower_stripped_text.startswith("warning:") # Renamed is_warning
                    border_color_result = "red" if is_error_result else ("yellow" if is_warning_result else "green") # Renamed border_col

                    content_to_render_final: Union[Markdown, Text] # Renamed content_to_render
                    if is_error_result or is_warning_result:
                        # For error/warning messages that are already Rich-formatted strings
                        content_to_render_final = Text.from_markup(analysis_result_text)
                    else:
                        # For actual LLM-generated Markdown reports
                        content_to_render_final = Markdown(analysis_result_text)
                    
                    # Display the report in a panel
                    self.console.print(Panel(content_to_render_final, title="Analysis Report", border_style=border_color_result, expand=False, title_align="left"))

                    if not is_error_result: # Only offer to save non-error reports
                        should_save_report = False # Renamed save_report
                        output_save_format = "markdown" # Default # Renamed save_format
                        no_auto_save_arg = getattr(self.args, "no_auto_save", False) # Renamed no_auto_save_flag
                        specified_format_arg = getattr(self.args, "format", "markdown") # Renamed specified_format

                        if no_auto_save_arg: # If --no-auto-save is set, always prompt
                            if Confirm.ask("Save this analysis report?", default=True):
                                should_save_report = True
                                output_save_format = Prompt.ask("Save format?", choices=["markdown", "json"], default=specified_format_arg)
                        else: # Auto-save is enabled (default behavior)
                            output_save_format = specified_format_arg # Use format from args, or its default
                            self.console.print(f"[cyan]Auto-saving analysis report as {output_save_format}...[/cyan]")
                            should_save_report = True
                        
                        if should_save_report:
                            self._save_output(analysis_result_text, query_input, platform_names_list_for_save, output_save_format)
                else:
                    self.console.print("[red]Analysis returned no result (None). Check logs.[/red]")

            except (KeyboardInterrupt, EOFError): # Ctrl+C/D during query input
                self.console.print("\n[yellow]Analysis query cancelled.[/yellow]")
                if Confirm.ask("\nExit this analysis session (return to menu)?", default=False): break
                else: last_query_val = ""; continue # Clear last query and restart loop
            except RateLimitExceededError: # Should be caught by analyze, but defensive
                 self.console.print("[yellow]A rate limit was hit during analysis. Please wait before trying again.[/yellow]")
            except RuntimeError as e_runtime_loop: # From API call failure or other programmatic issues # Renamed e # Renamed e_run
                logger.error(f"Runtime error during analysis query processing: {e_runtime_loop}", exc_info=True)
                self.console.print(f"\n[bold red]An error occurred during analysis:[/bold red] {e_runtime_loop}")
                self.console.print("[yellow]Check logs for details. You can try again or exit.[/yellow]")
            except Exception as e_exception_loop: # Unexpected error during analysis loop # Renamed e # Renamed e_exc
                logger.error(f"Unexpected error during analysis loop: {e_exception_loop}", exc_info=True)
                self.console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {e_exception_loop}")
                if not Confirm.ask("An error occurred. Continue session?", default=True): break
    
    def process_stdin(self): # pragma: no cover
        """Processes a single analysis request received via stdin."""
        stderr_console = Console(stderr=True) # For messages not part of report
        
        # Display offline mode status if active for stdin mode
        if self.args.offline:
            stderr_console.print(Panel("[bold yellow]OFFLINE MODE ENABLED[/bold yellow]\nData will be sourced only from local cache. No new data will be fetched from platforms, and new media will not be downloaded or analyzed by vision models.", title_align="center", border_style="yellow"))
        else:
            stderr_console.print("[cyan]Processing analysis request from stdin...[/cyan]")


        try:
            try:
                input_data_json = json.load(sys.stdin) # Renamed input_data
            except json.JSONDecodeError as json_err_stdin: # Renamed json_err
                raise ValueError(f"Invalid JSON received on stdin: {json_err_stdin}")

            platforms_from_stdin = input_data_json.get("platforms") # Renamed platforms_in
            query_from_stdin = input_data_json.get("query") # Renamed query

            if not isinstance(platforms_from_stdin, dict) or not platforms_from_stdin:
                raise ValueError("Invalid or missing 'platforms' data in JSON input. Must be a non-empty dictionary.")
            if not isinstance(query_from_stdin, str) or not query_from_stdin.strip():
                raise ValueError("Invalid or missing 'query' in JSON input.")
            query_from_stdin = query_from_stdin.strip()

            valid_platforms_to_process: Dict[str, List[str]] = {} # Renamed valid_platforms_to_analyze
            available_platforms_configured = self.get_available_platforms(check_creds=True) # Platforms with creds # Renamed available_configured
            available_platforms_conceptual = self.get_available_platforms(check_creds=False) # All platforms tool knows # Renamed available_conceptual

            for platform_key_stdin, usernames_val_stdin in platforms_from_stdin.items(): # Renamed platform, usernames # Renamed platform_key, usernames_val
                platform_key_lower_stdin = platform_key_stdin.lower() # Normalize platform name # Renamed platform # Renamed platform_key_lower
                if platform_key_lower_stdin not in available_platforms_conceptual:
                    logger.warning(f"Platform '{platform_key_lower_stdin}' specified in stdin is not supported by this tool. Skipping.")
                    continue
                
                # Check if platform requires config and if it's configured
                # In offline mode, we still need the "concept" of the platform to exist, but don't need live creds to be working
                # as we only use cache. However, client instantiation might still fail if basic env vars are missing.
                platform_requires_config = platform_key_lower_stdin != "hackernews" # HN needs no special config # Renamed requires_config
                if platform_requires_config and platform_key_lower_stdin not in available_platforms_configured and not self.args.offline:
                    # Only strictly enforce configured platforms if *not* in offline mode
                    logger.warning(f"Platform '{platform_key_lower_stdin}' specified in stdin requires configuration, but credentials/setup seem missing or invalid. Skipping for online mode.")
                    continue

                # Process usernames (string or list)
                processed_usernames_stdin = [] # Renamed processed_users
                if isinstance(usernames_val_stdin, str):
                    if usernames_val_stdin.strip(): processed_usernames_stdin = [usernames_val_stdin.strip()]
                elif isinstance(usernames_val_stdin, list):
                    processed_usernames_stdin = [u.strip() for u in usernames_val_stdin if isinstance(u, str) and u.strip()]
                else:
                    logger.warning(f"Invalid username format for platform '{platform_key_lower_stdin}' in stdin. Expected string or list of strings. Skipping platform.")
                    continue
                
                if not processed_usernames_stdin:
                    logger.warning(f"No valid usernames provided for platform '{platform_key_lower_stdin}' in stdin. Skipping platform.")
                    continue

                # Validate/normalize usernames per platform
                validated_usernames_for_platform_stdin = [] # Renamed validated_users_for_platform
                if platform_key_lower_stdin == "mastodon":
                    for u_mastodon_stdin in processed_usernames_stdin: # Renamed u # Renamed u_mast
                        if "@" in u_mastodon_stdin and self._get_instance_domain_from_acct(u_mastodon_stdin): 
                            validated_usernames_for_platform_stdin.append(u_mastodon_stdin)
                        else: 
                            logger.warning(f"Invalid Mastodon username format in stdin for '{u_mastodon_stdin}'. Needs 'user@instance.domain'. Skipping user.")
                elif platform_key_lower_stdin == "twitter": 
                    validated_usernames_for_platform_stdin = [u.lstrip("@") for u in processed_usernames_stdin]
                elif platform_key_lower_stdin == "reddit": 
                    validated_usernames_for_platform_stdin = [u.replace("u/", "").replace("/u/", "") for u in processed_usernames_stdin]
                else: # Bluesky, HackerNews
                    validated_usernames_for_platform_stdin = processed_usernames_stdin

                if not validated_usernames_for_platform_stdin:
                    logger.warning(f"No valid usernames remained for platform '{platform_key_lower_stdin}' after validation. Skipping platform.")
                    continue

                if validated_usernames_for_platform_stdin:
                    valid_platforms_to_process[platform_key_lower_stdin] = sorted(list(set(validated_usernames_for_platform_stdin))) # Store unique, sorted

            if not valid_platforms_to_process:
                raise ValueError("No valid and configured platforms with valid usernames found in the processed input.")
            
            # Initialize clients for requested platforms
            logger.info(f"Initializing API client systems for stdin request{' (Offline mode checks)' if self.args.offline else ''}...")
            platforms_failed_init_list = [] # Renamed platforms_failed_init
            for platform_name_for_init_stdin in list(valid_platforms_to_process.keys()): # Renamed platform_name_iter # Renamed platform_name_init
                if platform_name_for_init_stdin == "hackernews": # No client to init for HN
                    logger.info("HackerNews requires no specific client system initialization.")
                    continue
                try:
                    if platform_name_for_init_stdin == "mastodon":
                        # Explicitly trigger Mastodon client initialization from config
                        if not self._mastodon_clients_initialized: # Check if already done
                            self._initialize_mastodon_clients()
                        # After initialization, check if any clients were actually loaded
                        if not self._mastodon_clients:
                            raise RuntimeError("Mastodon client system initialization failed: No clients loaded from configuration. Check config file and logs.")
                        logger.info(f"Mastodon client system (clients: {len(self._mastodon_clients)}) initialized/checked successfully for stdin.")
                    else:
                        _ = getattr(self, platform_name_for_init_stdin) # Trigger client init for other platforms
                    logger.info(f"{platform_name_for_init_stdin.capitalize()} client system initialized successfully for stdin.")
                except (RuntimeError, ValueError, MastodonError, tweepy.errors.TweepyException, prawcore.exceptions.PrawcoreException, atproto_exceptions.AtProtocolError, OpenAIError) as client_err_stdin: # Added OpenAIError # Renamed client_err
                    logger.error(f"Error initializing {platform_name_for_init_stdin.capitalize()} client system for stdin: {client_err_stdin}")
                    del valid_platforms_to_process[platform_name_for_init_stdin] # Remove if client fails
                    platforms_failed_init_list.append(platform_name_for_init_stdin)
            
            if not valid_platforms_to_process and not platforms_failed_init_list: # This means no platforms were requested that needed init (e.g., only HN)
                pass 
            elif not valid_platforms_to_process : # All platforms that needed init failed
                raise RuntimeError(f"No client systems could be initialized successfully for the requested platforms: {', '.join(platforms_failed_init_list)}.")
            if platforms_failed_init_list:
                logger.warning(f"Analysis via stdin will proceed without platforms that failed client system initialization: {', '.join(platforms_failed_init_list)}")


            # Perform analysis
            logger.info(f"Starting stdin analysis for query: '{query_from_stdin}' on platforms: {list(valid_platforms_to_process.keys())}")
            analysis_report_content = self.analyze(valid_platforms_to_process, query_from_stdin) # Renamed analysis_report

            if not analysis_report_content: # Should be caught by analyze(), but defensive
                raise RuntimeError("Analysis function returned no result (None). Check logs for errors during data collection or formatting.")
            
            # Check if the report itself is an error message from analyze()
            result_lower_stripped_stdin = analysis_report_content.strip().lower() # Renamed result_lower_stripped
            is_error_report_stdin = any(result_lower_stripped_stdin.startswith(prefix) for prefix in ["[red]", "error:", "analysis failed", "analysis aborted"]) # Renamed is_error_report

            if not is_error_report_stdin:
                platform_names_list_for_save_stdin = sorted(valid_platforms_to_process.keys()) # Renamed platform_names_list # Renamed platform_names_list_save
                no_auto_save_arg_stdin = getattr(self.args, "no_auto_save", False) # Renamed no_auto_save_flag
                output_format_arg_stdin = getattr(self.args, "format", "markdown") # Renamed output_format

                if no_auto_save_arg_stdin: # Print to stdout if no-auto-save
                    # The analysis_report_content might contain Rich markup.
                    # To print it as intended (e.g., Markdown rendered or Text with colors),
                    # we should ideally use the console or parse it.
                    # For simple stdout, printing raw string is okay, but might show markup.
                    # If it's a Markdown string from LLM, printing it raw is fine.
                    print(analysis_report_content) 
                    logger.info("Analysis complete. Report printed to stdout (--no-auto-save).")
                    sys.exit(0) # Success
                else: # Auto-save enabled
                    self._save_output(analysis_report_content, query_from_stdin, platform_names_list_for_save_stdin, output_format_arg_stdin)
                    stderr_console.print(f"[green]Analysis complete. Output auto-saved ({output_format_arg_stdin}).[/green]")
                    sys.exit(0) # Success
            else: # The report from analyze() was an error message
                sys.stderr.write("Analysis completed with errors:\n")
                # analysis_report_content here is likely a Rich-formatted string like "[red]Error...[/red]"
                # For stderr, we might want to strip Rich tags if not using Rich Console for stderr.
                # Or, if we are okay with Rich tags in stderr logs, just print.
                # For now, print as is.
                sys.stderr.write(analysis_report_content + "\n") 
                logger.error(f"Analysis via stdin completed but generated an error report. Query: '{query_from_stdin}'")
                sys.exit(2) # Indicate error completion

        except (json.JSONDecodeError, ValueError, RuntimeError) as e_proc_stdin: # Catch specific processing errors # Renamed e # Renamed e_proc
            logger.error(f"Error processing stdin request: {e_proc_stdin}", exc_info=False) # No full stack for these expected errors
            sys.stderr.write(f"Error: {e_proc_stdin}\n")
            sys.exit(1) # Indicate input/processing error
        except Exception as e_crit_stdin: # Unexpected critical errors during stdin processing # Renamed e # Renamed e_crit
            logger.critical(f"Unexpected critical error during stdin processing: {e_crit_stdin}", exc_info=True)
            sys.stderr.write(f"Critical Error: An unexpected error occurred - {e_crit_stdin}\n")
            sys.exit(1) # Indicate critical failure


if __name__ == "__main__": # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Social Media OSINT analyzer using LLMs. Fetches user data from various platforms, performs text and image analysis, and generates reports.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Environment Variables Required for LLM (using OpenAI-compatible API):
  LLM_API_KEY             : API key for your chosen LLM provider.
                            (e.g., OpenRouter key if using OpenRouter, your OpenAI key, 
                             or a specific key/placeholder like "NULL" for some self-hosted models).
  LLM_API_BASE_URL        : Base URL for the LLM API endpoint.
                            (e.g., https://openrouter.ai/api/v1 for OpenRouter,
                             https://api.openai.com/v1 for OpenAI,
                             or http://localhost:8000/v1 for a local OpenAI-compatible server).
  IMAGE_ANALYSIS_MODEL    : Vision model name recognized by your LLM provider/endpoint.
                            (e.g., openai/gpt-4o, google/gemini-pro-vision for OpenRouter, 
                             gpt-4-vision-preview for OpenAI, or model ID for self-hosted).
  ANALYSIS_MODEL          : Text model name recognized by your LLM provider/endpoint.
                            (e.g., mistralai/mixtral-8x7b-instruct for OpenRouter, 
                             gpt-4-turbo for OpenAI, or model ID for self-hosted).

Optional for OpenRouter (if LLM_API_BASE_URL points to OpenRouter):
  OPENROUTER_REFERER      : Your site URL or app name for OpenRouter's `HTTP-Referer` header. (Default: http://localhost:3000)
                            (Example: https://my-app.com)
  OPENROUTER_X_TITLE      : Your project name for OpenRouter's `X-Title` header. (Default: SocialOSINTLM)
                            (Example: My OSINT Project)

Platform Credentials (at least one set required, or use HackerNews / Mastodon config):
  TWITTER_BEARER_TOKEN    : Twitter API v2 Bearer Token.
  REDDIT_CLIENT_ID        : Reddit App Client ID.
  REDDIT_CLIENT_SECRET    : Reddit App Client Secret.
  REDDIT_USER_AGENT       : Reddit App User Agent string.
  BLUESKY_IDENTIFIER      : Bluesky handle or DID.
  BLUESKY_APP_SECRET      : Bluesky App Password.
  
Mastodon Configuration (replaces old MASTODON_* vars):
  MASTODON_CONFIG_FILE    : Path to a JSON file for Mastodon instance configurations.
                            (Default: "mastodon_instances.json" in the script's working directory or data/ folder)
                            Example mastodon_instances.json content:
                            [
                              {
                                "name": "Main Instance", 
                                "api_base_url": "https://mastodon.social",
                                "access_token": "YOUR_TOKEN_FOR_MASTODON_SOCIAL",
                                "is_default_lookup_instance": true 
                              },
                              {
                                "name": "Tech Hub",
                                "api_base_url": "https://fosstodon.org",
                                "access_token": "YOUR_TOKEN_FOR_FOSSTODON_ORG"
                              }
                            ]

Place these in a `.env` file in the same directory or set them in your environment.
The Mastodon JSON file should be placed according to the MASTODON_CONFIG_FILE path.
""",
    )
    parser.add_argument("--stdin", action="store_true", help="Read analysis request from stdin as JSON.\nExpected JSON format example:\n{\n  \"platforms\": {\n    \"twitter\": [\"user1\", \"user2\"],\n    \"reddit\": [\"user3\"],\n    \"hackernews\": [\"user4\"],\n    \"bluesky\": [\"handle1.bsky.social\"],\n    \"mastodon\": [\"user@instance.social\", \"another@other.server\"]\n  },\n  \"query\": \"Analyze communication style and main topics.\"\n}")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown", help="Output format for saving analysis reports (default: markdown).\n- markdown: Saves as a .md file with YAML frontmatter.\n- json: Saves as a .json file containing metadata and the markdown report.")
    parser.add_argument("--no-auto-save", action="store_true", help="Disable automatic saving of reports.\n- Interactive mode: Prompt user before saving.\n- Stdin mode: Print the report directly to stdout instead of saving.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="WARNING", help="Set the logging level (default: WARNING).")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode. Uses only cached data, no new API calls to social platforms or for new media downloads/vision model analysis.") # Added offline argument

    args = parser.parse_args()

    # Configure logging based on command-line argument
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.getLogger().setLevel(log_level_numeric) # Set root logger level
    logger.setLevel(log_level_numeric) # Set specific logger level for this module
    # Ensure handlers also respect this level
    for handler in logging.getLogger().handlers: handler.setLevel(log_level_numeric)
    logger.info(f"Logging level set to {args.log_level}")
    if args.offline:
        logger.info("Running in OFFLINE mode. API calls to social platforms will be skipped; using cache only.")


    analyzer_instance = None
    try:
        analyzer_instance = SocialOSINTLM(args=args)
        if args.stdin:
            analyzer_instance.process_stdin()
        else:
            analyzer_instance.run()
    except RuntimeError as e_main_run: # Catch critical init errors from SocialOSINTLM constructor # Renamed e
        logging.getLogger("SocialOSINTLM").critical(f"Initialization or Configuration failed: {e_main_run}", exc_info=False) # No stack for this known type
        # Use a simple console for this critical error as Rich might not be fully set up or part of issue
        error_console = Console(stderr=True, style="bold red")
        error_console.print(f"\nCRITICAL ERROR: {e_main_run}")
        error_console.print("Ensure necessary API keys (LLM_API_KEY, LLM_API_BASE_URL) and platform credentials/URLs are correctly set in .env or environment, or Mastodon JSON config.")
        error_console.print("Check analyzer.log for more details.")
        sys.exit(1)
    except Exception as e_main_crit: # Catch any other unexpected critical errors during startup # Renamed e
        logging.getLogger("SocialOSINTLM").critical(f"An unexpected critical error occurred: {e_main_crit}", exc_info=True)
        error_console = Console(stderr=True, style="bold red")
        error_console.print(f"\nUNEXPECTED CRITICAL ERROR: {e_main_crit}")
        error_console.print("Check analyzer.log for the full traceback.")
        sys.exit(1)
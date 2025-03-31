import os
import sys
import json
import hashlib
import logging
import argparse
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
from functools import lru_cache
import httpx
import tweepy
import praw
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
    handlers=[logging.FileHandler('analyser.log'), logging.StreamHandler()]
)
logger = logging.getLogger('SocialOSINTLM')

class RateLimitExceededError(Exception):
    pass

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class SocialOSINTLM:
    def __init__(self):
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
        self._verify_env_vars()
    
    def _verify_env_vars(self):
        required = ['OPENROUTER_API_KEY', 'IMAGE_ANALYSIS_MODEL']
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
            
    def _setup_directories(self):
        for dir_name in ['cache', 'media', 'outputs']:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    @property
    def bluesky(self):
        if not hasattr(self, '_bluesky_client'):
            try:
                client = Client()
                client.login(
                    os.environ['BLUESKY_IDENTIFIER'],
                    os.environ['BLUESKY_APP_SECRET']
                )
                self._bluesky_client = client
                logger.debug("Bluesky login successful")
                
            except (KeyError, exceptions.AtProtocolError) as e:
                raise RuntimeError(f"Bluesky setup failed: {e}")
        return self._bluesky_client

    @property
    def openrouter(self):
        if not hasattr(self, '_openrouter'):
            try:
                self._openrouter = httpx.Client(
                    base_url="https://openrouter.ai/api/v1",
                    headers={
                        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Social Media Analyser",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
            except KeyError as e:
                raise RuntimeError(f"Missing OpenRouter API key: {e}")
        return self._openrouter

    @property
    def reddit(self):
        if not hasattr(self, '_reddit'):
            try:
                self._reddit = praw.Reddit(
                    client_id=os.environ['REDDIT_CLIENT_ID'],
                    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
                    user_agent=os.environ['REDDIT_USER_AGENT']
                )
            except KeyError as e:
                raise RuntimeError(f"Missing Reddit credentials: {e}")
        return self._reddit

    @property
    def twitter(self):
        if not hasattr(self, '_twitter'):
            try:
                self._twitter = tweepy.Client(bearer_token=os.environ['TWITTER_BEARER_TOKEN'])
            except KeyError as e:
                raise RuntimeError(f"Missing Twitter credentials: {e}")
        return self._twitter

    def _handle_rate_limit(self, platform: str, exception=None):
        error_message = f"{platform} API rate limit exceeded"
        
        # Try to get reset time information if we have a Tweepy exception
        reset_info = ""
        if platform == 'Twitter' and isinstance(exception, tweepy.TooManyRequests):
            try:
                # Extract reset timestamp from the Tweepy exception
                reset_timestamp = exception.response.headers.get('x-rate-limit-reset')
                if reset_timestamp:
                    reset_time = datetime.fromtimestamp(int(reset_timestamp), tz=timezone.utc)
                    current_time = datetime.now(timezone.utc)
                    wait_seconds = max(int((reset_time - current_time).total_seconds()), 1)
                    
                    # Format wait time in a user-friendly way
                    if wait_seconds > 3600:
                        wait_str = f"{wait_seconds // 3600}h {(wait_seconds % 3600) // 60}m"
                    elif wait_seconds > 60:
                        wait_str = f"{wait_seconds // 60}m {wait_seconds % 60}s"
                    else:
                        wait_str = f"{wait_seconds}s"
                    
                    reset_info = f"Reset at: {reset_time.strftime('%H:%M:%S UTC')}\nWait time: {wait_str}"
            except (ValueError, AttributeError, TypeError) as e:
                logger.debug(f"Could not extract reset time: {e}")
        
        self.console.print(Panel(
            f"[bold red]Rate Limit Blocked: {platform}[/bold red]\n"
            f"API requests are currently unavailable.{f'\n\n{reset_info}' if reset_info else ''}",
            title="🚫 Rate Limit",
            border_style="red"
        ))
        raise RateLimitExceededError(error_message)

    def _get_media_path(self, url: str, platform: str, username: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.base_dir / 'media' / f"{platform}_{username}_{url_hash}.jpg"

    def _download_media(self, url: str, platform: str, username: str, headers: Optional[dict] = None) -> Optional[Path]:
        """Downloads media from URL with platform-specific handling."""
        media_path = self._get_media_path(url, platform, username)
        
        if media_path.exists():
            return media_path

        valid_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
        
        try:
            if platform == 'twitter':
                headers = {'Authorization': f'Bearer {os.environ["TWITTER_BEARER_TOKEN"]}'}
            elif platform == 'bluesky' and 'cdn.bsky.app' in url:
                url = url.replace('http://', 'https://')
                if '@jpeg' in url:
                    url = url.split('@jpeg')[0] + '@jpeg'

            resp = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
            resp.raise_for_status()

            if not any(vtype in resp.headers.get('content-type', '').lower() for vtype in valid_types):
                return None

            media_path.write_bytes(resp.content)
            return media_path

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limit
                self._handle_rate_limit(platform, e)
            logger.error(f"HTTP {e.response.status_code} for {url}")
            return None
        except Exception as e:
            logger.error(f"Download failed for {url}: {str(e)}")
            return None

    def fetch_hackernews(self, username: str, force=False) -> Optional[dict]:
        """Fetches user submissions from Hacker News via Algolia API"""
        if not force and (cached := self._load_cache('hackernews', username)):
            return cached

        try:
            url = f"https://hn.algolia.com/api/v1/search?tags=author_{quote_plus(username)}&hitsPerPage=50"
            
            with httpx.Client() as client:
                response = client.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()

            submissions = []
            for hit in data.get('hits', []):
                submissions.append({
                    'title': hit.get('title', '[No Title]'),
                    'url': hit.get('url', f"https://news.ycombinator.com/item?id={hit['objectID']}"),
                    'points': hit.get('points', 0),
                    'num_comments': hit.get('num_comments', 0),
                    'created_at': datetime.fromtimestamp(hit['created_at_i'], tz=timezone.utc),
                    'text': (hit.get('story_text') or hit.get('comment_text', ''))[:1000]
                })

            result = {
                'submissions': submissions,
                'stats': {
                    'total_submissions': len(submissions),
                    'average_points': sum(s['points'] for s in submissions)/len(submissions) if submissions else 0
                }
            }
            
            self._save_cache('hackernews', username, result)
            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._handle_rate_limit('HackerNews')
            logger.error(f"Hacker News fetch failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Hacker News fetch failed: {str(e)}")
            return None

    def _analyse_image(self, file_path: Path, context: str = "") -> Optional[str]:
        try:
            # Validate image before processing
            with Image.open(file_path) as img:
                if img.format.lower() not in ['jpeg', 'png', 'webp']:
                    logger.warning(f"Unsupported image type: {img.format}")
                    return None
                
                # Resize large images to prevent API limits
                max_size = (1024, 1024)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    temp_path = file_path.with_suffix('.thumb.jpg')
                    img.save(temp_path, 'JPEG')
                    file_path = temp_path

            # Prepare image data
            base64_image = base64.b64encode(file_path.read_bytes()).decode('utf-8')
            
            # Clean up temporary file if it exists
            if file_path.name.endswith('.thumb.jpg'):
                file_path.unlink()

            prompt_text = (
                f"Analyze this image from {context} for OSINT and user profiling insights. "
                "Focus *objectively* on identifying and describing key elements relevant to understanding the user, their environment, activities, or potential context. Specifically describe:\n"
                "- **Setting/Environment:** Indoor/outdoor? Urban/rural? Apparent room type? Any potential clues about the general environment (e.g., weather, architecture style)?\n"
                "- **Key Objects/Items:** Noticeable objects like tools, books (titles if legible), technology, unique items, branded products (mention brand if clear logo visible).\n"
                "- **People (if present):** Describe general appearance (clothing style, estimated age group), activity, number of people. *Do not guess identities.* Focus on observable details.\n"
                "- **Text/Symbols:** Any clearly legible text (signs, labels, writing on clothing), logos, or distinct symbols.\n"
                "- **Activity/Event:** What action or event seems to be taking place?\n"
                "- **Overall Impression:** What is the general theme or mood conveyed (e.g., professional, casual, personal, technical, artistic, political)?\n\n"
                "Provide a concise, descriptive summary or bulleted list of these observations. Avoid subjective interpretations or assumptions not directly supported by visual evidence."
            )

            response = self.openrouter.post(
                "/chat/completions",
                json={
                    "model": os.getenv('IMAGE_ANALYSIS_MODEL'),
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {
                                # Consider "detail": "high" for potentially richer analysis,
                                # but be aware of increased token cost and latency.
                                # "low" is faster/cheaper but might miss fine details.
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high" # Or "low"
                            }}
                        ]
                    }],
                    "max_tokens": 1000 # Adjust if needed based on typical response length
                }
            )
            response.raise_for_status()
            result = response.json()

            if 'choices' not in result or not result['choices']:
                logger.error(f"Invalid API response: {result.get('error', {}).get('message', 'No choices in response')}")
                return None
            return result['choices'][0]['message']['content']
                
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return None

    @lru_cache(maxsize=100)
    def _get_cache_path(self, platform: str, username: str) -> Path:
        return self.base_dir / 'cache' / f"{platform}_{username}.json"

    def _load_cache(self, platform: str, username: str) -> Optional[dict]:
        cache_path = self._get_cache_path(platform, username)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            # Convert ISO strings to datetime objects
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            if platform == 'bluesky':
                for post in data.get('posts', []):
                    post['created_at'] = datetime.fromisoformat(post['created_at'])
            if datetime.now(timezone.utc) - data['timestamp'] < timedelta(hours=24):
                return data
        except Exception:
            cache_path.unlink(missing_ok=True)
        return None


    def _save_cache(self, platform: str, username: str, data: dict):
        data['timestamp'] = datetime.now(timezone.utc).isoformat()
        self._get_cache_path(platform, username).write_text(
            json.dumps(data, indent=2, cls=DateTimeEncoder)
        )

    def fetch_twitter(self, username: str, force=False) -> Optional[dict]:
        
        try:
            if not force and (cached := self._load_cache('twitter', username)):
                return cached

            try:
                user_response = self.twitter.get_user(username=username)
                if not user_response or not user_response.data:
                    logger.error(f"User @{username} not found")
                    return None
                
                user = user_response.data
                tweets = self.twitter.get_users_tweets(
                    id=user['id'],  
                    max_results=30,
                    tweet_fields=['created_at', 'public_metrics', 'attachments'],
                    expansions=['attachments.media_keys'],
                    media_fields=['url', 'preview_image_url', 'type']
                )
            except tweepy.TooManyRequests as e:
                self._handle_rate_limit('Twitter', exception=e)
                return None
            except tweepy.NotFound:
                logger.error(f"User @{username} not found")
                return None
            except tweepy.Forbidden:
                logger.error(f"Access to @{username} is forbidden")
                return None

            processed = []
            media_analysis = []
            media_paths = []
            
            for tweet in tweets.data or []:
                tweet_data = {
                    'id': tweet['id'], 
                    'text': tweet['text'], 
                    'created_at': tweet['created_at'], 
                    'metrics': tweet['public_metrics'], 
                    'media': []
                }

                if tweet.get('attachments') and tweets.includes:  
                    media_includes = tweets.includes.get('media', [])
                    for media_key in tweet['attachments'].get('media_keys', []):  
                        media = next((m for m in media_includes if m['media_key'] == media_key), None)  
                        if media:
                            url = media.get('url') if media['type'] == 'photo' else media.get('preview_image_url')  
                            if url:
                                media_path = self._download_media(url=url, platform='twitter', username=username)
                                if media_path:
                                    media_paths.append(str(media_path))
                                    if analysis := self._analyse_image(media_path, f"Twitter user @{username}"):
                                        tweet_data['media'].append({
                                            'type': media['type'],  
                                            'analysis': analysis,
                                            'url': url,
                                            'local_path': str(media_path)
                                        })
                                        media_analysis.append(analysis)

                processed.append(tweet_data)

            data = {
                'tweets': processed,
                'media_analysis': media_analysis,
                'media_paths': media_paths,
                'user_info': {
                    'id': user['id'], 
                    'name': user['name'],  
                    'username': user['username'], 
                    'created_at': user.get('created_at') 
                }
            }
            self._save_cache('twitter', username, data)
            return data

        except Exception as e:
            logger.error(f"Twitter fetch failed: {str(e)}")
            return None

    def fetch_bluesky(self, username: str, force=False) -> Optional[dict]:
        """
        Updated fetch_bluesky method with proper image handling
        """
        try:
            if not force and (cached := self._load_cache('bluesky', username)):
                return cached

            cursor = None
            all_posts = []
            media_analysis = []
            media_paths = []

            for _ in range(3):  # Max 3 pages
                try:
                    response = self.bluesky.get_author_feed(
                        actor=username,
                        cursor=cursor,
                        limit=100
                    )
                except exceptions.AtProtocolError as e:
                    if 'rate limit' in str(e).lower():
                        self._handle_rate_limit('Bluesky', exception=e)
                    raise

                if not response.feed:
                    break

                for feed_item in response.feed:
                    post = feed_item.post
                    record = getattr(post, 'record', {})

                    raw_date = getattr(record, 'created_at', None)
                    if isinstance(raw_date, str):
                        created_at = datetime.fromisoformat(raw_date)
                    else:
                        created_at = raw_date or datetime.now(timezone.utc)

                    post_data = {
                        'uri': post.uri,
                        'text': getattr(record, 'text', ''),
                        'created_at': created_at.isoformat(),  # Now safe
                        'likes': getattr(post.viewer, 'like', 0) or 0,
                        'reposts': getattr(post.viewer, 'repost', 0) or 0,
                        'media': []
                    }

                    # Process images from embed
                    embed = getattr(record, 'embed', None)
                    if embed:
                        # Handle direct images
                        if hasattr(embed, 'images'):
                            for image in embed.images:
                                if analysis := self._process_bluesky_image(image, post, username, media_paths, post_data):
                                    media_analysis.append(analysis)

                        # Handle embedded record images
                        if hasattr(embed, 'record'):
                            embed_record = getattr(embed.record, 'value', {})
                            if hasattr(embed_record, 'embed'):
                                record_embed = embed_record.embed
                                if hasattr(record_embed, 'images'):
                                    for image in record_embed.images:
                                        if analysis := self._process_bluesky_image(image, post, username, media_paths, post_data):
                                            media_analysis.append(analysis)

                    all_posts.append(post_data)

                cursor = response.cursor
                if not cursor:
                    break

            # Calculate statistics
            data = {
                'posts': all_posts,
                'media_analysis': media_analysis,
                'media_paths': media_paths,
                'stats': {
                    'total_posts': len(all_posts),
                    'posts_with_media': len([p for p in all_posts if p['media']]),
                    'total_media': len(media_paths),
                    'avg_likes': sum(p['likes'] for p in all_posts)/len(all_posts) if all_posts else 0,
                    'avg_reposts': sum(p['reposts'] for p in all_posts)/len(all_posts) if all_posts else 0
                }
            }

            self._save_cache('bluesky', username, data)
            return data

        except Exception as e:
            logger.error(f"Bluesky fetch failed: {str(e)}")
            return None
       
    def _process_bluesky_image(self, image, post, username, media_paths, post_data):
        """Process Bluesky images with proper authentication and CDN URL handling"""
        try:
            # Get image ref - handle both direct and nested structures
            img_ref = getattr(image, 'image', None)
            if not img_ref:
                logger.warning("No image reference found")
                return None

            cid = getattr(img_ref.ref, 'link', None)
            if not cid:
                logger.warning(f"Missing CID in image ref")
                return None

            did = getattr(post.author, 'did', None)
            if not did:
                logger.warning("Missing author DID")
                return None

            # Get the access token from the client's protected session attribute
            access_token = self.bluesky._session.access_jwt
            if not access_token:
                logger.error("No access token available")
                return None

            cdn_url = f"https://cdn.bsky.app/img/feed_fullsize/plain/{quote_plus(did)}/{cid}@jpeg"

            media_path = self._download_media(
                url=cdn_url,
                platform='bluesky',
                username=username,
                headers={'Authorization': f"Bearer {access_token}"}
            )

            if not media_path:
                return None

            media_paths.append(str(media_path))
            
            analysis = self._analyse_image(
                media_path,
                f"Bluesky user {username}'s post"
            )

            if analysis:
                post_data['media'].append({
                    'type': 'image',
                    'analysis': analysis,
                    'url': cdn_url,
                    'local_path': str(media_path)
                })
                return analysis

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}", exc_info=True)
        
        return None

    def analyse(self, platforms: Dict[str, Union[str, List[str]]], query: str) -> str:
        collected_data = []
        media_analysis = []

        try:
            collect_task = self.progress.add_task(
                "[cyan]Collecting data...", 
                total=sum(len(v) if isinstance(v, list) else 1 for v in platforms.values())
            )
            
            for platform, usernames in platforms.items():
                if isinstance(usernames, str):
                    usernames = [usernames]
                
                for username in usernames:
                    if fetcher := getattr(self, f'fetch_{platform}', None):
                        try:
                            data = fetcher(username)
                        except RateLimitExceededError as e:
                            self.console.print(f"[red]{e}[/red]")
                            return f"Analysis aborted: {str(e)}"
                        
                        if data:
                            collected_data.append(self._format_text_data(platform, username, data))
                            
                            if platform in ['twitter', 'bluesky', 'reddit']:
                                media_analysis.extend(data.get('media_analysis', []))
                            
                            self.progress.advance(collect_task)
            
            self.progress.remove_task(collect_task)

            if not collected_data:
                return "No data available for analysis"

            analysis_components = []
            
            if media_analysis:
                analysis_components.append("## Media Analysis using {image_model}\n" + "\n".join(f"- {m}" for m in media_analysis))
            
            if collected_data:
                analysis_components.append("## Text Analysis using {model}\n" + "\n".join(collected_data))

            prompt = f"Analysis request: {query}\n\n" + "\n\n".join(analysis_components)
            model = os.getenv('ANALYSIS_MODEL')
            
            analysis_task = self.progress.add_task(f"[magenta]Final analysis...", total=None)
            try:
                api_thread = threading.Thread(
                    target=self._call_openrouter,
                    kwargs={
                        "json_data": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": """**Objective:** Generate a comprehensive behavioral and linguistic profile based on the provided social media data, employing structured analytic techniques focused on objectivity, evidence-based reasoning, and clear articulation.

                                    **Input:** You will receive summaries of user activity (text posts, engagement metrics, descriptive analyses of images shared) from platforms like Twitter, Reddit, Bluesky, and Hacker News for one or more specified users. The user will also provide a specific analysis query.

                                    **Primary Task:** Address the user's specific analysis query using the data provided and the analytical framework below.

                                    **Analysis Domains (Use these to structure your thinking):**
                                    1.  **Behavioral Patterns:** Analyze interaction frequency, contextual patterns (e.g., replies vs. original posts), potential engagement triggers, and temporal communication rhythms apparent in the data.
                                    2.  **Semantic Content & Themes:** Decode linguistic indicators such as emotional tone, potential ideological positioning, cognitive framing (how topics are discussed), and recurring themes or topics. Assess information source credibility *only if* the user shares external links/content within the provided data.
                                    3.  **Interests & Network Context:** Identify primary topic clusters and apparent domain interests suggested by the content. Note any interaction network dynamics visible *within the provided posts* (e.g., frequent retweets of specific accounts, if data shows this). Avoid inferring broad influence without strong evidence in the data.
                                    4.  **Communication Style:** Assess communication attributes like linguistic complexity (simple/complex language), use of rhetorical strategies, markers of emotional expression, and narrative consistency or patterns.

                                    **Analytical Constraints & Guidelines:**
                                    *   **Evidence-Based:** Base all conclusions *strictly and exclusively* on the provided source materials (text summaries and image analyses). Clearly state what data supports your points.
                                    *   **Objectivity:** Maintain strict analytical neutrality. Avoid personal bias, moral judgments, or speculative interpretations beyond what the data directly supports.
                                    *   **Integrate All Data:** Synthesize insights coherently from *both* textual content and the provided analyses of visual media (images). Note how they complement or contradict each other, if applicable.
                                    *   **Acknowledge Gaps & Limitations:** If data for a specific platform, time period, or aspect of the query is insufficient or missing, explicitly state this limitation. Do not speculate or fill in gaps without evidence.
                                    *   **Focus on Query:** Directly address the user's specific query. Use the analysis domains as tools to formulate your answer, not as a mandatory checklist to report on.
                                    *   **Clarity & Conciseness:** Provide insights in a structured, clear, and concise manner. Use formatting (like bullet points or distinct sections related to the query) to enhance readability.

                                    **Output:** A structured analytical response addressing the user's query, rigorously grounded in the provided data and adhering to all specified constraints."""},
                                {"role": "user", "content": prompt}
                            ]
                        }
                    }
                )
                api_thread.start()
                
                while api_thread.is_alive():
                    api_thread.join(0.1)
                    self.progress.refresh()
                
                if self._analysis_exception:
                    if isinstance(self._analysis_exception, httpx.HTTPStatusError):
                        err_details = f"HTTP {self._analysis_exception.response.status_code}"
                        err_details += f"\nResponse: {self._analysis_exception.response.text}"
                    else:
                        err_details = str(self._analysis_exception)
                    raise RuntimeError(f"API request failed: {err_details}")
                    
                response = self._analysis_response
                response.raise_for_status()
            finally:
                self.progress.remove_task(analysis_task)
                self._analysis_response = None
                self._analysis_exception = None

            response_data = response.json()
            if 'choices' not in response_data or not response_data['choices']:
                return f"Analysis failed: Invalid API response format - {response_data}"
            analysis = response_data['choices'][0]['message']['content']
            return f"## Comprehensive Analysis Report\n\n{analysis}"

        except Exception as e:
            return f"Analysis failed: {str(e)}"

    def fetch_reddit(self, username: str, force=False) -> Optional[dict]:
        """Fetches user submissions and comments from Reddit with image handling"""
        if not force and (cached := self._load_cache('reddit', username)):
            return cached

        try:
            user = self.reddit.redditor(username)
            submissions = []
            comments = []
            media_analysis = []
            media_paths = []

            # Fetch submissions
            for submission in user.submissions.new(limit=20):
                submission_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext[:500] if hasattr(submission, 'selftext') else '',
                    'score': submission.score,
                    'subreddit': submission.subreddit.display_name,
                    'permalink': submission.permalink,
                    'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                    'media': []
                }
                
                # Check for images in submission
                if hasattr(submission, 'url') and submission.url:
                    url = submission.url.lower()
                    # Check if URL is a direct image
                    if any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                        media_path = self._download_media(
                            url=submission.url, 
                            platform='reddit', 
                            username=username
                        )
                        if media_path:
                            media_paths.append(str(media_path))
                            if analysis := self._analyse_image(
                                media_path, 
                                f"Reddit user u/{username}'s post in r/{submission.subreddit.display_name}"
                            ):
                                submission_data['media'].append({
                                    'type': 'image',
                                    'analysis': analysis,
                                    'url': submission.url,
                                    'local_path': str(media_path)
                                })
                                media_analysis.append(analysis)
                    # Handle Reddit gallery posts
                    elif hasattr(submission, 'is_gallery') and submission.is_gallery:
                        if hasattr(submission, 'media_metadata'):
                            for image_id, image_data in submission.media_metadata.items():
                                if image_data['e'] == 'Image':  # e = 'Image' means it's an image
                                    if 's' in image_data and 'u' in image_data['s']:  # s = source, u = url
                                        image_url = image_data['s']['u']
                                        media_path = self._download_media(
                                            url=image_url, 
                                            platform='reddit', 
                                            username=username
                                        )
                                        if media_path:
                                            media_paths.append(str(media_path))
                                            if analysis := self._analyse_image(
                                                media_path, 
                                                f"Reddit user u/{username}'s gallery post in r/{submission.subreddit.display_name}"
                                            ):
                                                submission_data['media'].append({
                                                    'type': 'gallery_image',
                                                    'analysis': analysis,
                                                    'url': image_url,
                                                    'local_path': str(media_path)
                                                })
                                                media_analysis.append(analysis)
                
                submissions.append(submission_data)
            
            # Fetch comments
            for comment in user.comments.new(limit=30):
                comments.append({
                    'id': comment.id,
                    'text': comment.body[:500],
                    'score': comment.score,
                    'subreddit': comment.subreddit.display_name,
                    'permalink': comment.permalink,
                    'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
                })
            
            # Compile the data
            data = {
                'submissions': submissions,
                'comments': comments,
                'media_analysis': media_analysis,
                'media_paths': media_paths,
                'stats': {
                    'total_submissions': len(submissions),
                    'total_comments': len(comments),
                    'submissions_with_media': len([s for s in submissions if s['media']]),
                    'total_media': len(media_paths),
                    'avg_submission_score': sum(s['score'] for s in submissions)/max(len(submissions), 1),
                    'avg_comment_score': sum(c['score'] for c in comments)/max(len(comments), 1)
                }
            }
            
            self._save_cache('reddit', username, data)
            return data
            
        except prawcore.exceptions.RequestException as e:
            if '429' in str(e):  # Rate limit error
                self._handle_rate_limit('Reddit', exception=e)
            logger.error(f"Reddit fetch failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Reddit fetch failed: {str(e)}")
            return None

    def _format_text_data(self, platform: str, username: str, data: dict) -> str:
        """Formats fetched data into a more detailed text block for the analysis LLM."""
        # Define how many items to include per platform (adjust as needed for token limits)
        MAX_ITEMS = 15
        TEXT_SNIPPET_LENGTH = 500 # Max characters for long text fields

        output_lines = []

        if platform == 'twitter':
            output_lines.append(f"### Twitter Data Summary for @{username}")
            user_info = data.get('user_info', {})
            if user_info:
                output_lines.append(f"- User: {user_info.get('name')} (@{user_info.get('username')}), ID: {user_info.get('id')}")
                if user_info.get('created_at'):
                     output_lines.append(f"- Account Created: {user_info['created_at']}")

            tweets = data.get('tweets', [])
            output_lines.append(f"\n**Recent Tweets (up to {MAX_ITEMS}):**")
            if not tweets:
                output_lines.append("- No tweets found in fetched data.")
            for i, t in enumerate(tweets[:MAX_ITEMS]):
                media_info = f" (Media Attached: {len(t.get('media', []))})" if t.get('media') else ""
                output_lines.append(
                    f"- Tweet {i+1} ({t.get('created_at', 'N/A')}):\n"
                    f"  Text: {t.get('text', '[No Text]')[:TEXT_SNIPPET_LENGTH]}{'...' if len(t.get('text', '')) > TEXT_SNIPPET_LENGTH else ''}\n"
                    f"  Metrics: Likes={t.get('metrics', {}).get('like_count', 0)}, "
                    f"Retweets={t.get('metrics', {}).get('retweet_count', 0)}, "
                    f"Replies={t.get('metrics', {}).get('reply_count', 0)}{media_info}"
                )
            # Add overall media analysis summary if needed, although it's also sent separately
            # output_lines.append(f"\n- Total image analyses from this user: {len(data.get('media_analysis', []))}")

        elif platform == 'reddit':
            output_lines.append(f"### Reddit Data Summary for u/{username}")
            stats = data.get('stats', {})
            output_lines.append(
                f"- Stats Overview: Submissions={stats.get('total_submissions', 0)}, "
                f"Comments={stats.get('total_comments', 0)}, "
                f"Media Posts={stats.get('submissions_with_media', 0)}, "
                f"Avg Sub Score={stats.get('avg_submission_score', 0):.1f}, "
                f"Avg Comment Score={stats.get('avg_comment_score', 0):.1f}"
            )

            submissions = data.get('submissions', [])
            output_lines.append(f"\n**Recent Submissions (up to {MAX_ITEMS}):**")
            if not submissions:
                 output_lines.append("- No submissions found.")
            for i, s in enumerate(submissions[:MAX_ITEMS]):
                media_info = f" (Media Attached: {len(s.get('media', []))})" if s.get('media') else ""
                text_preview = (s.get('text') or "")[:TEXT_SNIPPET_LENGTH]
                text_info = f"\n  Text Preview: {text_preview}{'...' if len(s.get('text', '')) > TEXT_SNIPPET_LENGTH else ''}" if text_preview else ""
                output_lines.append(
                    f"- Submission {i+1} in r/{s.get('subreddit', 'N/A')} ({s.get('created_utc', 'N/A')}):\n"
                    f"  Title: {s.get('title', '[No Title]')}\n"
                    f"  Score: {s.get('score', 0)}{media_info}"
                    f"{text_info}"
                )

            comments = data.get('comments', [])
            output_lines.append(f"\n**Recent Comments (up to {MAX_ITEMS}):**")
            if not comments:
                 output_lines.append("- No comments found.")
            for i, c in enumerate(comments[:MAX_ITEMS]):
                output_lines.append(
                    f"- Comment {i+1} in r/{c.get('subreddit', 'N/A')} ({c.get('created_utc', 'N/A')}):\n"
                    f"  Text: {c.get('text', '[No Text]')[:TEXT_SNIPPET_LENGTH]}{'...' if len(c.get('text', '')) > TEXT_SNIPPET_LENGTH else ''}\n"
                    f"  Score: {c.get('score', 0)}"
                )
            # output_lines.append(f"\n- Total image analyses from this user: {len(data.get('media_analysis', []))}")

        elif platform == 'hackernews':
            output_lines.append(f"### Hacker News Data Summary for {username}")
            stats = data.get('stats', {})
            output_lines.append(
                f"- Stats Overview: Submissions={stats.get('total_submissions', 0)}, "
                f"Avg Points={stats.get('average_points', 0):.1f}"
            )
            submissions = data.get('submissions', [])
            output_lines.append(f"\n**Recent Submissions (up to {MAX_ITEMS}):**")
            if not submissions:
                output_lines.append("- No submissions found.")
            for i, s in enumerate(submissions[:MAX_ITEMS]):
                 text_preview = (s.get('text') or "")[:TEXT_SNIPPET_LENGTH]
                 text_info = f"\n  Text Preview: {text_preview}{'...' if len(s.get('text', '')) > TEXT_SNIPPET_LENGTH else ''}" if text_preview else ""
                 url_info = f"\n  URL: {s.get('url', 'N/A')}" if s.get('url') else ""
                 output_lines.append(
                     f"- Submission {i+1} ({s.get('created_at', 'N/A')}):\n"
                     f"  Title: {s.get('title', '[No Title]')}\n"
                     f"  Points: {s.get('points', 0)}, Comments: {s.get('num_comments', 0)}"
                     f"{url_info}{text_info}"
                 )

        elif platform == 'bluesky':
            output_lines.append(f"### Bluesky Data Summary for {username}") # Assuming username includes handle
            stats = data.get('stats', {})
            output_lines.append(
                 f"- Stats Overview: Posts={stats.get('total_posts', 0)}, "
                 f"Media Posts={stats.get('posts_with_media', 0)}, "
                 f"Avg Likes={stats.get('avg_likes', 0):.1f}, "
                 f"Avg Reposts={stats.get('avg_reposts', 0):.1f}"
            )
            posts = data.get('posts', [])
            output_lines.append(f"\n**Recent Posts (up to {MAX_ITEMS}):**")
            if not posts:
                 output_lines.append("- No posts found.")
            for i, p in enumerate(posts[:MAX_ITEMS]):
                 media_info = f" (Media Attached: {len(p.get('media', []))})" if p.get('media') else ""
                 # Ensure created_at is formatted correctly if it's already a string
                 created_at_str = p.get('created_at', 'N/A')
                 if isinstance(created_at_str, datetime):
                     created_at_str = created_at_str.isoformat()

                 output_lines.append(
                     f"- Post {i+1} ({created_at_str}):\n"
                     f"  Text: {p.get('text', '[No Text]')[:TEXT_SNIPPET_LENGTH]}{'...' if len(p.get('text', '')) > TEXT_SNIPPET_LENGTH else ''}\n"
                     f"  Likes: {p.get('likes', 0)}, Reposts: {p.get('reposts', 0)}{media_info}"
                 )
            # output_lines.append(f"\n- Total image analyses from this user: {len(data.get('media_analysis', []))}")

        else:
            # Fallback for any other platform if added later
            output_lines.append(f"### {platform.capitalize()} Data Summary for {username}")
            output_lines.append(f"- Raw Data Preview: {str(data)[:500]}...") # Very basic preview

        return "\n".join(output_lines)
        
        
        if platform == 'hackernews':
            return f"Hacker News data for {username}:\n" + "\n".join(
                f"- Submission: {s['title']}\n  Points: {s['points']}, Comments: {s['num_comments']}"
                for s in data.get('submissions', [])[:5]
            )
        
        if platform == 'bluesky':
            posts = "\n".join(
            f"- Post: {p['text']}\n  Likes: {p['likes']}, Reposts: {p['reposts']}"
            for p in data.get('posts', [])[:5]
        )
        return f"Bluesky data for {username}:\n{posts}"

    def _call_openrouter(self, json_data: dict):
        try:
            response = self.openrouter.post("/chat/completions", json=json_data)
            response.raise_for_status()
            self._analysis_response = response
        except Exception as e:
            logger.error(f"OpenRouter API Error: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response content: {e.response.text}")
            self._analysis_exception = e

    def _save_output(self, content: str, format_type: str = "markdown"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_dir / 'outputs'
        
        try:
            if format_type == "json":
                filename = output_dir / f"analysis_{timestamp}.json"
                data = {
                    "timestamp": timestamp,
                    "content": content,
                    "format": "json"
                }
                filename.write_text(json.dumps(data, indent=2), encoding='utf-8')
            else:
                filename = output_dir / f"analysis_{timestamp}.md"
                filename.write_text(content, encoding='utf-8')
            
            self.console.print(f"[green]Analysis saved to: {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to save output: {str(e)}[/red]")

    def run(self):
        self.console.print(Panel(
            "[bold blue]Social Media Analysis[/bold blue]\n"
            "This tool analyses user activity across multiple platforms.",
            border_style="blue"
        ))
        
        while True:
            self.console.print("\n[bold cyan]Options:[/bold cyan]")
            self.console.print("1. Twitter\n2. Reddit\n3. HackerNews\n4. Bluesky\n5. Cross-Platform\n6. Exit")
        
            choice = Prompt.ask("Select").strip()
            if choice == "6":
                break
                
            try:
                platforms = {}
                
                # Twitter selection
                if choice in ["1", "5"]:
                    twitter_users = Prompt.ask("Twitter usernames (comma-separated, without @)", default="").strip()
                    if twitter_users:
                        platforms['twitter'] = [u.strip() for u in twitter_users.split(',') if u.strip()]
                
                # Reddit selection
                if choice in ["2", "5"]:
                    reddit_users = Prompt.ask("Reddit usernames (comma-separated, without u/)", default="").strip()
                    if reddit_users:
                        platforms['reddit'] = [u.strip() for u in reddit_users.split(',') if u.strip()]
                
                # HackerNews selection
                if choice in ["3", "5"]:
                    hn_users = Prompt.ask("HackerNews usernames (comma-separated)", default="").strip()
                    if hn_users:
                        platforms['hackernews'] = [u.strip() for u in hn_users.split(',') if u.strip()]

                # Bluesky selection
                if choice in ["4", "5"]:
                    bluesky_users = Prompt.ask("Bluesky handles (comma-separated) remember .bsky.social", default="").strip()
                    if bluesky_users:
                        platforms['bluesky'] = [u.strip() for u in bluesky_users.split(',') if u.strip()]

                if not platforms:
                    self.console.print("[yellow]No valid platforms selected[/yellow]")
                    continue

                self._run_analysis_loop(platforms)
            
            except KeyboardInterrupt:
                if Confirm.ask("\nExit program?"):
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                if not Confirm.ask("Try again?"):
                    break

    def process_stdin(self):
        try:
            input_data = json.load(sys.stdin)
            platforms = input_data.get("platforms", {})
            query = input_data.get("query", "")
            output_format = input_data.get("format", "markdown")
            
            if not platforms or not query:
                raise ValueError("Invalid input format")
                
            for platform, value in platforms.items():
                if platform in ['twitter', 'reddit', 'hackernews'] and isinstance(value, str):
                    platforms[platform] = [value]
                    
            if analysis := self.analyse(platforms, query):
                self._save_output(analysis, output_format)
                return
                
        except Exception as e:
            sys.stderr.write(f"Error: {str(e)}\n")
            sys.exit(1)

    def _run_analysis_loop(self, platforms: Dict[str, Union[str, List[str]]]):
        platform_labels = []
        if 'twitter' in platforms:
            platform_labels.append(f"Twitter: {', '.join(['@'+u for u in platforms['twitter']])}")
        if 'reddit' in platforms:
            platform_labels.append(f"Reddit: {', '.join(['u/'+u for u in platforms['reddit']])}")
        if 'hackernews' in platforms:
            platform_labels.append(f"HN: {', '.join(platforms['hackernews'])}")
        
        platform_info = " | ".join(platform_labels)
        
        self.console.print(Panel(
            f"Analysing: {platform_info}\nCommands: exit, refresh, help",
            title="Analysis Session",
            border_style="cyan"
        ))

        while True:
            try:
                query = Prompt.ask("\nAnalysis query").strip()
                if not query:
                    continue
                
                if query.lower() == 'exit':
                    break
                if query.lower() == 'refresh':
                    with self.progress:
                        refresh_task = self.progress.add_task(
                            "[yellow]Refreshing data...", 
                            total=sum(len(v) if isinstance(v, list) else 1 for v in platforms.values())
                        )
                        for platform, usernames in platforms.items():
                            if isinstance(usernames, str):
                                usernames = [usernames]
                            for username in usernames:
                                getattr(self, f'fetch_{platform}')(username, force=True)
                                self.progress.advance(refresh_task)
                        self.progress.remove_task(refresh_task)
                    self.console.print("[green]Data refreshed[/green]")
                    continue
                if query.lower() == 'help':
                    self.console.print(Panel(
                        "Available commands:\n"
                        "- exit: End current session\n"
                        "- refresh: Force fresh data fetch\n"
                        "- help: Show this help\n"
                        "- Any other text: Analysis query",
                        title="Help",
                        border_style="blue"
                    ))
                    continue

                with self.progress:
                    if analysis := self.analyse(platforms, query):
                        self.console.print(Panel(
                            Markdown(analysis),
                            border_style="green"
                        ))
                        if args.format:
                            self._save_output(analysis, args.format)

            except KeyboardInterrupt:
                if Confirm.ask("\nExit analysis session?"):
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Media Text and Image Analysis")
    parser.add_argument('--stdin', action='store_true', help="Read input from stdin as JSON")
    parser.add_argument('--format', choices=['json', 'markdown'], default='markdown',help="Output format")
    args = parser.parse_args()

    analyser = SocialOSINTLM()
    if args.stdin:
        analyser.process_stdin()
    else:
        analyser.run()
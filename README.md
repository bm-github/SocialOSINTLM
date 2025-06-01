# 🚀 SocialOSINTLM

**SocialOSINTLM** is a powerful Python-based tool designed for Open Source Intelligence (OSINT) gathering and analysis. It aggregates and analyzes user activity across multiple social media platforms, including **Twitter / X, Reddit, Hacker News (via Algolia), and Mastodon (multi-instance),  Bluesky**. Leveraging AI through OpenAI-compatible APIs (e.g., OpenRouter, OpenAI, self-hosted models), it provides comprehensive insights into user engagement, content themes, behavioral patterns, and media content analysis.

## 🌟 Key Features

✅ **Multi-Platform Data Collection:** Aggregates data from Twitter/X, Reddit, Bluesky, Hacker News (via Algolia API), and Mastodon.  .

✅ **AI-Powered Analysis:** Utilises configurable models via OpenAI-compatible APIs for sophisticated text and image analysis.

✅ **Structured AI Prompts:** Employs detailed system prompts for objective, evidence-based analysis focusing on behavior, semantics, interests, and communication style.

✅ **Vision-Capable Image Analysis:** Analyzes downloaded images (`JPEG, PNG, GIF, WEBP`) for OSINT insights using a vision-enabled LLM, focusing on objective details (setting, objects, people, text, activity).

✅ **Efficient Media Handling:** Downloads media, stores it locally, handles platform-specific authentication (e.g., Twitter Bearer, Bluesky JWT for CDN), processes Reddit galleries, and resizes large images (max 1536x1536 recommended for many models) for analysis.

✅ **Cross-Account Comparison:** Analyze profiles across multiple selected platforms simultaneously.

✅ **Intelligent Rate Limit Handling:** Detects API rate limits (especially detailed for Twitter & LLM APIs, showing reset times), provides informative feedback, and prevents excessive requests. Raises `RateLimitExceededError`.

✅ **Robust Caching System:** Caches fetched data for 24 hours (`data/cache/`) to reduce API calls and speed up subsequent analyses. Media files are cached in `data/media/`.

✅ **Offline Mode (`--offline`):** Run analysis using only locally cached data, ignores cache expiry, skipping all external network requests (social platforms, media downloads, *new* vision analysis).

✅ **Interactive CLI:** User-friendly command-line interface with rich formatting (`rich`) for platform selection, user input, and displaying results.

✅ **Programmatic/Batch Mode:** Supports input via JSON from stdin for automated workflows (`--stdin`).

✅ **Configurable Fetch Limits:** Fetches a defined number of recent items per platform (e.g., 50 tweets, 50 Reddit submissions/comments, 50 HN items, ~40 Mastodon/Bluesky posts per API call) to balance depth and API usage.

✅ **Detailed Logging:** Logs errors and operational details to `analyzer.log`.

✅ **Environment Variable Configuration:** Easy setup using environment variables or a `.env` file, and a JSON file for Mastodon instances.

```mermaid
flowchart TD
    %% Initialization
    A([Start SocialOSINTLM]) --> AA{{Setup Directories & API Clients
    Verify Environment}}:::setupClass

    %% Mode Selection
    AA --> B{Interactive or
    Stdin Mode?}:::decisionClass

    %% Interactive Mode Path
    B -->|Interactive| B1([Prompt Auto-Save Setting]):::inputClass
    B1 --> C[/Display Platform Menu/]:::menuClass
    C --> D{Platform
    Selection}:::decisionClass

    %% Platform-Specific Branches
    D -->|Twitter| E1([Twitter]):::twitterClass
    D -->|Reddit| E2([Reddit]):::redditClass
    D -->|HackerNews| E3([HackerNews]):::hnClass
    D -->|Bluesky| E4([Bluesky]):::bskyClass
    D -->|Mastodon| E5([Mastodon]):::MastodonClass
    D -->|Cross-Platform| E6([Multiple Platforms]):::multiClass

    %% Stdin Mode Path
    B -->|Stdin| F([Parse JSON Input]):::inputClass
    F --> GA([Get Auto-Save Setting]):::inputClass
    GA --> G([Extract Platforms & Query]):::inputClass

    %% Analysis Loop Entry Points
    E1 & E2 & E3 & E4 & E5 & E6--> H([Enter Analysis Loop]):::loopClass
    G --> J([Run Analysis]):::analysisClass

    %% Command Processing in Analysis Loop
    H -->|Query Input| I{Command
    Type}:::decisionClass
    I -->|Analysis Query| J
    I -->|exit| Z([End Session]):::endClass
    I -->|refresh| Y([Force Refresh Cache]):::refreshClass
    Y --> H

    %% Data Fetching and Caching
    J --> K{Cache
    Available?}:::cacheClass
    K -->|Yes| M([Load Cached Data]):::cacheClass
    K -->|No| L([Fetch Platform Data]):::apiClass

    %% API & Rate Limit Handling Subgraph
    subgraph API_Handling [API & Rate Limit Handling]
        direction TB
        L --> L1{Rate
        Limited?}:::errorClass
        L1 -->|Yes| L2([Handle Rate Limit]):::errorClass
        L2 --> L5([Abort or Retry]):::errorClass
        L1 -->|No| L3([Extract Text & URLs]):::dataClass
        L3 --> L4([Save to Cache]):::apiClass
    end

    L4 --> M

    %% Parallel Processing Paths
    M --> N([Process Text Data]):::textClass
    M --> O([Process Media Data]):::mediaClass

    %% Media Analysis Pipeline Subgraph
    subgraph Media_Analysis [Media Analysis Pipeline]
        direction TB
        O --> P([Download Media Files]):::mediaClass
        P --> PA{File Exists}:::decisionClass
        PA -->|Yes| Q1([Load existing cached File]):::mediaClass
        PA -->|No| Q([Image Analysis via LLM]):::llmClass

    end
    Q --> R([Collect Media Analysis]):::mediaClass
    Q1 --> R

    %% Text Formatting and Combining Results
    N --> S([Format Platform Text]):::textClass

    R & S --> T([Combine All Data]):::dataClass

    %% Final Analysis and Output
    T --> U([Call Analysis LLM with Query]):::llmClass
    U --> V([Format Analysis Results]):::outputClass
     %% Auto-Save Decision
    V --> V1{Auto-Save 
    Enabled?}:::decisionClass

    %% Handle Saving
    V1 -->|Yes| WA([Save Results Automatically]):::outputClass
    WA --> H
    V1 -->|No| WB{Prompt User to Save?}:::decisionClass
    WB --> |Yes|WC([Save Results]):::outputClass
    WC --> H
    WB --> |No| H

    %% Result display for each save option
    GA --> V1
    %% Custom Styling
    classDef defaultClass fill:#FFFFFF,stroke:#333,stroke-width:1px,color:#000
    classDef menuClass fill:#BBDEFB,stroke:#0D47A1,stroke-width:2px,color:#000
    classDef decisionClass fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#000
    classDef twitterClass fill:#1DA1F2,stroke:#0D47A1,stroke-width:2px,color:#FFF
    classDef redditClass fill:#FF5700,stroke:#8D2202,stroke-width:2px,color:#FFF
    classDef hnClass fill:#FF6600,stroke:#7F3300,stroke-width:2px,color:#FFF
    classDef bskyClass fill:#66BB6A,stroke:#1B5E20,stroke-width:2px,color:#FFF
    classDef MastodonClass fill:#9575CD,stroke:#4527A0,stroke-width:2px,color:#FFF %% Changed color for Mastodon
    classDef multiClass fill:#4DB6AC,stroke:#004D40,stroke-width:2px,color:#FFF
    classDef inputClass fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#000
    classDef loopClass fill:#CE93D8,stroke:#6A1B9A,stroke-width:2px,color:#000
    classDef analysisClass fill:#BBDEFB,stroke:#1565C0,stroke-width:2px,color:#000
    classDef endClass fill:#FFCDD2,stroke:#C62828,stroke-width:2px,color:#000
    classDef refreshClass fill:#80CBC4,stroke:#004D40,stroke-width:2px,color:#000
    classDef cacheClass fill:#B2EBF2,stroke:#006064,stroke-width:2px,color:#000
    classDef apiClass fill:#C5E1A5,stroke:#33691E,stroke-width:2px,color:#000
    classDef errorClass fill:#E57373,stroke:#B71C1C,stroke-width:2px,color:#000
    classDef dataClass fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px,color:#000
    classDef textClass fill:#90CAF9,stroke:#1565C0,stroke-width:2px,color:#000
    classDef mediaClass fill:#F48FB1,stroke:#AD1457,stroke-width:2px,color:#000
    classDef llmClass fill:#FFCC80,stroke:#EF6C00,stroke-width:2px,color:#000
    classDef outputClass fill:#F0F4C3,stroke:#827717,stroke-width:2px,color:#000
    classDef setupClass fill:#D1C4E9,stroke:#4527A0,stroke-width:2px,color:#000

    %% Style all nodes with default class if not otherwise specified
    class A,AA,B defaultClass
```

*Flowchart Description Note:* In **Offline Mode (`--offline`)**, the "Fetch Platform Data" step and the "Download Media File" step within the Media Analysis Pipeline are *bypassed* if the data/media is not already in the cache. Analysis proceeds only with available cached information.

## 🛠 Installation

### Prerequisites
*   **Python 3.8+**
*   Pip (Python package installer)

### Steps
1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/bm-github/SocialOSINTLM.git
    cd SocialOSINTLM
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes: `httpx`, `tweepy`, `praw`, `Mastodon.py`, `beautifulsoup4`, `rich`, `Pillow`, `atproto`, `python-dotenv`, `openai`)*

3.  **Set up Configuration:**

    **a. Environment Variables (`.env` file):**
    Create a `.env` file in the project root or export the following environment variables:

    ```dotenv
    # .env

    # --- LLM Configuration (Required) ---
    LLM_API_KEY="your_llm_api_key"
    LLM_API_BASE_URL="https://api.example.com/v1" # e.g., https://openrouter.ai/api/v1
    ANALYSIS_MODEL="your_text_analysis_model_name"
    IMAGE_ANALYSIS_MODEL="your_vision_model_name"

    # --- Optional: OpenRouter Specific Headers (if LLM_API_BASE_URL is OpenRouter) ---
    # OPENROUTER_REFERER="http://localhost:3000"
    # OPENROUTER_X_TITLE="SocialOSINTLM"

    # --- Platform API Keys (as needed) ---
    # Twitter/X
    TWITTER_BEARER_TOKEN="your_twitter_v2_bearer_token"

    # Reddit
    REDDIT_CLIENT_ID="your_reddit_client_id"
    REDDIT_CLIENT_SECRET="your_reddit_client_secret"
    REDDIT_USER_AGENT="YourAppName/1.0 by YourUsername"

    # Bluesky
    BLUESKY_IDENTIFIER="your-handle.bsky.social"
    BLUESKY_APP_SECRET="xxxx-xxxx-xxxx-xxxx" # App Password

    # --- Mastodon Configuration File Path ---
    # Path to your Mastodon JSON config. Defaults to "mastodon_instances.json" if not set.
    # MASTODON_CONFIG_FILE="config/my_mastodon_servers.json"
    ```
    *Note: HackerNews does not require API keys.*

    **b. Mastodon Instance Configuration (JSON file):**
    If using Mastodon, create a JSON file (e.g., `mastodon_instances.json` in the root directory, or specify path in `.env` via `MASTODON_CONFIG_FILE`).

    **Example `mastodon_instances.json`:**
    ```json
    [
      {
        "name": "Mastodon.Social (Default for Lookups)",
        "api_base_url": "https://mastodon.social",
        "access_token": "YOUR_ACCESS_TOKEN_FOR_MASTODON_SOCIAL",
        "is_default_lookup_instance": true
      },
      {
        "name": "Tech Instance",
        "api_base_url": "https://example2.org",
        "access_token": "YOUR_ACCESS_TOKEN_FOR_OTHER_ORG"
      },
      {
        "name": "Another Server",
        "api_base_url": "https://mastodon.example.net",
        "access_token": "YOUR_ACCESS_TOKEN_FOR_EXAMPLE_NET"
      }
    ]
    ```
    *   **`name`**: A user-friendly name for the instance (optional).
    *   **`api_base_url`**: **Required.** The full base URL of the Mastodon instance (e.g., `https://mastodon.social`).
    *   **`access_token`**: **Required.** Your application's access token for this specific instance.
    *   **`is_default_lookup_instance`**: (Optional, boolean) If `true`, this instance's client will be used for looking up users on Mastodon instances not explicitly listed in this config (federated lookup). **Only one instance should be marked as `true`.** If none are marked, the first successfully initialized client may be used as a fallback.

## 🚀 Usage

### Interactive Mode
Run the script without arguments to start the interactive CLI session:
```bash
python socialosintlm.py
```
Add the `--offline` flag to run the session using only cached data:
```bash
python socialosintlm.py --offline
```
1.  You'll be prompted to select platform(s).
2.  Enter the username(s) for each selected platform (comma-separated if multiple).
    *   **Twitter:** Usernames *without* the leading `@`.
    *   **Reddit:** Usernames *without* the leading `u/`.
    *   **Hacker News:** Usernames as they appear.
    *   **Bluesky:** Full handles including `.bsky.social` (or custom domain).
    *   **Mastodon:** Full handles in `user@instance.domain` format. If an instance is missing for a username and a default Mastodon lookup instance is configured (via `is_default_lookup_instance: true` in your JSON), the tool will prompt to assume the user is on that default instance.
3.  Once platforms/users are selected, you enter an analysis loop for that session. Enter your analysis queries (e.g., "analyze recent activity patterns", "Identify key interests", "Assess communication style").
4.  **Commands within the analysis loop:**
    *   `refresh`: Clears the cache for the current users/platforms and fetches fresh data. **Note: This command is disabled in offline mode (`--offline`).**
    *   `help`: Displays available commands.
    *   `exit`: Exits the current analysis session and returns to the platform selection menu.
    *   Press `Ctrl+C` to potentially exit the program (will prompt for confirmation).
5.  **Offline Mode Behavior:** In offline mode, the tool will only load data from the local cache (`data/cache/`). If no cache exists for a requested user/platform, analysis for that target will be skipped (a warning will be shown). No new data is fetched from social platforms, and *no new media is downloaded or analyzed*.

### Programmatic Mode (via Stdin)
Provide input as a JSON object via standard input using the `--stdin` flag. This is useful for scripting or batch processing.

```bash
echo '{
  "platforms": {
    "twitter": ["someTwitterUser", "anotherXAccount"],
    "reddit": "aRedditUsername",
    "mastodon": [
      "user1@mastodon.social",
      "researcher@example2.org",
      "curious@some.other.instance"
    ],
    "bluesky": "handle.bsky.social",
    "hackernews": "pg"
  },
  "query": "Analyze the primary topics of discussion and any indications of technical expertise across these accounts. Are there notable differences in communication style between platforms for the same individual, if applicable?"
}' | python socialosintlm.py --stdin
```
Combine with `--offline` to use only cached data:
```bash
echo '{
  "platforms": {
    "twitter": ["user1", "user2"],
    "reddit": ["user3"]
  },
  "query": "Summarize cached activity."
}' | python socialosintlm.py --stdin --offline
```
When using `--stdin --offline`, only cached data will be used. If a platform/user has no cache entry, it will be skipped. The tool will exit with a non-zero status code if *no* data could be loaded for *any* requested target due to missing cache entries.

### Command-line Arguments
*   `--stdin`: Read analysis configuration from standard input as a JSON object.
*   `--format [json|markdown]`: Specifies the output format when saving results (default: `markdown`). Also affects output format in `--stdin` mode if `--no-auto-save` is not used.
*   `--no-auto-save`: Disable automatic saving of reports.
    *   In interactive mode: Prompts the user whether to save the report and in which format.
    *   In stdin mode: Prints the report directly to standard output instead of saving to a file.
*   `--log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`: Set the logging level (default: `WARNING`).
*   `--offline`: Run in offline mode. Uses only cached data, no new API calls to social platforms or for new media downloads/vision model analysis.

## ⚡ Cache System
*   **Text/API Data:** Fetched platform data is cached for **24 hours** in `data/cache/` as JSON files (`{platform}_{username}.json`). This minimizes redundant API calls.
*   **Media Files:** Downloaded images and media are stored in `data/media/` using hashed filenames (`{url_hash}.media` with actual extension). These are not automatically purged by the 24-hour cache expiry but are reused if the same URL is encountered.
*   In **Offline Mode (`--offline`)**, new data is *not* fetched, and the cache files are *not* updated or extended. The tool relies purely on the existing cache contents. New media files are *not* downloaded.
*   Use the `refresh` command in interactive mode (online mode only) to force a bypass of the cache for the current session.
*   Use the "Purge Data" option in the main menu to clear cache, media, or output reports.

## 🔍 Error Handling & Logging
*   **Rate Limits:** Detects API rate limits. For Twitter, Mastodon, and some LLM providers, it attempts to display the reset time and estimated wait duration. For others, it provides a general rate limit message. The specific `RateLimitExceededError` is raised internally. **Note:** Rate limit handling is bypassed in offline mode as no API calls are made.
*   **API Errors:** Handles common platform-specific errors (e.g., user not found, forbidden access, general request issues) during online fetching. **Note:** These errors are avoided in offline mode as fetching is skipped.
*   **LLM API Errors:** Handles errors from the LLM API (e.g., authentication, rate limits, bad requests), providing informative messages.
*   **Media Download Errors:** Logs issues during media download or processing (online mode only).
*   **Offline Mode Specifics:** In offline mode, if cache is missing for a requested target, a warning is logged and the target is skipped for analysis. No errors related to network connectivity or API issues will occur.
*   **Logging:** Detailed errors and warnings are logged to `analyzer.log`. The log level can be configured using the `--log-level` argument.

## 🤖 AI Analysis Details
*   **Text Analysis:**
    *   Uses the model specified by `ANALYSIS_MODEL` via the configured `LLM_API_BASE_URL`.
    *   Receives **formatted summaries** of fetched data (user info, stats, recent post/comment text snippets, media presence indicators) per platform, *not* raw API dumps.
    *   Guided by a detailed **system prompt** focusing on objective, evidence-based analysis across domains: Behavioral Patterns, Semantic Content, Interests/Network, Communication Style.
    *   **Offline Mode Impact:** The LLM is informed via the system prompt that it is running in offline mode and analysis is based *only* on potentially stale cached data. It will not have access to real-time or newly generated data.
*   **Image Analysis:**
    *   Uses the vision-capable model specified by `IMAGE_ANALYSIS_MODEL` via the configured `LLM_API_BASE_URL`.
    *   Images larger than 1536x1536 are resized before analysis. Animated GIFs use their first frame. Images are converted to a common format (e.g., JPEG) if necessary.
    *   Guided by a specific **prompt** requesting objective identification of key OSINT-relevant elements (setting, objects, people details, text, activity, overall theme). Avoids speculation.
    *   **Offline Mode Impact:** Image analysis is **only performed if the image file was already downloaded and cached** in a previous online run. New media linked in cached posts will *not* be downloaded or analyzed visually when `--offline` is used. The LLM is aware that visual context may be missing for some data points.
*   **Integration:** The final text analysis incorporates insights derived from both the formatted text data summaries and the individual image analysis reports *that were available from the cache*.

## 📸 Media Processing Details
*   Downloads media files (images: `JPEG, PNG, GIF, WEBP`; some videos might be downloaded but not analyzed visually by default) linked in posts/tweets. **Note:** This step is skipped in Offline Mode (`--offline`) if the media is not already cached.
*   Stores files locally in `data/media/`.
*   Handles platform-specific access during download (online mode).
*   Analyzes valid downloaded images using the vision LLM. **Note:** This step is skipped in Offline Mode if the image file is not in the local cache.

## 🔒 Security Considerations
*   **API Keys:** Requires potentially sensitive API keys and secrets (e.g., `LLM_API_KEY`, platform tokens, Mastodon instance tokens in the JSON config) stored in environment variables or in a `.env` file and `mastodon_instances.json`. Ensure these files are secured and added to `.gitignore`. LLM keys/URLs are still needed even in offline mode as the analysis itself is performed by the LLM.
*   **Data Caching:** Fetched data and downloaded media are stored locally in the `data/` directory. Be mindful of the sensitivity of the data being analyzed and secure the directory appropriately. **In offline mode, this cache is the *only* data source.**
*   **Terms of Service:** Ensure your use of the tool complies with the Terms of Service of each social media platform and your chosen LLM API provider. Automated querying can be subject to restrictions. Using offline mode may mitigate some ToS concerns related to excessive querying, but does not negate ToS related to data storage or analysis.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests, report issues, or suggest enhancements via the project's issue tracker.

## 📜 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

# 🚀 SocialOSINTLM

## 📌 Overview
**SocialOSINTLM** is a powerful Python-based tool that aggregates and analyses user activity across multiple social media platforms, including **Twitter / X, Reddit, Hacker News, and Bluesky**. With AI-driven insights, it provides a comprehensive look into user engagement, trends, and media content.

## 🌟 Features
✅ Multi-platform data collection (Twitter, Reddit, Hacker News, Bluesky)  
✅ AI-powered analysis with OpenRouter API  
✅ Image analysis for media content  
✅ Cross-account comparison  
✅ Rate limit handling with informative feedback  
✅ Caching system for efficient data retrieval  
✅ Interactive CLI with rich formatting  
✅ Supports both interactive and programmatic usage  

```mermaid
flowchart TD
    A[Start Analysis] --> B{Platform Selection}
    B -->|Twitter| C1[Fetch Twitter Data]
    B -->|Reddit| C2[Fetch Reddit Data]
    B -->|HackerNews| C3[Fetch HackerNews Data]
    B -->|Bluesky| C4[Fetch Bluesky Data]
    
    C1 --> D1[Extract User Posts]
    C4 --> D1
    C3 --> D1
    C2 --> D1
    
    C1 --> D2[Extract Media URLs]
    C4 --> D2
    
    D2 --> E1[Download Media]
    E1 --> F[Analyse Images via OpenRouter]
    F --> G[Store Image Analysis]
    
    D1 --> H[Collect Text Data]
    
    subgraph "Media Analysis Path"
        D2
        E1
        F
        G
    end
    
    subgraph "Text-Only Path"
        D1
        H
    end
    
    G --> I[Combine All Data]
    H --> I
    
    I --> J[Create Analysis Prompt]
    J --> K[Send to OpenRouter LLM]
    K --> L[Format and Return Analysis]
    
    L --> M[Save Output as markdown]
    L --> N[Display Results]
```

## 🛠 Installation
### Prerequisites
Ensure you have **Python 3.8+** installed. Then, install the dependencies:

```sh
pip install -r requirements.txt
```

### Environment Variables
Set up the required API keys:
```sh
export TWITTER_BEARER_TOKEN='your_token_here'
export REDDIT_CLIENT_ID='your_client_id'
export REDDIT_CLIENT_SECRET='your_client_secret'
export REDDIT_USER_AGENT='your_user_agent'
export BLUESKY_IDENTIFIER='your_bluesky_handle'
export BLUESKY_APP_SECRET='your_bluesky_secret'
export OPENROUTER_API_KEY='your_openrouter_api_key'
export ANALYSIS_MODEL='your_preferred_model'  # OpenRouter-compatible model
export IMAGE_ANALYSIS_MODEL='your_preferred_model'  # OpenRouter-vision compatible model
```

## 🚀 Usage
### Interactive Mode
Run the script without arguments to start an interactive session:
```sh
python socialosimtlm.py
```
📌 Commands:
- Select platform(s) for analysis
- Enter usernames (comma-separated)
- Input analysis queries
- Type `refresh` to force data refresh
- Type `exit` to quit
- Type `help` to display available commands

### Programmatic Mode (Batch Processing)
You can provide input via JSON:
```sh
echo '{"platforms": {"twitter": ["user1"], "reddit": ["user2"]}, "query": "Analyse engagement", "format": "markdown"}' | python socialosintlm.py --stdin
```

### Command-line Arguments
- `--stdin` : Reads JSON input from standard input
- `--format [json|markdown]` : Specifies the output format

## 📊 Output
Results are saved in `data/outputs/` with timestamps:
- `analysis_YYYYMMDD_HHMMSS.md` for markdown
- `analysis_YYYYMMDD_HHMMSS.json` for JSON

## ⚡ Cache System
- Data cached in `data/cache/` for **24 hours**
- Media files stored in `data/media/`
- Format: `{platform}_{username}.json`

## 🔍 Error Handling
- **Rate limits**: Displays reset time and wait duration
- **API errors**: Detailed logs in `analyser.log`
- **Media handling**: Full download with proper authentication

## 🤖 AI Analysis
The tool uses OpenRouter API for:
- **Text Analysis**: Configurable via `ANALYSIS_MODEL` env var
- **Image Analysis**: Configurable via `IMAGE_ANALYSIS_MODEL` env var
- Image analysis automatically resizes large images and provides contextual insights

## 📸 Media Processing
- Downloads and stores media locally
- Supports JPEG, PNG, GIF, and WEBP formats
- Platform-specific authentication for media access
- Proper CDN handling for Bluesky images

## 🔒 Security
🔹 API keys required for all platforms  
🔹 Local caching of data  
🔹 Secure authentication for protected content  

## 🤝 Contributing
We welcome contributions! Feel free to submit pull requests or report issues.

## 📜 Licence
This project is licensed under the **MIT Licence**.

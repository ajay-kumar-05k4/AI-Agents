# Deployment Guide

## Local Testing with OpenAI

1. The `.streamlit/secrets.toml` file is already configured with your API key (gitignored)
2. Run: `streamlit run Chatbot.py`
3. The app will use OpenAI automatically

## Streamlit Cloud Deployment

### Step 1: Add Secrets in Streamlit Cloud

1. Go to your Streamlit Cloud app dashboard
2. Click on "Settings" or "Manage app"
3. Go to "Secrets" tab
4. Add the following secrets:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-3.5-turbo"
```

### Step 2: Deploy

1. Push your code to GitHub
2. Streamlit Cloud will automatically redeploy
3. Your app should now work with OpenAI!

## Environment Variables (Alternative)

You can also use environment variables instead of secrets:

- `OPENAI_API_KEY` - Your OpenAI API key
- `LLM_PROVIDER` - Set to "openai" for cloud, "ollama" for local
- `OPENAI_MODEL` - Model name (default: "gpt-3.5-turbo")

## Security Notes

⚠️ **IMPORTANT**: Never commit API keys to git!
- The `.streamlit/secrets.toml` file is in `.gitignore`
- Always use Streamlit secrets or environment variables for production
- If you accidentally commit a key, rotate it immediately in OpenAI dashboard

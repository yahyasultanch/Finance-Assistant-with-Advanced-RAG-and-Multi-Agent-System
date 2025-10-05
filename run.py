"""
Main entry point for AI Finance Assistant.
Run this to start the Streamlit UI.
"""
import subprocess
import sys
import os
from pathlib import Path

def check_azure_config():
    """Check if Azure OpenAI configuration is set."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_DEPLOYMENT"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    return len(missing) == 0, missing

def main():
    """Main entry point."""
    print("="*60)
    print("AI Investment Research Assistant")
    print("="*60)
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("\n⚠️  Warning: .env file not found!")
        print("Please create .env from .env.example and configure Azure OpenAI credentials")
        print("Example: copy .env.example .env")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check Azure OpenAI config
    print("\n🔍 Checking Azure OpenAI configuration...")
    is_configured, missing_vars = check_azure_config()
    if not is_configured:
        print("❌ Azure OpenAI configuration is incomplete!")
        print("\nMissing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease configure these in your .env file")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    else:
        print("✅ Azure OpenAI configuration found")
    
    # Start Streamlit
    print("\n🚀 Starting Streamlit UI...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/ui_app.py"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure dependencies are installed: pip install -r requirements.txt")
        print("  2. Check Python version: python --version (need 3.12+)")
        print("  3. Verify Streamlit is installed: streamlit --version")

if __name__ == "__main__":
    main()

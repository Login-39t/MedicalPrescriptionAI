#!/usr/bin/env python3
# Simple runner for the clean Flask app
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the clean app
if __name__ == "__main__":
    print("🚀 Starting Clean IBM Granite AI Flask App...")
    print("📁 Working directory:", os.getcwd())
    print("🐍 Python version:", sys.version)
    
    try:
        # Import the clean app
        import granite_clean_flask
        app = granite_clean_flask.app
        
        print("✅ Clean app imported successfully")
        print("🌐 Starting server on http://localhost:5007")
        print("🔧 Mode: Type-safe, error-free simulation")
        
        # Run the app
        app.run(host='0.0.0.0', port=5007, debug=False)
        
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        import traceback
        traceback.print_exc()

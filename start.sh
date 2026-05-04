#!/bin/bash

# SignSight Backend Startup Script
# This script sets up and runs the SignSight backend application

set -e

echo "🚀 SignSight Backend Startup Script"
echo "===================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed"
    exit 1
fi

echo "✓ pip version: $(pip3 --version)"

# Create required directories
echo ""
echo "📁 Creating required directories..."
mkdir -p uploads models static/sign_images reports

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt --upgrade

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found, creating one..."
    cp .env .env.backup 2>/dev/null || true
fi

# Run type checking if available
if command -v pyright &> /dev/null; then
    echo ""
    echo "🔍 Running type checking..."
    pyright . --outputjson || true
elif command -v mypy &> /dev/null; then
    echo ""
    echo "🔍 Running type checking with mypy..."
    mypy main.py mentor_backend.py app.py --ignore-missing-imports --no-implicit-optional || true
fi

# Start the application
echo ""
echo "🎯 Starting SignSight Backend..."
echo "   Server will be available at: http://localhost:5080"
echo ""

# Use gunicorn if available for production, otherwise use Flask dev server
if command -v gunicorn &> /dev/null; then
    echo "📌 Using Gunicorn WSGI server..."
    gunicorn -w 4 -b 0.0.0.0:5080 main:app --timeout 120
else
    echo "📌 Using Flask development server..."
    python3 main.py
fi


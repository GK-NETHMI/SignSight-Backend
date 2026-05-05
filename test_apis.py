#!/usr/bin/env python3
"""
Test Suite for SignSight Backend - All 4 Branches
Quick API endpoint testing to verify system functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:5080"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test an API endpoint"""
    try:
        url = f"{BASE_URL}{endpoint}"
        if method.upper() == "GET":
            response = requests.get(url, timeout=5)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=5)

        status = "✅" if response.status_code < 400 else "⚠️"
        print(f"{status} [{method:4}] {response.status_code} {endpoint:50} {description}")
        return response.status_code < 400
    except Exception as e:
        print(f"❌ [    ] ERROR {endpoint:50} {str(e)[:40]}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*100)
    print("SignSight Backend - API Test Suite".center(100))
    print("="*100 + "\n")

    # Wait for server
    print("Waiting for server to respond...")
    for i in range(10):
        try:
            requests.get(BASE_URL, timeout=1)
            print("✅ Server is responding\n")
            break
        except:
            if i < 9:
                time.sleep(1)
            else:
                print("❌ Server not responding. Is it running?")
                print("   Run: python3 main.py")
                return

    print("-"*100)
    print("BRANCH 1: Audio-to-Sign API".ljust(50) + "Status".rjust(50))
    print("-"*100)
    test_endpoint("POST", "/api/audio-to-sign/text-to-signs",
                  {"text": "nandri"}, "Convert text to signs")
    test_endpoint("GET", "/api/audio-to-sign/get-sign-image/nandri",
                  description="Get sign image")

    print("\n" + "-"*100)
    print("BRANCH 2: Mentor Dashboard API".ljust(50) + "Status".rjust(50))
    print("-"*100)
    test_endpoint("GET", "/api/admin/mentors", description="List mentors")
    test_endpoint("GET", "/api/admin/students", description="List students")
    test_endpoint("GET", "/api/dashboard/overview", description="Dashboard overview")

    print("\n" + "-"*100)
    print("BRANCH 3: Jeran ML Inference (Optional)".ljust(50) + "Status".rjust(50))
    print("-"*100)
    test_endpoint("GET", "/api/ml/health", description="ML health check (optional)")
    test_endpoint("GET", "/api/ml/variant_status", description="ML variant status (optional)")

    print("\n" + "-"*100)
    print("BRANCH 4: Emotion Analysis (Optional)".ljust(50) + "Status".rjust(50))
    print("-"*100)
    test_endpoint("GET", "/api/emotion/status/test", description="Emotion status (optional)")

    print("\n" + "-"*100)
    print("Health & Admin".ljust(50) + "Status".rjust(50))
    print("-"*100)
    test_endpoint("GET", "/", description="Health check")
    test_endpoint("GET", "/api/admin/status", description="Admin status")

    print("\n" + "="*100)
    print("✅ TEST SUITE COMPLETE".center(100))
    print("="*100 + "\n")
    print("Legend:")
    print("  ✅ Success (API responded)")
    print("  ⚠️ Anticipated (404 or expected error)")
    print("  ❌ Failure (connection error)")
    print("\nNote: Branches 3 & 4 may not be available (optional features)")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()


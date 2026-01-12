#!/usr/bin/env python3
"""
Test script for Flask backend API endpoints
"""

import requests
import os
from pathlib import Path

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_audio_to_sign():
    """Test the audio-to-sign endpoint"""
    print("\n=== Testing Audio to Sign ===")
    # Create a dummy audio file for testing
    test_file_path = "test_audio.mp3"
    with open(test_file_path, "wb") as f:
        f.write(b"dummy audio content")

    try:
        with open(test_file_path, "rb") as f:
            files = {"file": (test_file_path, f, "audio/mpeg")}
            response = requests.post(f"{BASE_URL}/audio-to-sign", files=files)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            print("✓ Audio to sign endpoint works")
            return True
        else:
            print(f"✗ Audio to sign endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Audio to sign test failed: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_text_to_sign():
    """Test the text-to-sign endpoint"""
    print("\n=== Testing Text to Sign ===")
    # Create a dummy text file for testing
    test_file_path = "test_text.txt"
    with open(test_file_path, "w") as f:
        f.write("Hello, this is a test text.")

    try:
        with open(test_file_path, "rb") as f:
            files = {"file": (test_file_path, f, "text/plain")}
            response = requests.post(f"{BASE_URL}/text-to-sign", files=files)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            print("✓ Text to sign endpoint works")
            return True
        else:
            print(f"✗ Text to sign endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Text to sign test failed: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_invalid_file_type():
    """Test file type validation"""
    print("\n=== Testing Invalid File Type ===")
    # Create a dummy invalid file
    test_file_path = "test_invalid.xyz"
    with open(test_file_path, "w") as f:
        f.write("invalid file")

    try:
        with open(test_file_path, "rb") as f:
            files = {"file": (test_file_path, f)}
            response = requests.post(f"{BASE_URL}/audio-to-sign", files=files)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 400:
            print("✓ File type validation works")
            return True
        else:
            print(f"✗ File type validation failed")
            return False
    except Exception as e:
        print(f"✗ Invalid file type test failed: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Flask Backend API Test Suite")
    print("=" * 60)
    print(f"\nTesting against: {BASE_URL}")
    print("\nMake sure the Flask server is running!")
    print("Run: python app.py")

    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("Audio to Sign", test_audio_to_sign()))
    results.append(("Text to Sign", test_text_to_sign()))
    results.append(("Invalid File Type", test_invalid_file_type()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        exit(1)


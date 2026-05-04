#!/usr/bin/env python3
"""
SignSight Backend - Complete API Endpoint Reference
Generated automatically from registered routes
"""

from main import app

def print_api_reference():
    """Print all available API endpoints with documentation."""

    print("\n" + "="*80)
    print("SignSight Backend - API Endpoint Reference".center(80))
    print("="*80 + "\n")

    # Get all routes
    routes_by_category = {
        'Health & Status': [],
        'Audio-to-Sign API': [],
        'Mentor Dashboard': [],
        'Other': []
    }

    for rule in app.url_map.iter_rules():
        if 'static' not in str(rule):
            methods = ','.join([m for m in rule.methods if m not in ['HEAD', 'OPTIONS']])
            endpoint = rule.endpoint
            path = str(rule)

            # Categorize
            if path in ['/', '/api/admin/status']:
                routes_by_category['Health & Status'].append((path, methods, endpoint))
            elif '/audio-to-sign' in path:
                routes_by_category['Audio-to-Sign API'].append((path, methods, endpoint))
            elif '/api/' in path:
                routes_by_category['Mentor Dashboard'].append((path, methods, endpoint))
            else:
                routes_by_category['Other'].append((path, methods, endpoint))

    # Documentation
    endpoint_docs = {
        # Health
        '/': 'Service health check',
        '/api/admin/status': 'System status, environment, and configuration info',

        # Audio-to-Sign
        '/api/audio-to-sign/upload-audio': 'Upload .wav audio file → predict sign language',
        '/api/audio-to-sign/upload-video': 'Upload video file → extract audio → predict signs',
        '/api/audio-to-sign/text-to-signs': 'Convert Tamil text directly → sign predictions',
        '/api/audio-to-sign/get-sign-image/<sign_name>': 'Get sign language image/GIF URL',

        # Mentor Dashboard
        '/api/<mentorEmail>/dashboard/users': 'List students assigned to mentor',
        '/api/dashboard/users/<user_id>/summary': 'Get comprehensive student performance summary',
        '/api/dashboard/users/<user_id>/attempts': 'Get paginated student attempt history',
        '/api/dashboard/overview': 'Get global system statistics',
        '/api/mentors': 'Create new mentor account',
        '/api/admin/mentors': 'List all mentors',
        '/api/admin/students': 'List all students',
        '/api/admin/save-mentor-users': 'Assign students to mentor',
        '/api/students': 'Create new student account',
        '/api/students/by-username/<username>': 'Get student by username',
    }

    # Print
    for category in routes_by_category:
        routes = routes_by_category[category]
        if routes:
            print(f"\n📌 {category}")
            print("-" * 80)

            for path, methods, endpoint in sorted(routes):
                doc = endpoint_docs.get(path, 'N/A')
                print(f"\n  {methods:12} {path}")
                print(f"  {'':12} → {doc}")

    print("\n" + "="*80)
    print("Total Endpoints: {}".format(sum(len(v) for v in routes_by_category.values())))
    print("="*80 + "\n")


if __name__ == '__main__':
    print_api_reference()

    # Additional info
    print("\n📚 API Usage Examples")
    print("="*80 + "\n")

    print("1️⃣  Health Check")
    print("   curl http://localhost:5080/")
    print("   → Response: {\"status\": \"ok\", \"service\": \"SignSight Backend\"}\n")

    print("2️⃣  Text to Signs")
    print("   curl -X POST http://localhost:5080/api/audio-to-sign/text-to-signs \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"text\": \"nandri\"}'")
    print("   → Response: Signs with image URLs\n")

    print("3️⃣  Upload Audio")
    print("   curl -X POST http://localhost:5080/api/audio-to-sign/upload-audio \\")
    print("     -F 'audio=@/path/to/audio.wav'")
    print("   → Response: Predicted sign with confidence\n")

    print("4️⃣  Admin Status")
    print("   curl http://localhost:5080/api/admin/status")
    print("   → Response: Service status and DB connectivity\n")

    print("5️⃣  List Mentors")
    print("   curl http://localhost:5080/api/admin/mentors")
    print("   → Response: Array of all mentors\n")

    print("="*80 + "\n")


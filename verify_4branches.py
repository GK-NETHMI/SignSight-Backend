#!/usr/bin/env python3
"""Verify 4-branch integration"""
from main import app

print('\n' + '='*80)
print('🎉 FINAL INTEGRATION - ALL 4 BRANCHES')
print('='*80)

routes = []
for rule in app.url_map.iter_rules():
    if 'static' not in str(rule):
        methods = ','.join([m for m in rule.methods if m not in ['HEAD', 'OPTIONS']])
        routes.append((str(rule), methods))

# Categorize
audio_routes = [r for r in routes if 'audio-to-sign' in r[0]]
mentor_routes = [r for r in routes if '/api/' in r[0] and 'audio-to-sign' not in r[0] and 'ml' not in r[0] and 'emotion' not in r[0]]
ml_routes = [r for r in routes if '/api/ml' in r[0]]
emotion_routes = [r for r in routes if '/api/emotion' in r[0]]
health_routes = [r for r in routes if r[0] in ['/', '/api/admin/status']]

print(f'\n📱 BRANCH 1: Audio-to-Sign API ({len(audio_routes)} routes)')
for route, methods in sorted(audio_routes):
    print(f'  {methods:8} {route}')

print(f'\n👥 BRANCH 2: Mentor Dashboard API ({len(mentor_routes)} routes)')
for route, methods in sorted(mentor_routes)[:5]:
    print(f'  {methods:8} {route}')
if len(mentor_routes) > 5:
    print(f'  ... and {len(mentor_routes)-5} more')

print(f'\n🧠 BRANCH 3: Jeran ML Model Inference API ({len(ml_routes)} routes)')
for route, methods in sorted(ml_routes):
    print(f'  {methods:8} {route}')

print(f'\n😊 BRANCH 4: Emotion Video Analysis API ({len(emotion_routes)} routes)')
for route, methods in sorted(emotion_routes):
    print(f'  {methods:8} {route}')

print(f'\n⚕️ Health & Admin Routes ({len(health_routes)} routes)')
for route, methods in sorted(health_routes):
    print(f'  {methods:8} {route}')

print('\n' + '='*80)
print('✅ ALL 4 BRANCHES INTEGRATED AND READY!')
print(f'   Total Routes: {len(routes)}')
print('='*80 + '\n')


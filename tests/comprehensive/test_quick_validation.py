#!/usr/bin/env python3
"""
Quick Validation Test
Tests key parameter combinations with shorter tracks and timeouts.
"""

import os
import sys
import subprocess
import time


def run_quick_test(name: str, args: list) -> bool:
    """Run a single quick test"""
    print(f"🧪 {name}... ", end="", flush=True)
    
    # Find test tracks
    tracks = []
    for dir_name in ['data/house_tracks/House', 'data/test_tracks', 'House', 'test_tracks']:
        if os.path.exists(dir_name):
            for file in os.listdir(dir_name):
                if file.endswith('.wav'):
                    tracks.append(os.path.join(dir_name, file))
                    if len(tracks) >= 2:
                        break
            if len(tracks) >= 2:
                break
    
    if len(tracks) < 2:
        print("❌ No test tracks found")
        return False
    
    # Build command
    cmd = ['python', 'dj_mix_generator.py'] + args + ['--transition-measures', '2'] + tracks[:2]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Check output
        expected_output = 'dj_transitions_preview.wav' if '--transitions-only' in args else 'dj_mix.wav'
        output_created = os.path.exists(expected_output)
        
        if output_created:
            file_size = os.path.getsize(expected_output)
            os.remove(expected_output)  # Clean up
        else:
            file_size = 0
        
        success = result.returncode == 0 and output_created and file_size > 1000  # At least 1KB
        
        if success:
            print(f"✅ ({duration:.1f}s, {file_size//1024}KB)")
        else:
            print(f"❌ ({duration:.1f}s)")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()[:100]}")
        
        return success
        
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run quick validation tests"""
    print("🎵 DJ Mix Generator - Quick Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Mix", []),
        ("LF Transition", ["--lf-transition"]),
        ("MF Transition", ["--mf-transition"]), 
        ("Match-Track Tempo", ["--tempo-strategy", "match-track"]),
        ("Match-Track + MF", ["--tempo-strategy", "match-track", "--mf-transition"]),
        ("Transitions Only", ["--transitions-only"]),
        ("No Cache", ["--no-cache"]),
        ("High Quality", ["--eq-strength", "0.75"]),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_args in tests:
        if run_quick_test(test_name, test_args):
            passed += 1
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✅ All quick validation tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
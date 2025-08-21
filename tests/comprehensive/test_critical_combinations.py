#!/usr/bin/env python3
"""
Critical Parameter Combinations Test
Tests the most important parameter combinations to ensure core functionality works.
"""

import os
import sys
import subprocess
import time
from typing import Dict, List, Any


class CriticalCombinationTests:
    """Test critical parameter combinations"""
    
    def __init__(self):
        self.test_tracks = self._find_test_tracks()
        self.results = []
        
        # Define critical test combinations
        self.test_combinations = [
            # Basic functionality tests
            {
                'name': 'Default Settings',
                'params': {}
            },
            {
                'name': 'Sequential Tempo + LF Transition',
                'params': {'--tempo-strategy': 'sequential', '--lf-transition': True}
            },
            {
                'name': 'Uniform Tempo + MF Transition', 
                'params': {'--tempo-strategy': 'uniform', '--mf-transition': True}
            },
            {
                'name': 'Match-Track Tempo Strategy',
                'params': {'--tempo-strategy': 'match-track'}
            },
            {
                'name': 'Match-Track + LF Transition',
                'params': {'--tempo-strategy': 'match-track', '--lf-transition': True}
            },
            {
                'name': 'Match-Track + MF Transition',
                'params': {'--tempo-strategy': 'match-track', '--mf-transition': True}
            },
            
            # Quality feature combinations
            {
                'name': 'All Quality Features Disabled',
                'params': {
                    '--no-eq-matching': True,
                    '--no-volume-matching': True, 
                    '--no-peak-alignment': True,
                    '--no-tempo-correction': True
                }
            },
            {
                'name': 'High Quality Settings',
                'params': {
                    '--eq-strength': '0.75',
                    '--transition-measures': '8'
                }
            },
            
            # Advanced feature combinations
            {
                'name': 'Harmonic Mixing',
                'params': {'--reorder-by-key': True}
            },
            {
                'name': 'Transitions Only Preview',
                'params': {'--transitions-only': True}
            },
            {
                'name': 'Full Feature Test',
                'params': {
                    '--tempo-strategy': 'match-track',
                    '--mf-transition': True,
                    '--reorder-by-key': True,
                    '--transition-measures': '4',
                    '--eq-strength': '0.5'
                }
            },
            
            # Different transition lengths
            {
                'name': 'Short Transitions',
                'params': {'--transition-measures': '2'}
            },
            {
                'name': 'Long Transitions', 
                'params': {'--transition-measures': '8'}
            },
            
            # Cache tests
            {
                'name': 'No Cache',
                'params': {'--no-cache': True}
            },
            
            # Edge cases
            {
                'name': 'Minimal EQ Strength',
                'params': {'--eq-strength': '0.1'}
            },
            {
                'name': 'Maximum EQ Strength',
                'params': {'--eq-strength': '1.0'}
            }
        ]
    
    def _find_test_tracks(self) -> List[str]:
        """Find available test tracks"""
        possible_dirs = ['House', 'test_tracks', 'example_tracks']
        tracks = []
        
        for dir_name in possible_dirs:
            if os.path.exists(dir_name):
                for file in os.listdir(dir_name):
                    if file.endswith('.wav'):
                        tracks.append(os.path.join(dir_name, file))
                        if len(tracks) >= 3:
                            break
                if tracks:
                    break
        
        return tracks[:3]
    
    def _build_command(self, test_params: Dict[str, Any]) -> List[str]:
        """Build command from test parameters"""
        cmd = ['python', 'dj_mix_generator.py']
        
        for key, value in test_params.items():
            if key.startswith('--'):
                if value is True:
                    cmd.append(key)
                elif value is not True:  # String value
                    cmd.extend([key, str(value)])
        
        # Add test tracks
        cmd.extend(self.test_tracks[:2])
        
        return cmd
    
    def _run_test(self, test_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test"""
        print(f"🧪 Running: {test_name}")
        
        start_time = time.time()
        cmd = self._build_command(params)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout
                cwd=os.getcwd()
            )
            
            duration = time.time() - start_time
            
            # Check if output file was created
            expected_output = 'dj_mix.wav'
            if params.get('--transitions-only'):
                expected_output = 'dj_transitions_preview.wav'
                
            output_created = os.path.exists(expected_output)
            
            # Get file size if created
            file_size = 0
            if output_created:
                file_size = os.path.getsize(expected_output)
                os.remove(expected_output)  # Clean up
            
            success = result.returncode == 0 and output_created and file_size > 0
            
            test_result = {
                'name': test_name,
                'params': params,
                'command': ' '.join(cmd),
                'success': success,
                'return_code': result.returncode,
                'duration': duration,
                'output_created': output_created,
                'file_size_mb': file_size / (1024 * 1024) if file_size > 0 else 0,
                'error_message': result.stderr if result.returncode != 0 else None
            }
            
            status = "✅ PASS" if success else "❌ FAIL"
            duration_str = f"{duration:.1f}s"
            size_str = f"{test_result['file_size_mb']:.1f}MB" if file_size > 0 else "0MB"
            
            print(f"   {status} ({duration_str}, {size_str})")
            
            if not success and result.stderr:
                print(f"   Error: {result.stderr.strip()[:100]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ TIMEOUT (>3min)")
            return {
                'name': test_name,
                'params': params,
                'success': False,
                'duration': time.time() - start_time,
                'error_message': 'Test timed out'
            }
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            return {
                'name': test_name,
                'params': params,
                'success': False,
                'duration': time.time() - start_time,
                'error_message': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all critical combination tests"""
        if not self.test_tracks:
            print("❌ No test tracks found!")
            return {'total': 0, 'passed': 0, 'failed': 0}
        
        print("🎵 DJ Mix Generator - Critical Combinations Test")
        print(f"📁 Test tracks: {self.test_tracks}")
        print(f"🧪 Running {len(self.test_combinations)} critical tests...")
        print("=" * 60)
        
        for i, test_config in enumerate(self.test_combinations, 1):
            print(f"[{i:2d}/{len(self.test_combinations)}] ", end="")
            
            result = self._run_test(test_config['name'], test_config['params'])
            self.results.append(result)
        
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed
        
        avg_duration = sum(r['duration'] for r in self.results) / total if total > 0 else 0
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 CRITICAL TESTS SUMMARY")
        print("=" * 60)
        print(f"Total tests:    {total}")
        print(f"Passed:         {passed} ({success_rate:.1f}%)")
        print(f"Failed:         {failed}")
        print(f"Avg duration:   {avg_duration:.1f}s")
        
        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result['success']:
                    print(f"   • {result['name']}: {result.get('error_message', 'Unknown error')}")
        
        return {'total': total, 'passed': passed, 'failed': failed, 'success_rate': success_rate}


def main():
    """Run critical combination tests"""
    tester = CriticalCombinationTests()
    summary = tester.run_all_tests()
    
    if summary['failed'] > 0:
        print(f"\n❌ {summary['failed']} critical tests failed!")
        sys.exit(1)
    else:
        print(f"\n✅ All {summary['passed']} critical tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
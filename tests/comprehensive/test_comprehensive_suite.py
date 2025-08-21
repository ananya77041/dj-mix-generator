#!/usr/bin/env python3
"""
Comprehensive Test Suite for DJ Mix Generator
Tests every combination of configuration parameters to ensure compatibility and functionality.
"""

import os
import sys
import subprocess
import itertools
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json


class ComprehensiveTestSuite:
    """Test suite that validates all parameter combinations"""
    
    def __init__(self):
        self.test_tracks = self._find_test_tracks()
        self.results = []
        self.failed_tests = []
        self.skipped_tests = []
        
        # Define all testable parameters (simplified for manageable testing)
        self.parameters = {
            # Tempo strategies
            'tempo_strategy': ['sequential', 'uniform', 'match-track'],
            
            # Transition parameters
            'transition_measures': [2, 4, 8],
            
            # Audio quality features
            'eq_matching': [True, False],
            'volume_matching': [True, False], 
            'peak_alignment': [True, False],
            'eq_strength': [0.25, 0.5, 0.75],  # Only when eq_matching is True
            
            # Frequency transition features
            'lf_transition': [True, False],
            'mf_transition': [True, False],
            
            # Advanced features
            'reorder_by_key': [True, False],
            'transitions_only': [True, False],
            
            # Cache settings
            'use_cache': [True, False]
        }
        
        # Incompatible combinations to skip
        self.skip_combinations = [
            # Can't have both LF and MF transition at same time
            lambda params: params['lf_transition'] and params['mf_transition'],
            # EQ strength only matters when EQ matching is enabled
            lambda params: not params['eq_matching'] and params['eq_strength'] > 0.5,  # Skip only non-default values
        ]
    
    def _find_test_tracks(self) -> List[str]:
        """Find available test tracks"""
        possible_dirs = ['test_tracks', 'House', 'example_tracks']
        tracks = []
        
        for dir_name in possible_dirs:
            if os.path.exists(dir_name):
                for file in os.listdir(dir_name):
                    if file.endswith('.wav'):
                        tracks.append(os.path.join(dir_name, file))
                        if len(tracks) >= 3:  # We only need a few tracks for testing
                            break
                if tracks:
                    break
        
        if len(tracks) < 2:
            print("Warning: Need at least 2 WAV files for testing")
            print(f"Found tracks: {tracks}")
        
        return tracks[:3]  # Use max 3 tracks to keep tests reasonable
    
    def _should_skip_combination(self, params: Dict[str, Any]) -> bool:
        """Check if parameter combination should be skipped"""
        for skip_condition in self.skip_combinations:
            if skip_condition(params):
                return True
        return False
    
    def _generate_command(self, params: Dict[str, Any]) -> List[str]:
        """Generate command line arguments from parameters"""
        cmd = ['python', 'dj_mix_generator.py']
        
        # Tempo strategy
        cmd.extend(['--tempo-strategy', params['tempo_strategy']])
        
        # Transition length - always use measures
        cmd.extend(['--transition-measures', str(params['transition_measures'])])
        
        # Audio quality flags
        if not params['eq_matching']:
            cmd.append('--no-eq-matching')
        elif params['eq_strength'] != 0.5:  # Default is 0.5
            cmd.extend(['--eq-strength', str(params['eq_strength'])])
            
        if not params['volume_matching']:
            cmd.append('--no-volume-matching')
            
        if not params['peak_alignment']:
            cmd.append('--no-peak-alignment')
        
        # Frequency transitions
        if params['lf_transition']:
            cmd.append('--lf-transition')
        elif params['mf_transition']:
            cmd.append('--mf-transition')
        
        # Advanced features
        if params['reorder_by_key']:
            cmd.append('--reorder-by-key')
            
        if params['transitions_only']:
            cmd.append('--transitions-only')
        
        # Cache settings
        if not params['use_cache']:
            cmd.append('--no-cache')
        
        # Add tracks
        cmd.extend(self.test_tracks[:2])  # Use 2 tracks for faster testing
        
        return cmd
    
    def _run_test(self, test_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test with given parameters"""
        start_time = time.time()
        cmd = self._generate_command(params)
        
        try:
            # Run the command with timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1800,  # 30 minute timeout
                cwd=os.getcwd()
            )
            
            duration = time.time() - start_time
            
            # Check if output file was created
            expected_output = 'dj_mix.wav' if not params['transitions_only'] else 'dj_transitions_preview.wav'
            output_created = os.path.exists(expected_output)
            
            # Clean up output file
            if output_created:
                try:
                    os.remove(expected_output)
                except:
                    pass
            
            test_result = {
                'test_id': test_id,
                'params': params,
                'command': ' '.join(cmd),
                'success': result.returncode == 0 and output_created,
                'return_code': result.returncode,
                'duration': duration,
                'output_created': output_created,
                'stdout_lines': len(result.stdout.split('\n')),
                'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0,
                'error_message': result.stderr if result.returncode != 0 else None
            }
            
            return test_result
            
        except subprocess.TimeoutExpired:
            return {
                'test_id': test_id,
                'params': params,
                'command': ' '.join(cmd),
                'success': False,
                'return_code': -1,
                'duration': time.time() - start_time,
                'output_created': False,
                'error_message': 'Test timed out after 30 minutes'
            }
        except Exception as e:
            return {
                'test_id': test_id,
                'params': params,
                'command': ' '.join(cmd),
                'success': False,
                'return_code': -2,
                'duration': time.time() - start_time,
                'output_created': False,
                'error_message': f'Test execution failed: {str(e)}'
            }
    
    def run_all_tests(self, max_tests: int = None) -> Dict[str, Any]:
        """Run all parameter combinations"""
        if not self.test_tracks:
            print("❌ No test tracks found! Please ensure WAV files exist in test_tracks/, House/, or example_tracks/")
            return {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
        
        print(f"🎵 DJ Mix Generator Comprehensive Test Suite")
        print(f"📁 Test tracks: {len(self.test_tracks)}")
        for i, track in enumerate(self.test_tracks, 1):
            print(f"   [{i}] {track}")
        print()
        
        # Generate all parameter combinations
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[name] for name in param_names]
        all_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(all_combinations)
        if max_tests and total_combinations > max_tests:
            print(f"⚠️  Limiting tests to {max_tests} out of {total_combinations} total combinations")
            all_combinations = all_combinations[:max_tests]
        
        print(f"🧪 Testing {len(all_combinations)} parameter combinations...")
        print("=" * 80)
        
        test_id = 0
        for combination in all_combinations:
            test_id += 1
            params = dict(zip(param_names, combination))
            
            # Skip incompatible combinations
            if self._should_skip_combination(params):
                self.skipped_tests.append({
                    'test_id': test_id,
                    'params': params,
                    'reason': 'Incompatible parameter combination'
                })
                continue
            
            # Run the test
            result = self._run_test(test_id, params)
            self.results.append(result)
            
            # Print progress
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = f"{result['duration']:.1f}s"
            
            print(f"[{test_id:3d}] {status} ({duration:>6}) {result['command']}")
            
            if not result['success']:
                self.failed_tests.append(result)
                if result['error_message']:
                    print(f"     Error: {result['error_message']}")
        
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = len(self.failed_tests)
        skipped_tests = len(self.skipped_tests)
        
        if total_tests > 0:
            avg_duration = sum(r['duration'] for r in self.results) / total_tests
            success_rate = (passed_tests / total_tests) * 100
        else:
            avg_duration = 0
            success_rate = 0
        
        summary = {
            'total': total_tests + skipped_tests,
            'run': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'total_duration': sum(r['duration'] for r in self.results)
        }
        
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"Total combinations: {summary['total']}")
        print(f"Tests run:          {summary['run']}")
        print(f"Passed:             {summary['passed']} ({summary['success_rate']:.1f}%)")
        print(f"Failed:             {summary['failed']}")
        print(f"Skipped:            {summary['skipped']}")
        print(f"Average duration:   {summary['avg_duration']:.1f}s")
        print(f"Total duration:     {summary['total_duration']:.1f}s")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for failure in self.failed_tests[:5]:  # Show first 5 failures
                print(f"   Test {failure['test_id']}: {failure['error_message'] or 'Unknown error'}")
                print(f"   Command: {failure['command']}")
                print()
            
            if len(self.failed_tests) > 5:
                print(f"   ... and {len(self.failed_tests) - 5} more failures")
        
        return summary
    
    def save_results(self, filename: str = 'test_results.json'):
        """Save detailed test results to JSON file"""
        results_data = {
            'summary': self._generate_summary(),
            'results': self.results,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'test_tracks': self.test_tracks,
            'parameters': self.parameters
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Detailed results saved to {filename}")


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DJ Mix Generator Comprehensive Test Suite')
    parser.add_argument('--max-tests', type=int, help='Maximum number of tests to run')
    parser.add_argument('--save-results', default='test_results.json', help='File to save results')
    parser.add_argument('--quick', action='store_true', help='Run a quick subset of tests')
    
    args = parser.parse_args()
    
    # Quick mode - test key combinations only
    if args.quick:
        max_tests = 50
        print("🚀 Running in quick mode (50 tests)")
    else:
        max_tests = args.max_tests
    
    suite = ComprehensiveTestSuite()
    summary = suite.run_all_tests(max_tests=max_tests)
    
    if args.save_results:
        suite.save_results(args.save_results)
    
    # Exit with appropriate code
    if summary['failed'] > 0:
        print(f"\n❌ {summary['failed']} tests failed!")
        sys.exit(1)
    else:
        print(f"\n✅ All {summary['passed']} tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
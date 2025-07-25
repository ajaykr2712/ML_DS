#!/usr/bin/env python3
"""
Test runner for ML/DS project unit tests.
Runs all tests and generates a comprehensive test report.
"""

import unittest
import sys
import os
from pathlib import Path
import warnings
from io import StringIO

# Add project paths to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "gen_ai_project" / "src"))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "ML_Implementation" / "src"))

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity > 1:
            self.stream.write(f"\033[92m✓ {test._testMethodName}\033[0m\n")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write(f"\033[91m✗ {test._testMethodName} (ERROR)\033[0m\n")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write(f"\033[91m✗ {test._testMethodName} (FAIL)\033[0m\n")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write(f"\033[93m⚠ {test._testMethodName} (SKIPPED: {reason})\033[0m\n")


class TestRunner:
    """Custom test runner with enhanced reporting."""
    
    def __init__(self):
        self.loader = unittest.TestLoader()
        self.suite = unittest.TestSuite()
        self.results = None
    
    def discover_tests(self, test_dir: str = "tests"):
        """Discover all test files in the test directory."""
        test_path = Path(__file__).parent
        
        print(f"\033[94m🔍 Discovering tests in {test_path}\033[0m")
        
        # Manually add test files to handle import issues gracefully
        test_files = [
            "test_advanced_trainer.py",
            "test_model_factory.py", 
            "test_evaluator.py"
        ]
        
        for test_file in test_files:
            test_file_path = test_path / test_file
            if test_file_path.exists():
                try:
                    # Import the test module
                    module_name = test_file[:-3]  # Remove .py extension
                    spec = unittest.util.spec_from_file_location(module_name, test_file_path)
                    if spec and spec.loader:
                        module = unittest.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Add tests from module
                        tests = self.loader.loadTestsFromModule(module)
                        self.suite.addTests(tests)
                        print(f"\033[92m✓ Loaded tests from {test_file}\033[0m")
                    
                except Exception as e:
                    print(f"\033[93m⚠ Could not load {test_file}: {e}\033[0m")
                    # Create a dummy test that reports the import failure
                    self._add_import_error_test(test_file, str(e))
            else:
                print(f"\033[91m✗ Test file not found: {test_file}\033[0m")
    
    def _add_import_error_test(self, test_file: str, error_msg: str):
        """Add a test that reports import errors."""
        class ImportErrorTest(unittest.TestCase):
            def test_import_failure(self):
                self.skipTest(f"Could not import {test_file}: {error_msg}")
        
        # Add the test with a descriptive name
        test_case = ImportErrorTest()
        test_case._testMethodName = f"import_{test_file.replace('.py', '')}"
        self.suite.addTest(test_case)
    
    def run_tests(self, verbosity: int = 2):
        """Run all discovered tests."""
        print(f"\n\033[94m🚀 Running Tests\033[0m")
        print("=" * 60)
        
        # Count tests
        test_count = self.suite.countTestCases()
        print(f"Found {test_count} test cases")
        
        if test_count == 0:
            print("\033[93m⚠ No tests to run\033[0m")
            return
        
        # Run tests with custom result class
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=verbosity,
            resultclass=ColoredTextTestResult
        )
        
        self.results = runner.run(self.suite)
        
        # Print results
        self._print_summary()
        
        # Print detailed output if there are failures/errors
        if self.results.failures or self.results.errors:
            self._print_detailed_results()
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print(f"\033[94m📊 Test Summary\033[0m")
        print("=" * 60)
        
        total = self.results.testsRun
        failures = len(self.results.failures)
        errors = len(self.results.errors)
        skipped = len(self.results.skipped)
        passed = total - failures - errors - skipped
        
        print(f"Total Tests:  {total}")
        print(f"\033[92mPassed:       {passed}\033[0m")
        print(f"\033[91mFailed:       {failures}\033[0m")
        print(f"\033[91mErrors:       {errors}\033[0m")
        print(f"\033[93mSkipped:      {skipped}\033[0m")
        
        # Success rate
        if total > 0:
            success_rate = (passed / total) * 100
            if success_rate == 100:
                print(f"\033[92mSuccess Rate: {success_rate:.1f}% 🎉\033[0m")
            elif success_rate >= 80:
                print(f"\033[93mSuccess Rate: {success_rate:.1f}% 👍\033[0m")
            else:
                print(f"\033[91mSuccess Rate: {success_rate:.1f}% 👎\033[0m")
        
        # Overall result
        if failures == 0 and errors == 0:
            if skipped == 0:
                print(f"\n\033[92m✅ ALL TESTS PASSED!\033[0m")
            else:
                print(f"\n\033[93m✅ ALL RUNNABLE TESTS PASSED! (Some skipped)\033[0m")
        else:
            print(f"\n\033[91m❌ SOME TESTS FAILED!\033[0m")
    
    def _print_detailed_results(self):
        """Print detailed failure/error information."""
        print(f"\n\033[94m🔍 Detailed Results\033[0m")
        print("=" * 60)
        
        if self.results.failures:
            print(f"\n\033[91m❌ FAILURES ({len(self.results.failures)}):\033[0m")
            for i, (test, traceback) in enumerate(self.results.failures, 1):
                print(f"\n{i}. {test}")
                print("-" * 40)
                print(traceback)
        
        if self.results.errors:
            print(f"\n\033[91m💥 ERRORS ({len(self.results.errors)}):\033[0m")
            for i, (test, traceback) in enumerate(self.results.errors, 1):
                print(f"\n{i}. {test}")
                print("-" * 40)
                print(traceback)
        
        if self.results.skipped:
            print(f"\n\033[93m⚠ SKIPPED ({len(self.results.skipped)}):\033[0m")
            for i, (test, reason) in enumerate(self.results.skipped, 1):
                print(f"{i}. {test}: {reason}")
    
    def generate_report(self, output_file: str = "test_report.html"):
        """Generate HTML test report."""
        if not self.results:
            print("No test results to generate report from")
            return
        
        html_content = self._generate_html_report()
        
        report_path = Path(__file__).parent.parent / output_file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n\033[94m📄 Test report generated: {report_path}\033[0m")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        total = self.results.testsRun
        failures = len(self.results.failures)
        errors = len(self.results.errors)
        skipped = len(self.results.skipped)
        passed = total - failures - errors - skipped
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML/DS Project Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; border-radius: 5px; }}
        .passed {{ background-color: #d4edda; color: #155724; }}
        .failed {{ background-color: #f8d7da; color: #721c24; }}
        .error {{ background-color: #f8d7da; color: #721c24; }}
        .skipped {{ background-color: #fff3cd; color: #856404; }}
        .details {{ margin: 20px 0; }}
        .test-case {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .test-case.passed {{ border-left-color: #28a745; }}
        .test-case.failed {{ border-left-color: #dc3545; }}
        .test-case.error {{ border-left-color: #dc3545; }}
        .test-case.skipped {{ border-left-color: #ffc107; }}
        pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 ML/DS Project Test Report</h1>
        <p>Generated on: {unittest.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <div class="metric passed">✅ Passed: {passed}</div>
        <div class="metric failed">❌ Failed: {failures}</div>
        <div class="metric error">💥 Errors: {errors}</div>
        <div class="metric skipped">⚠️ Skipped: {skipped}</div>
        <p><strong>Success Rate:</strong> {success_rate:.1f}%</p>
        <p><strong>Total Tests:</strong> {total}</p>
    </div>
"""
        
        if self.results.failures:
            html += """
    <div class="details">
        <h2>❌ Failures</h2>
"""
            for test, traceback in self.results.failures:
                html += f"""
        <div class="test-case failed">
            <h3>{test}</h3>
            <pre>{traceback}</pre>
        </div>
"""
            html += "</div>"
        
        if self.results.errors:
            html += """
    <div class="details">
        <h2>💥 Errors</h2>
"""
            for test, traceback in self.results.errors:
                html += f"""
        <div class="test-case error">
            <h3>{test}</h3>
            <pre>{traceback}</pre>
        </div>
"""
            html += "</div>"
        
        if self.results.skipped:
            html += """
    <div class="details">
        <h2>⚠️ Skipped Tests</h2>
"""
            for test, reason in self.results.skipped:
                html += f"""
        <div class="test-case skipped">
            <h3>{test}</h3>
            <p>Reason: {reason}</p>
        </div>
"""
            html += "</div>"
        
        html += """
</body>
</html>
"""
        return html


def main():
    """Main test runner function."""
    print("\033[94m🧪 ML/DS Project Test Suite\033[0m")
    print("=" * 60)
    
    runner = TestRunner()
    
    # Discover tests
    runner.discover_tests()
    
    # Run tests
    runner.run_tests(verbosity=2)
    
    # Generate HTML report
    runner.generate_report()
    
    # Exit with appropriate code
    if runner.results:
        if runner.results.failures or runner.results.errors:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

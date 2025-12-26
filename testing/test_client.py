"""
Clean Test Client for Excel MCP Server

Improvements:
- Better error handling
- Output directory for test results  
- Cleaner JSON parsing
- Proper subprocess management
- More detailed test reporting
"""

import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional
import time

class MCPTestClient:
    def __init__(self):
        self.project_root = self._get_project_root()
        self.output_dir = self._setup_output_dir()
        self.server_process: Optional[subprocess.Popen] = None
        
    def _get_project_root(self) -> Path:
        """Get the project root directory"""
        current = Path.cwd()
        if "testing" in current.name:
            return current.parent
        return current
    
    def _setup_output_dir(self) -> Path:
        """Create and return output directory"""
        output_path = self.project_root / "testing" / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def create_test_data(self):
        """Create test Excel file in output directory"""
        import openpyxl
        
        print(f"Creating test data in: {self.output_dir}")
        
        wb = openpyxl.Workbook()
        ws = wb.active
        if ws is not None:
            ws.title = "TestFinancials"
        
        # Sample financial data
        data = [
            ["Item", "2023", "2024", "2025E"],
            ["Revenue", 1000000, 1200000, 1440000],
            ["COGS", 600000, 720000, 864000],
            ["Gross Profit", 400000, 480000, 576000],
            ["SG&A", 150000, 180000, 216000],
            ["EBITDA", 250000, 300000, 360000]
        ]
        
        for row_idx, row_data in enumerate(data, 1):
            for col_idx, value in enumerate(row_data, 1):
                if ws is not None:
                    ws.cell(row=row_idx, column=col_idx, value=value)
        
        test_file_path = self.output_dir / "test_data.xlsx"
        wb.save(test_file_path)
        print(f"Created: {test_file_path}")
        return test_file_path
    
    def start_server(self) -> bool:
        """Start the MCP server"""
        try:
            server_path = self.project_root / "mcp_servers" / "excel_server" / "server.py"
            
            self.server_process = subprocess.Popen(
                [sys.executable, str(server_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_root
            )
            
            # Give server time to start
            time.sleep(0.5)
            
            if self.server_process.poll() is not None:
                if self.server_process.stderr:
                    stderr_output = self.server_process.stderr.read()
                    print(f"Server failed to start: {stderr_output}")
                return False
                
            print("MCP Server started")
            return True
            
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def send_request(self, request: Dict[str, Any], expect_response: bool = True) -> Optional[Dict[str, Any]]:
        """Send a request to the server and get response"""
        if not self.server_process or not self.server_process.stdin or not self.server_process.stdout:
            print("No server process or streams")
            return None
            
        try:
            request_json = json.dumps(request) + "\n"
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()
            
            if not expect_response:
                return None  # Don't wait for response
            
            response_line = self.server_process.stdout.readline()
            if not response_line.strip():
                print("Empty response from server")
                return None
                
            response = json.loads(response_line.strip())
            return response
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response_line if 'response_line' in locals() else 'No response'}")
            return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def initialize_mcp(self) -> bool:
        """Initialize MCP connection"""
        print("\nInitializing MCP connection...")
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize", 
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = self.send_request(init_request)
        if not response or "error" in response:
            print(f"Initialization failed: {response}")
            return False
            
        print(f"Server info: {response['result']['serverInfo']}")
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        
        self.send_request(initialized_notification, expect_response=False)
        print("MCP connection initialized")
        return True
    
    def test_tool_discovery(self) -> bool:
        """Test tool discovery"""
        print("\nTesting tool discovery...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        response = self.send_request(request)
        if not response or "error" in response:
            print(f"Tool discovery failed: {response}")
            return False
            
        tools = response.get("result", {}).get("tools", [])
        tool_names = [tool["name"] for tool in tools]
        print(f"Available tools: {tool_names}")
        
        return len(tools) > 0
    
    def test_read_range(self, test_file_path: Path) -> bool:
        """Test reading Excel range"""
        print("\nTesting read_range...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "read_range",
                "arguments": {
                    "file_path": str(test_file_path),
                    "sheet_name": "TestFinancials",
                    "range": "A1:D6"
                }
            }
        }
        
        response = self.send_request(request)
        if not response or "error" in response:
            print(f"Read range failed: {response}")
            return False
            
        # Parse the tool response
        try:
            content = response["result"]["content"][0]["text"]
            data = json.loads(content)
            
            if data.get("success"):
                print(f"Read {data['metadata']['rows']} rows, {data['metadata']['columns']} columns")
                print(f"  Sample data: {data['data'][0]}")
                
                # Save results to output
                results_file = self.output_dir / "read_results.json"
                with open(results_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Results saved to: {results_file}")
                
                return True
            else:
                print(f"Read failed: {data['error']}")
                return False
                
        except Exception as e:
            print(f"Failed to parse response: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid file"""
        print("\nTesting error handling...")
        
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "read_range",
                "arguments": {
                    "file_path": "nonexistent_file.xlsx",
                    "sheet_name": "Sheet1",
                    "range": "A1:A1"
                }
            }
        }
        
        response = self.send_request(request)
        if not response:
            print("No response for error test")
            return False
            
        # Should get an error response
        try:
            content = response["result"]["content"][0]["text"]
            data = json.loads(content)
            
            if not data.get("success"):
                print(f"Error handled correctly: {data['error']}")
                return True
            else:
                print("Should have failed but didn't")
                return False
                
        except Exception as e:
            print(f"Failed to parse error response: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.server_process:
            print("\nCleaning up...")
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            
            # Print any server errors
            if self.server_process and self.server_process.stderr:
                stderr_output = self.server_process.stderr.read()
                if stderr_output:
                    print(f"Server stderr: {stderr_output}")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 50)
        print("Excel MCP Server Test Suite")
        print("=" * 50)
        
        try:
            # Setup
            test_file = self.create_test_data()
            
            if not self.start_server():
                return False
                
            if not self.initialize_mcp():
                return False
            
            # Run tests
            tests_passed = 0
            total_tests = 3
            
            if self.test_tool_discovery():
                tests_passed += 1
                
            if self.test_read_range(test_file):
                tests_passed += 1
                
            if self.test_error_handling():
                tests_passed += 1
            
            # Results
            print("\n" + "=" * 50)
            print(f"Tests completed: {tests_passed}/{total_tests} passed")
            print(f"Output directory: {self.output_dir}")
            print("=" * 50)
            
            return tests_passed == total_tests
            
        except Exception as e:
            print(f" Test suite failed: {e}")
            return False
        finally:
            self.cleanup()

async def main():
    """Run the test suite"""
    client = MCPTestClient()
    success = client.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
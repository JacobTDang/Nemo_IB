"""
Debug Test Client - Fix hanging issue
"""

import asyncio
import json
import subprocess
import sys
import os
import time

def test_mcp_server():
    print("Starting MCP Server Test...")
    
    # Get project root
    project_root = os.path.dirname(os.getcwd())
    server_path = os.path.join(project_root, "mcp_servers", "excel_server", "server.py")
    
    # Start server
    server_process = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=project_root
    )
    
    try:
        print("Server started, waiting for initialization...")
        time.sleep(1)  # Give server time to start
        
        # Step 1: Initialize
        print("1. Sending initialize request...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        request_json = json.dumps(init_request) + "\n"
        server_process.stdin.write(request_json)
        server_process.stdin.flush()
        
        # Read response (Windows compatible)
        print("Waiting for init response...")
        try:
            response_line = server_process.stdout.readline()
            if response_line.strip():
                print(f"Init response: {response_line.strip()}")
                response = json.loads(response_line.strip())
                if "result" in response:
                    print(f"Server info: {response['result'].get('serverInfo', {})}")
                else:
                    print(f"Error in init: {response}")
            else:
                print("Empty response to initialize")
                return False
        except Exception as e:
            print(f"Failed to read init response: {e}")
            return False
            
        # Step 2: Send initialized notification (no response expected)
        print("2. Sending initialized notification...")
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        
        request_json = json.dumps(initialized_notification) + "\n"
        server_process.stdin.write(request_json)
        server_process.stdin.flush()
        
        # Don't wait for response to notification
        print("Initialized notification sent")
        
        # Step 3: Test tool discovery
        print("3. Testing tool discovery...")
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        request_json = json.dumps(list_tools_request) + "\n"
        server_process.stdin.write(request_json)
        server_process.stdin.flush()
        
        # Read response 
        print("Waiting for tools response...")
        try:
            response_line = server_process.stdout.readline()
            if response_line.strip():
                print(f"Tools response: {response_line.strip()}")
                response = json.loads(response_line.strip())
                if "result" in response:
                    tools = response["result"].get("tools", [])
                    tool_names = [tool["name"] for tool in tools]
                    print(f"Available tools: {tool_names}")
                else:
                    print(f"Error in tools/list: {response}")
            else:
                print("Empty response to tools/list")
        except Exception as e:
            print(f"Failed to read tools response: {e}")
            
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
        
    finally:
        print("Cleaning up...")
        server_process.terminate()
        server_process.wait()
        
        # Check for errors
        if server_process.stderr:
            stderr_output = server_process.stderr.read()
            if stderr_output:
                print(f"Server stderr: {stderr_output}")

if __name__ == "__main__":
    test_mcp_server()
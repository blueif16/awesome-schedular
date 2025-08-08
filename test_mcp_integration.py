#!/usr/bin/env python3
"""
Test MCP Integration - Testing MCP Server Connection
Tests the MultiServerMCPClient integration with our scheduler and mem0 servers
"""

import asyncio
import logging
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_mcp_servers():
    """Test MCP server connection and tool retrieval"""
    
    print("🧪 Testing MCP Server Integration")
    print("=" * 50)
    
    # Get current directory for absolute paths
    current_dir = Path(__file__).parent.absolute()
    
    try:
        # Setup MCP client
        print("🔧 Setting up MCP client...")
        client = MultiServerMCPClient({
            "scheduler": {
                "command": "python",
                "args": [str(current_dir / "core" / "scheduler_mcp.py")],
                "transport": "stdio",
            },
            "mem0": {
                "command": "python", 
                "args": [str(current_dir / "core" / "mem0_mcp.py")],
                "transport": "stdio",
            }
        })
        
        print("✅ MCP client configured")
        
        # Get tools from servers
        print("📋 Retrieving tools from MCP servers...")
        tools = await client.get_tools()
        
        print(f"✅ Retrieved {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:100]}...")
        
        # Test basic tool structure
        tool_names = [tool.name for tool in tools]
        expected_tools = ["schedule_with_pattern", "extract_and_store_text_insights"]
        
        for expected_tool in expected_tools:
            if expected_tool in tool_names:
                print(f"✅ Found expected tool: {expected_tool}")
            else:
                print(f"❌ Missing expected tool: {expected_tool}")
        
        print(f"\n🏁 MCP Integration Test Results:")
        print(f"  - Servers connected: ✅")
        print(f"  - Tools retrieved: {len(tools)}")
        print(f"  - Expected tools found: {len([t for t in expected_tools if t in tool_names])}/{len(expected_tools)}")
        
        return len(tools) > 0
        
    except Exception as e:
        print(f"❌ MCP integration test failed: {e}")
        logger.error(f"MCP test error: {e}", exc_info=True)
        return False

async def test_individual_servers():
    """Test each MCP server individually"""
    
    print("\n🧪 Testing Individual MCP Servers")
    print("=" * 50)
    
    current_dir = Path(__file__).parent.absolute()
    
    servers = {
        "scheduler": str(current_dir / "core" / "scheduler_mcp.py"),
        "mem0": str(current_dir / "core" / "mem0_mcp.py")
    }
    
    results = {}
    
    for server_name, server_path in servers.items():
        try:
            print(f"\n📋 Testing {server_name} server...")
            
            client = MultiServerMCPClient({
                server_name: {
                    "command": "python",
                    "args": [server_path],
                    "transport": "stdio",
                }
            })
            
            tools = await client.get_tools()
            print(f"✅ {server_name} server: {len(tools)} tools retrieved")
            
            for tool in tools:
                print(f"  - {tool.name}")
            
            results[server_name] = len(tools)
            
        except Exception as e:
            print(f"❌ {server_name} server failed: {e}")
            results[server_name] = 0
    
    return results

async def main():
    """Run all MCP integration tests"""
    
    print("🚀 Starting MCP Integration Tests\n")
    
    # Test 1: Combined servers
    success = await test_mcp_servers()
    
    # Test 2: Individual servers  
    individual_results = await test_individual_servers()
    
    print(f"\n🏁 Final Test Results:")
    print(f"  - Combined server test: {'✅ PASSED' if success else '❌ FAILED'}")
    
    for server, tool_count in individual_results.items():
        print(f"  - {server} server: {'✅ PASSED' if tool_count > 0 else '❌ FAILED'} ({tool_count} tools)")
    
    if success and all(count > 0 for count in individual_results.values()):
        print("\n🎉 All MCP integration tests PASSED!")
        print("Ready to run: python core/langgraph_scheduler_mcp.py")
    else:
        print("\n⚠️ Some tests failed. Check server configuration.")

if __name__ == "__main__":
    asyncio.run(main()) 
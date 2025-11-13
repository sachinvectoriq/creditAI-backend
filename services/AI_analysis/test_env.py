#!/usr/bin/env python3
"""
Test different ways to initialize AzureAIAgentClient.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from azure.identity import DefaultAzureCredential
from agent_framework.azure import AzureAIAgentClient
import inspect

print("=" * 60)
print("Agent Client Initialization Test")
print("=" * 60)
print()

# Get project endpoint
project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
print(f"Project Endpoint: {project_endpoint}")
print()

# Get credential
credential = DefaultAzureCredential()
print("✅ Credential obtained")
print()

# Check the AzureAIAgentClient signature
print("Inspecting AzureAIAgentClient.__init__ signature:")
sig = inspect.signature(AzureAIAgentClient.__init__)
print(f"Parameters: {sig}")
print()

# Try different initialization methods
print("=" * 60)
print("Testing Initialization Methods")
print("=" * 60)
print()

# Method 1: With credential parameter
print("Method 1: Using credential parameter...")
try:
    client = AzureAIAgentClient(
        endpoint=project_endpoint,
        credential=credential,
        api_version="2024-02-15-preview"
    )
    print("✅ SUCCESS with credential parameter")
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Method 2: With agents_client parameter (what the error suggests)
print("Method 2: Creating agents_client first...")
try:
    # Maybe we need to create an agents_client using azure-ai-projects
    from azure.ai.projects import AIProjectClient
    
    agents_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential
    )
    
    print(f"✅ Created AIProjectClient: {agents_client}")
    
    # Now try with agents_client
    client = AzureAIAgentClient(
        agents_client=agents_client
    )
    print("✅ SUCCESS with agents_client parameter")
    print()
    print("=" * 60)
    print("✅ SOLUTION FOUND!")
    print("=" * 60)
    print()
    print("Use this pattern:")
    print("  from azure.ai.projects import AIProjectClient")
    print("  agents_client = AIProjectClient(endpoint=..., credential=...)")
    print("  agent_client = AzureAIAgentClient(agents_client=agents_client)")
    
except ImportError as ie:
    print(f"❌ azure.ai.projects not installed: {ie}")
    print()
    print("Install it with: pip install azure-ai-projects")
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Method 3: Check if there's an async version
print("Method 3: Checking for async client...")
try:
    from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
    
    print("✅ Async AIProjectClient found")
    print()
    print("For async code, use:")
    print("  from azure.ai.projects.aio import AIProjectClient")
    print("  agents_client = AIProjectClient(endpoint=..., credential=...)")
    
except ImportError:
    print("ℹ️  No async version found")

print()
print("=" * 60)

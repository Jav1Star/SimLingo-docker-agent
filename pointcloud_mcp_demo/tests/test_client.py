"""
Run from the repository root:
    uv run examples/snippets/clients/streamable_basic.py
"""

import asyncio

from pydantic import AnyUrl

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.metadata_utils import get_display_name

async def display_tools(session: ClientSession):
    """Display available tools with human-readable names"""
    tools_response = await session.list_tools()

    for tool in tools_response.tools:
        # get_display_name() returns the title if available, otherwise the name
        display_name = get_display_name(tool)
        print(f"Tool: {display_name}")
        if tool.description:
            print(f"   {tool.description}")

async def display_resources(session: ClientSession):
    """Display available resources with human-readable names"""
    resources_response = await session.list_resources()

    for resource in resources_response.resources:
        display_name = get_display_name(resource)
        print(f"Resource: {display_name} ({resource.uri})")

    templates_response = await session.list_resource_templates()
    for template in templates_response.resourceTemplates:
        display_name = get_display_name(template)
        print(f"Resource Template: {display_name}")



async def main():
    """Run the display utilities example."""
    # Connect to a streamable HTTP server
    async with streamable_http_client("http://10.112.136.44:8123/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            print("=== Available Tools ===")
            await display_tools(session)

            print("\n=== Available Resources ===")
            await display_resources(session)

            # Read a resource 
            resource_content = await session.read_resource(AnyUrl("perception://2021_09_03_09_32_17/302/0"))
            content_block = resource_content.contents[0]
            if isinstance(content_block, types.TextResourceContents):
                print(f"Resource content: {content_block.text}")
                print(f"length of text content: {len(content_block.text)} characters")

    print("Client session closed.")



if __name__ == "__main__":
    asyncio.run(main())
# Build the image
```powershell
docker build -t mcp-demo-server . 
```

# Run the container
```powershell
docker run -it --rm mcp-demo-server
```

# Connect the Docker MCP server to Claude Desktop
- edit the claude_desktop_config.json to add the Docker MCP server to Claude Desktop
- edit the ~/.gemini/settings.json to add the Docker MCP server to Gemini CLI
```json
{
  ...file contains other config objects
  "mcpServers": {
    ...other mcp servers
    "demo-docker": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "mcp-demo-server"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```
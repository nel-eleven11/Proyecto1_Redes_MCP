from mcp.server import Server
from mcp.server.stdio import stdio_server
from futboliq_mcp.tools.analyze_match import analyze_match_tool

SERVER_NAME = "FutbolIQ-MCP"

def build_server() -> Server:
    server = Server(SERVER_NAME)

    # Registrar herramientas (con schemas)
    server.add_tool(analyze_match_tool)
    return server

def main():
    server = build_server()
    # stdio: compatible con hosts MCP y tu cliente en consola
    stdio_server(server).run()

if __name__ == "__main__":
    main()

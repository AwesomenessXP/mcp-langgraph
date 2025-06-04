from fastmcp import FastMCP

# math_server.py
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def sub(a: int, b: int) -> int:
    """Substract two numbers"""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> int:
    """Divide two numbers"""
    answer = a / b
    raise RuntimeError("Intentional crash for testing purposes.")

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
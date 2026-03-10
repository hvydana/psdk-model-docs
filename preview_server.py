#!/usr/bin/env python3
"""
Simple HTTP server to preview markdown documentation
"""
import http.server
import socketserver
import os
import markdown
from pathlib import Path

PORT = 8000

class MarkdownHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve markdown files as HTML
        file_path = self.path[1:] if self.path != '/' else 'README.md'
        file_path = file_path.replace('%20', ' ')

        # If path is a directory, try to serve README.md from that directory
        if os.path.isdir(file_path):
            readme_path = os.path.join(file_path, 'README.md')
            if os.path.isfile(readme_path):
                file_path = readme_path

        if self.path.endswith('.md') or self.path == '/' or os.path.isfile(file_path) and file_path.endswith('.md'):
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])

                    # Wrap in HTML template
                    full_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1">
                        <title>Physical AI SDK Documentation</title>
                        <style>
                            body {{
                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                                line-height: 1.6;
                                max-width: 900px;
                                margin: 0 auto;
                                padding: 20px;
                                background: #ffffff;
                                color: #24292e;
                            }}
                            h1 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
                            h2 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 24px; }}
                            code {{
                                background: #f6f8fa;
                                padding: 2px 6px;
                                border-radius: 3px;
                                font-family: 'Monaco', 'Courier New', monospace;
                                font-size: 85%;
                            }}
                            pre {{
                                background: #f6f8fa;
                                padding: 16px;
                                border-radius: 6px;
                                overflow: auto;
                            }}
                            pre code {{
                                background: none;
                                padding: 0;
                            }}
                            a {{ color: #0366d6; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                            table {{
                                border-collapse: collapse;
                                width: 100%;
                                margin: 16px 0;
                            }}
                            th, td {{
                                border: 1px solid #dfe2e5;
                                padding: 6px 13px;
                            }}
                            th {{ background: #f6f8fa; font-weight: 600; }}
                            hr {{ border: 0; border-top: 1px solid #eaecef; margin: 24px 0; }}
                            .nav {{
                                background: #f6f8fa;
                                padding: 10px;
                                border-radius: 6px;
                                margin-bottom: 20px;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="nav">
                            <a href="/">🏠 Home</a> |
                            <a href="/MODELS_INDEX.md">📑 Models Index</a> |
                            <a href="/CONTRIBUTING.md">📝 Contributing</a>
                        </div>
                        {html_content}
                    </body>
                    </html>
                    """

                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(full_html.encode('utf-8'))
            else:
                self.send_error(404, f"File not found: {file_path}")
        else:
            # Serve other files normally
            super().do_GET()

if __name__ == '__main__':
    os.chdir('/home/amd/Work/model_documentation')
    with socketserver.TCPServer(("", PORT), MarkdownHTTPRequestHandler) as httpd:
        print(f"🌐 Documentation server running at http://localhost:{PORT}")
        print(f"📖 Open in your browser to view the documentation")
        print(f"Press Ctrl+C to stop")
        httpd.serve_forever()

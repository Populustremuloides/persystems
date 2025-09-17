#!/usr/bin/env python3
"""Execute project notebooks and build static HTML for GitHub Pages."""
from __future__ import annotations

import argparse
import html
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook-dir",
        default="notebooks",
        help="Directory that contains the source .ipynb notebooks",
    )
    parser.add_argument(
        "--output-dir",
        default="site",
        help="Directory where the rendered site should be written",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Execution timeout (seconds) per notebook",
    )
    return parser.parse_args()


def friendly_title(name: str) -> str:
    # Keep numeric prefixes while making the remainder easier to read.
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        prefix, rest = parts
        return f"{prefix} {rest.replace('_', ' ').strip().title()}"
    return name.replace("_", " ").strip().title()


def main() -> int:
    args = parse_args()

    notebook_dir = Path(args.notebook_dir)
    if not notebook_dir.exists():
        print(f"Notebook directory '{notebook_dir}' does not exist", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    html_dir = output_dir / "notebooks"
    html_dir.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in '{notebook_dir}'", file=sys.stderr)
        return 1

    links = []
    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root)] + ([pythonpath] if pythonpath else [])
    )

    for notebook in notebooks:
        display_name = friendly_title(notebook.stem)
        output_name = notebook.with_suffix(".html").name

        cmd = [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "html",
            "--execute",
            f"--ExecutePreprocessor.timeout={args.timeout}",
            "--output-dir",
            str(html_dir),
            str(notebook),
        ]
        print(f"Executing {notebook} -> {output_name}")
        subprocess.run(cmd, check=True, env=env)

        links.append(
            f'<li><a href="notebooks/{html.escape(output_name)}">'
            f"{html.escape(display_name)}</a></li>"
        )

    links_markup = "\n      ".join(links)

    index_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>persystems notebooks</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; line-height: 1.6; }}
    h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
    p {{ max-width: 50rem; }}
    ul {{ padding-left: 1.25rem; }}
    li {{ margin-bottom: 0.25rem; }}
    a {{ color: #0b5394; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <main>
    <h1>persystems notebooks</h1>
    <p>These notebooks are executed on every push to the default branch so the published pages always include fresh outputs.</p>
    <ul>
      {links_markup}
    </ul>
  </main>
</body>
</html>
"""

    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

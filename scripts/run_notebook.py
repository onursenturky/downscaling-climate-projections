"""Execute this project's plain-Python notebook in a fresh process, offline.

The notebook deliberately uses no IPython magics. This small runner records its
explicit print/display/plt.show output without extra Jupyter server dependencies.
Use --write to store successful execution outputs in the notebook.
"""

import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "istanbul-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import IPython.display

    path = root / "Statistical_Downscaling_of_Climate_Projections_with_Deep_Learning.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__"}
    outputs = []

    def display(*objects, **kwargs):
        for obj in objects:
            data = {"text/plain": repr(obj)}
            if hasattr(obj, "_repr_html_"):
                html = obj._repr_html_()
                if html:
                    data["text/html"] = html
            outputs.append({"output_type": "display_data", "data": data, "metadata": {}})

    def show(*args, **kwargs):
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {
                "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "text/plain": "<Matplotlib figure>",
            }})
        plt.close("all")

    IPython.display.display = display
    plt.show = show
    count = 0
    started = time.perf_counter()
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        outputs = []
        stdout, stderr = io.StringIO(), io.StringIO()
        source = "".join(cell["source"])
        print(f"Cell {i + 1}: {source.splitlines()[0][:90] if source else '(empty)'}", flush=True)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(source, f"{path.name}:cell-{i + 1}", "exec"), namespace)
        streams = []
        for name, content in (("stdout", stdout.getvalue()), ("stderr", stderr.getvalue())):
            if content:
                streams.append({"output_type": "stream", "name": name, "text": content.splitlines(keepends=True)})
                print(content[-2000:], end="", flush=True)
        cell["outputs"] = streams + outputs
        cell["execution_count"] = count
    notebook["metadata"]["language_info"]["version"] = sys.version.split()[0]
    if args.write:
        temporary = path.with_suffix(".ipynb.tmp")
        temporary.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        temporary.replace(path)
    print(f"PASS: {count} code cells completed in {time.perf_counter() - started:.1f}s.")


if __name__ == "__main__":
    main()

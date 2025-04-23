"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contains examples.rst file

Copy the Example folder to Documentation folder
"""

import os

examples_dir = "Examples"
output_rst = "examples.rst"

def format_title(name):
    return name.replace("_", " ").replace(".ipynb", "").title()

examples_structure = []

for root, _, files in os.walk(examples_dir):
    rel_path = os.path.relpath(root, examples_dir)
    section_name = rel_path.replace("_", " ").replace("/", " ").title()
    entries = []

    for file in sorted(files):
        if file.endswith(".ipynb"):
            title = format_title(file)
            rel_file_path = os.path.join("Examples", rel_path, file).replace("\\", "/")
            entries.append((title, rel_file_path))

    if entries:
        examples_structure.append((section_name, entries))

with open(output_rst, "w") as f:
    f.write("PyOR Simulation Examples\n")
    f.write("=" * len("PyOR Simulation Examples") + "\n\n")

    for section, items in examples_structure:
        f.write(f"{section}\n")
        f.write(f"{'-' * len(section)}\n\n")
        for title, path in items:
            f.write(f"- `{title} <{path}>`_\n")
        f.write("\n")










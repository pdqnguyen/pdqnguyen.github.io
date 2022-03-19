import os
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser


HEADER_TEMPLATE = """---
layout: post
date: {date}
title: "{title}"
categories: jekyll update
permalink: {permalink}
summary: |
  {summary}
---

"""


parser = ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_dir")
parser.add_argument("--date", default="")
parser.add_argument("--title", default="")
parser.add_argument("--permalink", default="")
parser.add_argument("--summary", default="")
parser.add_argument("--image-dir", help="move images to here; relative to webpage repository root")
parser.add_argument("--omit-code", default=False, action="store_true")
parser.add_argument("--omit-output", default=False, action="store_true")
parser.add_argument("--omit-tables", default=False, action="store_true")
parser.add_argument("--hide-code", default=False, action="store_true")
parser.add_argument("--hide-output", default=False, action="store_true")
parser.add_argument("--hide-tables", default=False, action="store_true")
args = parser.parse_args()
input_file = args.input_file
input_name = os.path.basename(input_file).replace('.ipynb', '')
output_dir = os.path.abspath(args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, 'run.txt'), 'w') as f:
    f.write(" ".join(sys.argv))

# Use nbconvert to convert notebook to markdown
cmd = ['jupyter', 'nbconvert', '--to', 'markdown', input_file, '--output-dir', output_dir]
subprocess.run(cmd)

# Get locations of outputs
output_subdir = os.path.join(output_dir, input_name + '_files')
output_file = os.path.join(output_dir, input_name + '.md')

# Move image files to desired assets directory
image_dir = args.image_dir
if image_dir is not None:
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    image_files = os.listdir(output_subdir)
    for im_file in image_files:
        try:
            shutil.move(os.path.join(output_subdir, im_file), image_dir)
        except shutil.Error:
            continue

# Load raw output for editing
with open(output_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Collect "chunks" of markdown text to format each chunk type uniquely
chunks = []
while len(text) > 0:
    spans = {}
    groups = {}
    blocks = {}
    blocks['code'] = re.search(r"(```python\n(?:.|\n)*?```)", text)
    blocks['output'] = re.search(r"\n(    [^\n][^`#]+)", text)
    blocks['table'] = re.search(r"<div>(.|\n)*?</div>", text)
    blocks['image'] = re.search(r"!\[png\]\(.*?\)", text)
    for key, block in blocks.items():
        if block is not None:
            spans[key] = block.span()
            groups[key] = block.group()
    if len(spans) == 0:
        break
    first = min(spans.keys(), key=lambda x: spans[x][0])
    start, end = spans[first]
    group = groups[first]
    # Format matching text
    if first == 'code':
        if args.hide_code:
            # Switch to jekyll formatting when putting python code inside a collapsible
            group = re.sub(r"```python((.|\n)*)```", r"{% highlight python %}\1{% endhighlight %}", group)
            # group = group.replace("```python", "{% highlight python %}").replace("```", "{% endhighlight %}")
            group = "<details>\n<summary>Show code</summary>\n" + group.strip('\n') + "\n</details><br>"
        elif args.omit_code:
            group = ""
    elif first == 'output':
        # Un-indent and put it in a code-output class
        group = re.sub("    (.*)", r"\1", group)
        group = '<pre class="code-output">\n' + group.strip('\n') + '\n</pre>'
        if args.hide_output:
            group = "<details>\n<summary>Show output</summary>\n" + group.strip('\n') + "\n</details><br>"
        elif args.omit_output:
            group = ""
    elif first == 'table':
        if args.hide_tables:
            group = "<details>\n<summary>Show table</summary>\n" + group.strip('\n') + "\n</details><br>"
        elif args.omit_tables:
            group = ""
    elif first == 'image':
        if image_dir is not None:
            # Edit hyperlink to point to correct image output directory
            im_link = re.search(r"!\[png\]\((.*)\)", group).group(1)
            new_im_link = os.path.join("/", image_dir, os.path.basename(im_link))
            group = re.sub(im_link, new_im_link, group)
    if start > 0:
        # Add any uncategorized text before first matching group
        chunks.append(text[:start])
    # Add the matched group
    chunks.append(group + '\n')
    # Remove chunk from text so next iteration starts right after the end of previous chunk
    if end < len(text):
        text = text[end:]
    else:
        break

new_text = HEADER_TEMPLATE.format(date=args.date, title=args.title, permalink=args.permalink, summary=args.summary)
for chunk in chunks:
    new_text += chunk

# Replace every single $ with a double $$ for inline math, because jekyll requires double $$
new_text = re.sub(r"([^$])\$([^$]+)\$([^$])", r"\1$$\2$$\3", new_text)

# header = "" #HEADER_TEMPLATE.format(date=args.date, title=args.title, permalink=args.permalink, summary=args.summary)
# new_text = header + text
with open(output_file.replace('.md', '_formatted.md'), 'w', encoding='utf-8') as f:
    f.write(new_text)

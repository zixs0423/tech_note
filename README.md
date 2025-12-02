# Zixs' Tech Note

1. The repo need to be set as public to be displayed by github pages.
2. This repo uses the docs configuration, which is more suitable to be categorized:
   ```
   tech_note/
   │
   ├── README.md  
   ├── docs/  
   │   ├── index.md           # first page
   │   ├── _config.yml        # Jekyll configuration
   │   │
   │   ├── notes/             # notes Markdown
   │   │   ├── xxx.md
   │   │   ├── xxx.md
   │   │   └── xxx.md
   │   │
   │   ├── code/  
   │   │   ├── xxx.py
   │   │   └── xxx.ipynb
   │   │
   │   └── pdf/              # paper PDF
   │       ├── paper1.pdf
   │       └── paper2.pdf
   └── .gitignore             # optional
   ```

   and the docs file is set as the root in settings-pages-Build and deployment
   the repo can also use _posts configuration, which is automatically orgnized in added time and is more suitable for personal blog.
3. Github pages automatically use jekyll to build, which mostly takes 1-3 mins, and the process can be monitored in actions.
   all markdown files shoud have the front matter in order to be built by jekyll:
   ```
   ---
   layout: default
   ---
   ```
   the _layouts/default.html includes the mathjax to reder the latex equations in markdown files.
   the mathjax is v3 and the explicit config is set in the default.html to enable the features such as line breaking and inline math, which are enabled by default in the mathjax v2 in vscode.
4. Write markdown and python in codespace
   copy the preview version of markdown to citadel and the equations will be automatically transformed to images.
5. Move all acadamic papers and code into this repo, and paste their links in markdown file notes.

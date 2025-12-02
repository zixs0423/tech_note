# Zixs' Tech Note

1. The repo need to be set as public
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
   the repo can also use _posts configuration, which is automatically orgnized in time added and is more suitable for personal blog
3. Github pages automatically use jekyll to build, which mostly takes 1-3 mins, and can be found in actions
   all markdown shoud have the front matter in order to be built by jekyll:
   ```
   ---
   layout: default
   ---
   ```
4. Write markdown and python in codespace
   copy the preview version of markdown to citadel can automatically transform the equations
6. Move all acadamic papers and code into this repo, and paste their links in notes

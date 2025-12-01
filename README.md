# Zixs' Tech Note

1. Set the repo as public
2. use the docs configuration:
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

   and set as the root in settings-pages-Build and deployment, can also use _posts configuration, which is more suitable for personal blog
3. github pages automatically use jekyll to build
   all markdown shoud have the front matter:
   ```
   ---
   layout: default
   ---
   ```
4. write markdown and python in local environment, then copy and paste in this repo
5. move all acadamic papers and code into this repo, and paste their links in notes

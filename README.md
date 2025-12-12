# Zixs' Tech Note

# Preparation
1. The repo is set as public to be displayed by github pages.
2. This repo uses the docs configuration, which is more suitable to be categorized. and the docs file is set as the root in github-settings-pages-Build and deployment.
   
   <details markdown="1"><summary>The configuration</summary>

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
   │   └── pdf/              # paper PDF
   │       ├── paper1.pdf
   │       └── paper2.pdf
   └── .gitignore             # optional
   ```

   </details>

   The repo can also use _posts configuration, which is automatically orgnized in added time, but is more suitable for personal blog.
3. The _layouts/default.html includes the mathjax to reder the latex equations in markdown files. The mathjax is v3 and the explicit config is set in the default.html to enable the features such as line breaking and inline math, which are enabled by default in the mathjax v2 in vscode.
   
   
# Procedure
   
1. Write markdown and python in codespace instead of writing them on local environment.
2. Write the note in four sections:
   
   <details markdown="1"><summary>Sections</summary>

   #### Abstract

   <br>

   #### Paper

   <br>

   #### Tutorials

   <br>

   #### Code

   <br>

   ---
   
   </details>

3. Write equations in Latex and code in the markdown file directly. Use folding sections to optimize the display. 
4. Paste the links of academic papers and tutorial blogs in the markdown file directly.
5. Remmenber that all markdown files shoud have the front matter in order to be built by jekyll:

   <details markdown="1"><summary>The front matter</summary>

   ```
   ---
   layout: default
   ---
   ```

   </details>
6. Commit and sync the changes in codespace and the Github pages will automatically use jekyll to build the website, which mostly takes 1-3 min. The process can be monitored in actions.
7. Copy the markdown file (not the preview) to citadel directly, the equations and code blocks will be automatically transformed to correct form.

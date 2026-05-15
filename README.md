# Zixs' Tech Note

# Preparation
1. The repo is set as public to be displayed by github pages.
2. This repo uses the docs configuration displayed as follow. and the docs file is set as the root in github-settings-pages-Build and deployment.
   
   <details markdown="1"><summary>The configuration</summary>

   ```
   tech_note/
   │
   ├── README.md  
   ├── docs/  
   │   ├── index.md           # first page
   │   ├── _config.yml        # Jekyll configuration
   │   ├── _layouts           # html layout configuration
   │   │   └── default.html
   │   ├── code/              # code linked in notes
   │   │   ├── xxx.py
   │   │   ├── xxx.py
   │   │   └── xxx.py
   │   ├── images/            # images linked in notes
   │   │   ├── xxx.png
   │   │   ├── xxx.png
   │   │   └── xxx.png
   │   └── notes/             # notes Markdown files
   │       ├── xxx.md
   │       ├── xxx.md
   │       └── xxx.md
   ├── README.md  
   └── .gitignore             
   ```

   </details>

3. The _layouts/default.html includes the mathjax to reder the latex equations in markdown files. The mathjax is v3 and the explicit config is set in the default.html to enable the features such as line breaking and inline math, which are enabled by default in the mathjax v2 in vscode.
   
   
# Procedure
   
1. Write markdown and code in codespace instead of writing them on local environment. 
   
   [Markdown Cheatsheet](https://www.markdownguide.org/cheat-sheet/)

2. Remember that all markdown files shoud have the front matter in order to be built by jekyll:

   <details markdown="1"><summary>The front matter</summary>

   ```
   ---
   layout: default
   ---
   ```

   </details>
 
3. Generate a catalog or table of contents (TOC) by using the command palette (Cmd/Ctrl + Shift + P) and selecting Markdown: Create Table of Contents.
4. Write under different heading levels. Every level can be and should be filled with asterisk expressions. Don't need extra expression and prefix for code, images, tutorials and papers, also don't need asterisk beforehead.
5. Write equations in Latex in the markdown file directly. 
   
   [Latex Cheatsheet](https://quickref.me/latex.html)

6. Do note write code in markdown notes directly. Write them in code folder, which can be run in codespace. Then paste their links in the notes. 
7. Do not upload any pdf files to this github repo. Paste the links of books, papers and tutorial webs in the source section of the notes directly. Download the calssic textbooks and academic papers to work computer and personal computer. Upload them to the Baidu Disk on PC and check them on WC at work. For those saved in Baidu Disk, paste both source links and Baidu Disk links.
8. Copy the images from website and directly paste them in 'images' folder, then paste their links in the note.
   
   <details markdown="1"><summary>link examples</summary>

   ```markdown
   [CF](../code/cf.py)

   [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)

   ![iTransformer](../images/iTransformer.png)
   ```

   </details>

9. Use folding sections to optimize the display. Remember to add 'markdown="1"' in details, which means 'Enable Markdown parsing inside this HTML block'.
10. Commit and sync the changes in codespace and the Github pages will automatically use jekyll to build the website, which mostly takes 1-3 min. The process can be monitored in actions.
11. Copy the markdown file (not the preview) to citadel directly, the equations and code blocks will be automatically transformed to correct form.
12. The leetcode extension is hard be logged in inside the codespace (Using inspect to copy cookies everytime). So solve problems in local vscdoe and copy/upload the code to codespace. Most of the code can not be run individually, they need to request the leetcode website to get tested.
13. The vscode is logged in with my github account and will automatically sync between the local environment and github codespace.

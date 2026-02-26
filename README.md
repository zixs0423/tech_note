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
   └── .gitignore             # optional
   ```

   </details>

3. The _layouts/default.html includes the mathjax to reder the latex equations in markdown files. The mathjax is v3 and the explicit config is set in the default.html to enable the features such as line breaking and inline math, which are enabled by default in the mathjax v2 in vscode.
   
   
# Procedure
   
1. Write markdown and python in codespace instead of writing them on local environment. Do note write python in markdown notes directly, even though they can be written in folding sections. Write them in code folder with python format, which can be run in codespace. Then paste their links in the notes. 
   
   <details markdown="1"><summary>link examples</summary>

   ```markdown
   [CF](../code/cf.py)

   [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)

   ![iTransformer](../images/iTransformer.png)
   ```

   </details>

   [Markdown Cheatsheet](https://www.markdownguide.org/cheat-sheet/)

2. Write the note in Three sections:
   
   <details markdown="1"><summary>Sections</summary>

   #### Concepts

   <br>

   #### Source

   <br>

   #### Code

   <br>

   ---
   
   </details>

3. Write equations in Latex and code in the markdown file directly. Use folding sections to optimize the display. Remember to add 'markdown="1"' in details, which means 'Enable Markdown parsing inside this HTML block'.
4. Do not upload any pdf files to this github repo. Download the calssic textbooks and academic papers to work computer and personal computer. Upload them to the Baidu Disk on PC and check them on WC at work. Paste the links of books, papers and tutorial webs in the source section of the notes directly. for those saved in Baidu Disk, paste both source links and Baidu Disk links.
5. Remember that all markdown files shoud have the front matter in order to be built by jekyll:

   <details markdown="1"><summary>The front matter</summary>

   ```
   ---
   layout: default
   ---
   ```

   </details>
6. Generate a catalog or table of contents (TOC) by using the command palette (Cmd/Ctrl + Shift + P) and selecting Markdown: Create Table of Contents.
7. Commit and sync the changes in codespace and the Github pages will automatically use jekyll to build the website, which mostly takes 1-3 min. The process can be monitored in actions.
8. Copy the markdown file (not the preview) to citadel directly, the equations and code blocks will be automatically transformed to correct form.
9. The leetcode extension can not be logged in inside the codespace. So solve problems in local vscode and then upload the answers and notes. 
10. The vscode is logged in with my github account and will automatically sync between the local environment and github codespace.

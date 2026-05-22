---
layout: default
---

- [Operating System](#operating-system)
  - [Linux](#linux)
    - [Shell](#shell)
      - [Command](#command)
      - [Shell Scripting](#shell-scripting)
      - [SSH (Secure Shell)](#ssh-secure-shell)
    - [Kernel](#kernel)


# Operating System

## Linux

* Architecture: Hardware, Kernel, Shell, Utilities.
* Popular Distributions: Ubuntu, Debian.

![Linux](../images/Linux.png)

[introduction-to-linux-operating-system](https://www.geeksforgeeks.org/linux-unix/introduction-to-linux-operating-system/)

<br>

---

### Shell

* In Linux systems, users communicate with the operating system through a shell, which interprets and executes commands entered in a terminal. The shell acts as an intermediary between the user and the kernel, ensuring that instructions are processed correctly.
* Terminal: Interface used to access the shell
* Shell: Command interpreter that processes user input.
* Common Types of Shells: 
  * Bash (Bourne Again Shell): The most popular and standard shell for most Linux distributions.
  * Zsh (Z Shell): The default shell on modern macOS. It is highly customizable and offers advanced auto-completion features.
  * PowerShell: Microsoft’s advanced shell designed for Windows administration (which works quite differently from Linux shells).
  
[introduction-linux-shell-shell-scripting](https://www.geeksforgeeks.org/linux-unix/introduction-linux-shell-shell-scripting/)

<br>

---

#### Command

* ls: list files.
* du: analyze and report on disk usage within directories and files.
  * -h or --human-readable: Displays sizes in human-readable format, using units such as KB, MB, GB, etc. 
  * -s or --summarize: rovides a summary of the disk usage for the specified directory or file, without displaying individual usage details for subdirectories.
  
  [du-command-linux-examples](https://www.geeksforgeeks.org/linux-unix/du-command-linux-examples/)
  
* pwd: display the path of the working directory.
* cd: change directory.
* mv: move or rename files.
* rm: remove or delete the files.
* mkdir: create new directories or folders.
* cp: copy files or directories.
* grep: search text patterns.
* sed: stream editor, processes text line by line, applying the editing commands you specify. It can also be used for searching, inserting, deleting and modifying text efficiently.
  * sed 's/unix/linux/' geekfile.txt: replaces the word "unix" with "linux" in the file.

  [sed-command-in-linux-unix-with-examples](https://www.geeksforgeeks.org/linux-unix/sed-command-in-linux-unix-with-examples/)

* echo: display text in the terminal.
* cat: display file contents and combine multiple files.
  * cat file_name: View the Content of a Single File
  * cat file_name1 file_name2: View the Content of Multiple Files
  * cat > file_name: Create a New File and Add Content Using cat Command
    * Type your content
    * Press Ctrl + D to save and exit
  * cat file1 file2 > new_file: Copy or Merge File Contents

  [cat-command-in-linux-with-examples](https://www.geeksforgeeks.org/linux-unix/cat-command-in-linux-with-examples/)

* chmod: modify file and directory permissions. It controls who can read, write, or execute a file by setting access rights for the owner, group, and others.
  * r (Read): Permission to look inside the file and see its contents.

    w (Write): Permission to edit, modify, or delete the file.

    x (Execute): Permission to run the file as a program or script.

    Nine characters, first three represent the permision of the owner and middle three represent the permision of the group and last three represent the permision of the others.

  * Example: chmod 745 newfile.txt (chmod -rwxr--r-x newfile.txt)

    Owner (7): rwx : read, write, execute

    Group (4): r-- : read only

    Others (5): r-x : read & execute
  
  * 0:---, 1:--x, 4:r--, 5:r-x, 6:rw-, 7:rwx

  [chmod-command-linux](https://www.geeksforgeeks.org/linux-unix/chmod-command-linux/)

  [set-file-permissions-linux](https://www.geeksforgeeks.org/linux-unix/set-file-permissions-linux/)

* tar: create, view, extract, and manage archive files.
  * tar -cvf file.tar *.c: create a file.tar file compressing all .c format files in the directory.
  * -c: Creates a new archive
  * -f: Specifies the name of the archive file
  * -t: Lists files inside an archive
  * -v: Displays the archiving process
  * -z: Applies gzip compression
  
  [tar-command-linux-examples](https://www.geeksforgeeks.org/linux-unix/tar-command-linux-examples/)
  
[GeeksforGeeks – Basic Linux Commands](https://www.geeksforgeeks.org/linux-unix/basic-linux-commands/)

<br>

---

#### Shell Scripting

* shell script: a text file containing a sequence of commands that a Linux or Unix-like operating system can execute.
  * Shebang: '#!/bin/bash', tells the system which interpreter to use to read the file. /bin/bash means "use the Bash shell to run these commands."
  * cron: You can hook a .sh file up to tools like cron (a Linux task scheduler) to run backups, updates, or maintenance scripts automatically
* Variables
  * Global Variables: declared outside any function and can be accessed anywhere in the script,
  * Local variable: declared inside a function using the local keyword and is only accessible within that function.
* Decision Making
  * If–Else Statement: 
  
    if-fi

    if-else-fi

    if-elif-else-fi

    ```
    #!/bin/bash

    Age=17

    if [ "$Age" -ge 18 ]; then
        echo "You can vote"
    else
        echo "You cannot vote"
    fi
    ```

* Loop: 
  
  ```
  #!/bin/bash

  for n in a b c;
  do
    echo $n
  done
  ```

  [bash-scripting-for-loop](https://www.geeksforgeeks.org/linux-unix/bash-scripting-for-loop/)

* Functions

  ```
  #!/bin/bash

  myFunction() {
      echo "Hello World from GeeksforGeeks"
  }

  myFunction
  ```

[bash-scripting-introduction-to-bash-and-bash-scripting](https://www.geeksforgeeks.org/linux-unix/bash-scripting-introduction-to-bash-and-bash-scripting/)

<br>

---

#### SSH (Secure Shell)

* A network protocol that provides a secure way to access a computer (server) over an unsecured network.
  
* SSH Client: A program (like OpenSSH or a built-in terminal) that initiates the connection.

* SSH Server: A background process (daemon) running on the remote machine that listens for incoming connection requests (usually on Port 22).

* SSH Keys: A pair of long strings of characters. You keep the private key secret on your machine and place the public key on any server you want to access.
  
* Use cases
  * Remote Command Line
  * Secure File Transfer: SFTP (SSH File Transfer Protocol)
  * Git Operations.
  * Port Forwarding (Tunneling): 
  
    Forwarding traffic from a local port to a remote server, often used to access services behind a firewall or to secure unencrypted traffic. 
  
    Used in jumper servers. Instead of exposing all your sensitive database servers or application servers to the public internet, you hide them in a private network. You only allow one machine—the jumper—to be reachable from the outside.
  
[what-is-ssh](https://www.cloudflare.com/en-gb/learning/access-management/what-is-ssh/)

<br>

---

### Kernel

* A kernel is the core part of an operating system. The kernel manages system resources, such as the CPU, memory and devices. It handles tasks like running programs, accessing files and connecting to devices like printers and keyboards.
  
[kernel-in-operating-system](https://www.geeksforgeeks.org/operating-systems/kernel-in-operating-system/)
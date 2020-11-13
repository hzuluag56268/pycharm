Suppose your home directory /home/repl contains a repository called
dental, which has a sub-directory called data. Where is information
about the history of the files in /home/repl/dental/data stored?
/home/repl/dental/.git

/home/repl/dental/git status- to discover which file(s) have been changed

git diff filename. git diff without any filenames will show you all the changes
in your repository, while git diff directory will show you the changes to the files in some directory.

diff - -git a / report.txt  b / report.txt
index e713b17. .4 c0742a 100644
--- a / report.txt
+++ b / report.txt

@ @ -1, 4 + 1, 5 @ @
-  # Seasonal Dental Surgeries 2017-18
+  # Seasonal Dental Surgeries (2017) 2017-18
+  # TODO: write new summary

git diff -r HEAD path/to/file

You commit changes to a Git repository in two steps:
Add one or more files to the staging area.
Commit everything in the staging area.
To add a file to the staging area, use git add filename.
commit;  git commit -m "Adding a reference."
git log -   is used to view the log of the project's history.

This concludes chapter 1,
where you learned about git diff, git status, git add and git commit


...


Use -git log -2 - to see the las 2 hashes of recent commits, and then -git show- with
    the first few digits of a hash to look at the most recent commit
    as with git diff

git log displays the overall history of a project or file
git annotate let's you see who modified a file and when.
To see the changes between two commits, can use git diff ID1..ID2

git clean -n will show you a list of files that are in the
repository, but whose history Git is not currently tracking
git clean -f will then delete those files.

git config --list with one of three additional

--system: settings for every user on this computer.
--global: settings for every one of your projects.
--local: settings for one specific project.

git config --global user.email rep.loop @ datacamp.com

 ...:



git checkout -- data/northern.csv     to undo unstaged changes

git reset will unstage files that you previously staged using git add
git reset HEAD path/to/file
git checkout -- path/to/file

then git checkout 2242bd report.txt would replace the current version
of report.txt with the version that was committed on October 16. Notice
that this is the same syntax that you used to undo the unstaged
changes, except -- has been replaced by a hash.

.......


git branch
hard haro dliu hiuou  adffeseesli

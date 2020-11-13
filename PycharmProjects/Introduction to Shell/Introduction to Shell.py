'''Manipulating files and directories'''


pwd (short for "print working directory").
pwd tells you where you are. To find out what's there, ' \
 'type ls (which is short for "listing")

ls /home/repl shows you what's in your starting directory

If you are in the directory /home/repl, the relative path seasonal
specifies the same directory as the absolute path /home/repl/seasonal.


you can move around in the filesystem using the command cd
(which stands for "change directory").

You are in /home/repl/.
Change directory to /home/repl/seasonal using a relative path.

/home/repl/seasonal, then cd .. moves you up to /home/repl

One final special path is ~ (the tilde character), which means
"your home directory", such as /home/repl. No matter where you are,
ls ~ will always list the contents of your home directory,
and cd ~ will always take you home.

makes copy: cp original.txt duplicate.txt

'cp seasonal/autumn.csv seasonal/winter.csv backup' copies to backup dir
cp seasonal/summer.csv backup/summer.bck   copies to backup dir with different name

mv moves it from one directory to another, mv autumn.csv winter.csv ..
.. always refers to the directory above your current location
mv seasonal/spring.csv seasonal/summer.csv backup     same as cp

mv can also be used to rename files. If you run: mv course.txt old-course.txt

rm thesis.txt backup/thesis-2017-08.txt    removes files
you can use a separate command called  rmdir to remove dir

    mkdir yearly   to create dir
    
mv ~/people/agarwal.txt /tmp/scratch    Move a file to a dir in the temporary dir






...........


cat agarwal.txt         t will print all the files whose names you give it

less seasonal/spring.csv seasonal/summer.csv
:n :p :q  spacebar is to change page

head people/agarwal.txt   To display the first 10 lines
head -n 5 seasonal/winter.csv   (-n 5 )To display a certain number of times

Run ls with the two flags, -R and -F, and the absolute path to your home directory to see everything it contains. (
ls -R -F ~

Use tail with the flag -n +7 to display all but the first six lines of

man tail      Using man will display help information about that command

 If you want to select columns, use cut
cut -f 2-5,8 -d , values.csv
means "select columns 2 through 5 and columns 8, using comma as the separator". cut
uses -f (meaning "fields") to specify columns and -d (meaning "delimiter") to specify the separator.

!head or !cut, which will re-run the most recent use of that command.
!55 to re-run the 55th command in your history
the word history will print a list of commands you have run recently

head and tail select rows, cut selects columns,
and grep selects lines according to what they contain
grep bicuspid seasonal/winter.csv prints lines that contain "bicuspid".
Common flags:
-c: print a count of matching lines rather than the lines themselves
-h: do not print the names of files when searching multiple files
-i: ignore case (e.g., treat "Regression" and "regression" as matches)
-l: print the names of files that contain matches, not the matches
-n: print line numbers for matching lines
-v: invert the match, i.e., only show lines that don't match

paste that can be used to combine data files instead of cutting them up.

        ...:


head -n 5 seasonal/summer.csv > top.csv
head's output is put in a new file called top.csv

head -n 5 seasonal/summer.csv | tail -n 3
The pipe symbol tells the shell to use the output of the command on
the left as the input to the command on the right.

The command wc (short for "word count") prints the number of characters words
and lines in a file. You can make it print only one of these using -c, -w, or -l respectively.
grep 2017-07 seasonal/spring.csv | wc -l
-v flag select all except the given word

head -n 3 seasonal/s*
to get the first three lines from both seasonal/spring.csv and seasonal/summer.csv

? matches a single character, so 201?.txt will match 2017.txt or
2018.txt, but not 2017-01.txt.
[...] matches any one of the characters inside the square brackets,
so 201[78].txt matches 2017.txt or 2018.txt, but not 2016.txt.
{...} matches any of the comma-separated patterns inside the curly brackets
, so {*.txt, *.csv} matches any file whose name ends with .txt or .csv, but not files whose names end with .pdf.

the command sort puts data in order.  the flags -n and -r can be used to sort numerically and reverse the order
piping sort -n to head shows you the largest values.

Another command that is often used with sort is uniq, whose job is to remove adjacent duplicated lines
cut -d , -f 2 seasonal/winter.csv | grep -v Tooth | sort |uniq -c

If you decide that you don't want a program to keep running, you can type Ctrl + C to end it

wc -l seasonal/* | grep -v total | sort -n | head -n 1
 to find out how many records are in the shortest of the seasonal data files.

....


HOME	User's home directory	/home/repl
PWD	Present working directory	Same as pwd command
SHELL	Which shell program is being used	/bin/bash
USER	User's ID	repl

To get a complete list (which is quite long), you can type set in the shell.

echo, which prints its arguments.
To get the variable's value put a dollar sign $
echo $OSTYPE

To create a shell variable, you simply assign a value to a name:
training=seasonal/summer.csv

for filetype in docx odt pdf ; do echo $filetype ; done
for filename in seasonal/*.csv; do echo $filename; done
A common mistake is to forget to use $ before the name of a variable.

for file in seasonal/*.csv; do head -n 2 $file | tail -n 1; done
for file in seasonal / *.csv; do grep 2017-07 $file | tail -n 1; done


rm 'July 2017.csv' '2017 July data.csv'

...

Run nano names.txt to edit a new file in your home directory and enter the following four lines:

Lovelace
Hopper
Johnson
Wilson
to save what you have written, type Ctrl + O

Use nano dates.sh to create a file called dates.sh that contains this command:
cut -d , -f 1 seasonal/*.csv

Use bash to run the file dates.sh
bash dates.sh

For example, if unique-lines.sh contains sort $@ | uniq, when you run:

bash unique-lines.sh seasonal/summer.csv
the shell replaces $@ with seasonal/summer.csv and processes one file. If you run this:

bash unique-lines.sh seasonal/summer.csv seasonal/autumn.csv
it processes two data files, and so on.

Ctrl + O to write the file out, then Enter to confirm the filename, then Ctrl + X to exit the editor.

s well as $@, the shell lets you use $1, $2
cut -d , -f $2 $1
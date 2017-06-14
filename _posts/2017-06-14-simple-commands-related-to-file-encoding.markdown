---
layout: post
title:  "Simple commands related to file encoding"
date:   2017-06-14
comments: true
---

Today I had the need to shift a files encoding from `ISO-8859-1` to `UTF-8`. And it seems pretty simple to do this on Mac or Linux.

Firstly let's try check a file's encoding:

```
$ file -I TheFile.csv

TheFile.csv: text/plain; charset=unknown-8bit
```

Well, even though it didn't tell me it is for sure `ISO-8859-1` but I had 
previous viewed the file in an editor such as Atom(by changing encoding to `ISO-8859-1`and it seems correct to me).

Then let's use this command to change the encoding:

```
$ iconv -f ISO-8859-1 -t UTF-8 TheFile.csv
```

This will output the content of the file after shifting the encoding.
You might want to check if it is correct. Otherwise use the following command to save the content in a new file:

```
$ iconv -f ISO-8859-1 -t UTF-8 TheFile.csv > TheFile.UTF-8.csv
```

Sanity check the new file length is the same with the old one:

```
$ wc -l TheFile.csv
$ wc -l TheFile.UTF-8.csv
```

Each command above should output the same number of lines of those csv files.

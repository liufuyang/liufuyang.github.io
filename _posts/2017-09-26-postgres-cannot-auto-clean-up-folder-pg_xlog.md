---
layout: post
title:  "Postgres cannot auto clean up folder pg_xlog"
date:   2017-09-26
comments: true
---

Today I encountered this issue:

While inserting some data into a postgres db, suddenly an error stopped the process:

```
...
could not write to file "pg_xlog/xlogtemp.2752": No space left on device
SSL SYSCALL error: EOF detected
```

Then I noticed that under `/var/lib/postgresql/9.5/main/pg_xlog/` there are many
WAL(Write Ahead Log) files and it eats up the available 32GB space on my instance. 

Did some google search and it brought me here first:
http://blog.endpoint.com/2014/09/pgxlog-disk-space-problem-on-postgres.html, which
gives some nice discussion but my error log file contains nothing about `archive command failed` so
it looks like the issue for me is a postgres config setup issue, because by default 
the postgres should be able to auto clean up this folder.

### The solution:

Check config file `/etc/postgresql/9.5/main/postgresql.conf` to make sure:

* `max_wal_size = 1GB` is set, which controls the "maximum size to let the WAL grow to between automatic WAL checkpoints." However this is not enough, as it is a soft limit. "WAL size can exceed max_wal_size under special circumstances, like under heavy load, a failing archive_command, or a high wal_keep_segments setting." So we also check the setting below:
* `wal_keep_segments = 0` is set, or a very small number is here. This is the minimum number of past log file segments kept in the pg_xlog directory. My previous config has a setting of more than 8000, with each file having size 16MB, of course
	this setting will stop the auto clean up process...
	
So, after you change those two settings and restart the postgres service, you should
notice the postgres automatically cleaned up the folder `pg_xlog` for you.

### Some caveats: 

* Note that in the config file, the last config in the will take effect. I had the 
	problem earlier that with a setting in the end `wal_keep_segments = 8125`, I cannot
	change the setting by put `wal_keep_segments = 0` in the earlier part of the file.
* Oneway to find out where exactly the setting is defined, you could simply run this 
SQL command in psql:
	```
$ select name, setting, sourcefile, sourceline from pg_settings where name = 'wal_keep_segments';
		
	name        | setting |                sourcefile                | sourceline
	-------------------+---------+------------------------------------------+------------
	wal_keep_segments | 8125      | /etc/postgresql/9.5/main/postgresql.conf |        664
	```

	Then you can try edit the setting on that specific line number in that config file.
	
### Other references:
Search `wal_keep_segments` at https://www.postgresql.org/docs/9.5/static/runtime-config-replication.html

Search `max_wal_size` at https://www.postgresql.org/docs/9.5/static/runtime-config-wal.html

---
layout: post
title:  "SSH Tunnel"
date:   2017-06-08
comments: true
---

Let's say you have some host on `remote.address.com`, which has a postgres 
database running in it (on port 5432). And you have SSH access to this host.

And you hope to connect to that database from a program on your local laptop `localhost`.

How to do that? Well, use a SSH Tunnel.

Basically, open a shell on your local laptop and run command such as 

```
ssh -NL 50001:localhost:5432 user@remote.address.com
```

This means on my local latptop, open a tunnel on port 50001 and connect it to (or transfer whatever goes in and out there to) port 5432 on `user@remote.address.com`.

Then you can connect the database on your laptop with connect string as:
```
Host: localhost
Port: 50001
```

Cool, isn't it? :)
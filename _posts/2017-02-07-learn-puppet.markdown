---
layout: post
title:  "Learning Puppet"
date:   2017-02-07
comments: true
---

## What is puppet?

* It's a software piece that allows automation. You configure a central server
and the clients synchronize themselves to it.
* It's a descriptive system. You decide that a file should be there, with some 
permissions, content, that a software package should be installed, and at each run,
the system ensures that it ends in the state you have described.

## Getting started

Let's try learn some puppet via this youtube video:

<style>.embed-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; } .embed-container iframe, .embed-container object, .embed-container embed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
</style>
<div class="embed-container">
  <iframe title="YouTube video player" width="640" height="390" 
    src="//www.youtube.com/embed/Hiu_ui2nZa0" 
    frameborder="0" allowfullscreen="">
  </iframe>
</div>
<br>


## Prepare a AWS EC2 for hands on learning
As suggested in the video, one should setup an AWS EC2 for the hands on session. I just created a ubuntu instance on AWS EC2, with configurations out of the box. After set up the ssh private key I then use ssh to login as user `ubuntu`.

Then install puppetmaster on the master instance and puppet on the agent instance:

* on master instance: `$ sudo apt-get install puppetmaster`
* on agent instance: `$ sudo apt-get install puppet`

Then edit `/etc/hosts` file on both master and agent so they can both use host name to find each other (master.example.com and agent.example.com)


## Connect agent puppet with master

Edit `/etc/puppet/puppet.config` file to add line:
`server=master.example.com` in the config file.

* on agent, do `$ sudo puppet agent --no-daemonize --onetime --verbose`
* on master check it with `$ sudo puppet cert list`
* on master sign cert with `$ sudo puppet cert sign "agent.lifeinweeks.ml"`
* on agent, do connect again: `$ sudo puppet agent --no-daemonize --onetime --verbose`
* Then the agent should be connected to puppet and showing some state file is created.

## Make some puppet files

On master, `$ cd /etc/puppet/manifests/` and `$ vim site.pp`:

```
import 'classes/*.pp'

class toolbox {
  file {
    '/usr/local/sbin/puppetsimple.sh':
      owner => root,
      group => root,
      mode  => 0755,
      content => "#!/bin/sh\npuppet agent --onetime --no-daemonize --verbose $1\n",
  }
}

node "agent.lifeinweeks.ml" {
  include toolbox
}
```

Then you can run `$ sudo puppet agent --no-daemonize --onetime --verbose` again on agent to let the puppet master apply changes onto agent.

And you can do `$ sudo puppetsimple.sh` now to run the same command.

More stuff on the video, not going to write more here...

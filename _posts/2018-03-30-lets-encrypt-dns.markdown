---
layout: post
title:  "Let's Encrypt: Free HTTPS Wildcard Certificates Are Now Available"
date:   2018-03-30
comments: true
---

As we can see from [here](https://letsencrypt.org/upcoming-features/) that 
the **Wildcard Certificates** feature is already enabled on March 13, 2018, it is
very nice that now for everyone out there who needs wildcard certificates on 
domain names as Let's Encrypt is free :)

I checked on with some [posts](https://securityboulevard.com/2018/03/free-https-wildcard-certificates-are-now-available/) and found it that "*To get such certificates through Let’s Encrypt users will need to use an updated ACME client that supports version 2 of the protocol. Obtaining a wildcard certificate also will require DNS-based domain ownership validation, where a verification token issued by Let’s Encrypt will have to be added in a DNS TXT record for the domain.*"

I have a small dummy server with a domain name pointed at, though I don't really
need to use this wildcard feature yet, but nevertheless it would be fun just to
try out the DNS-based domain ownership validation anyway. 

Also it seems with the tool `certbot`, one could use some DNS plugins to even make 
automatic certificate generation an easy job. [Like what this post is doing](http://www.eigenmagic.com/2018/03/14/howto-use-certbot-with-lets-encrypt-wildcard-certificates/).

As my domain name is provided via freenom, it seems there is no available plugins there? 
So I am going to use a manual way to get certificates. You could also probably get a general idea of how this DNS-based domain ownership works.

## How to use `certbot` to get wildcard certificates - manually

So firstly, simply use docker to run `certbot` command on your server (or probably can be done anywhere?)
like this:
```
$ sudo docker run -it --rm --name certbot \
     -v "/etc/letsencrypt:/etc/letsencrypt"  \
     -v "/var/lib/letsencrypt:/var/lib/letsencrypt" \
     certbot/certbot certonly --manual \
     --preferred-challenges dns -d *.lifeinweeks.ml
```
Note that:
* `--preferred-challenges dns` means to use DNS-based domain ownership validation
* `-d *.lifeinweeks.ml` means to specify the domain name (the certificates signed for) as \*.lifeinweeks.ml
* Certificates will be saved under `/etc/letsencrypt/live`, you will see a message later when the command 
finishes successfully later.

Yep, so run this, you see output as:
```
...
Performing the following challenges:
dns-01 challenge for lifeinweeks.ml

-------------------------------------------------------------------------------
NOTE: The IP of this machine will be publicly logged as having requested this
certificate. If you're running certbot in manual mode on a machine that is not
your server, please ensure you're okay with that.

Are you OK with your IP being logged?
-------------------------------------------------------------------------------
(Y)es/(N)o: Y

-------------------------------------------------------------------------------
Please deploy a DNS TXT record under the name
_acme-challenge.lifeinweeks.ml with the following value:

NCBwbnEtPSA3sdc8ut-6Df8A55xDWnrbn9CwrUc4FLI

Before continuing, verify the record is deployed.
-------------------------------------------------------------------------------
Press Enter to Continue
```

So now you basically just have to what it asks for.

* Firstly go to your domain name provider to add a DNS TXT record. I do this on freenom. 
Click "Save Changes" to add the record.
![freenom](/assets/2018-03-30-lets-encrypt-dns/freenom.png)
* Secondly, use the following command in another terminal to check the changes have taken effect:
     ```
     $ dig -t txt +short _acme-challenge.lifeinweeks.ml
     ```
     which gives out
     ```
     "NCBwbnEtPSA3sdc8ut-6Df8A55xDWnrbn9CwrUc4FLI"
     ```
     then this means you are good to go. Go press enter to continue at the previous terminal with `certbot`

After hitting enter again, you get signed certificates:
```
Before continuing, verify the record is deployed.
-------------------------------------------------------------------------------
Press Enter to Continue
Waiting for verification...
Cleaning up challenges

IMPORTANT NOTES:
 - Congratulations! Your certificate and chain have been saved at:
   /etc/letsencrypt/live/lifeinweeks.ml-0001/fullchain.pem
   Your key file has been saved at:
   /etc/letsencrypt/live/lifeinweeks.ml-0001/privkey.pem
   Your cert will expire on 2018-06-27. To obtain a new or tweaked
   version of this certificate in the future, simply run certbot
   again. To non-interactively renew *all* of your certificates, run
   "certbot renew"
 - If you like Certbot, please consider supporting our work by:

   Donating to ISRG / Let's Encrypt:   https://letsencrypt.org/donate
   Donating to EFF:                    https://eff.org/donate-le
```

Congratulations, now you can happily serve those certificates on your server to have https enabled on your domain name :D

Reference: [[1]](https://mp.weixin.qq.com/s?__biz=MzIwMzg1ODcwMw==&mid=2247487566&idx=1&sn=3eea2bc71b123967fca82934ee1353e7&chksm=96c9a62ea1be2f3803a053df9ec46970a2fe24c64c08c1695b909ed6f836168a840dac777282#rd)
[[2]](https://securityboulevard.com/2018/03/free-https-wildcard-certificates-are-now-available/)
[[3]](http://www.eigenmagic.com/2018/03/14/howto-use-certbot-with-lets-encrypt-wildcard-certificates/)

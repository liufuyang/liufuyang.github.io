---
layout: post
title:  "DevOps learning workshop [WIP...]"
date:   2018-07-21
comments: true
---

As recently I have the interest to test the TiDB out, I feel I need to have some
Kubernetes clusters somewhere ready so I could test the deployment of a TiDB
cluster.

The idea is to create some private Kubernetes cluster on my private AWS account.
Earlier today I tired to use AWS's EKS service, seems kind of easy. 
However creating and shutting down those created clusters and stacks, seems 
pretty time consuming by using the UI. We need to figure out a way to make
configurations as code.

Also, at the place I am working, the devops guys uses Terraform for most of their
AWS configurations. So here I am thinking about making a small learning plan
to get some needed knowledge, and in the same time to see if it is possible 
in the end to make a TiDB cluster work on a Kubernetes cluster.

So the basic idea is this:

1. Finishing AWS solution architect course on Udemy
1. Try out the Terraform course on Udemy
1. Try use Terraform code to control a EKS service
1. Keep an eye on the tidb-operator project (hopefully it will be open source soon) 
1. When everything is ready, try deploy TiDB or whatever via helm onto a k8s stack
    
I think it would be fun it we could make it work in the end.

Here I will use this post as a notebook.

To be continued...

---
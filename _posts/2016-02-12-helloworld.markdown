---
layout: post
title:  "HelloWorld"
date:   2016-02-13
comments: true
---


![HelloWorld](/assets/2016-02-12-helloworld/start.png)*Above figure is from [waitbutwhy](http://waitbutwhy.com)*

Hello world! My name is Fuyang. Maybe I will continue using this blog to update the stuff I am learning for fun :)

Github Pages together with Jekyll could be a very nice solution for building a blog since it is relatively simple to set things up.

And locally I can just use Atom and git to edit and publish posts very easily. Also it is very easy to adding pictures and maths equations, writing a blog post now feels like writing a page in latex.

```
function(){
  this.stuff = 'looks good';
}
```

{% highlight rust linenos %}
async fn first_function() -> u32 { .. }

async fn another_function() {
    // Create the future:
    let future = first_function();
    
    // Await the future, which will execute it (and suspend
    // this function if we encounter a need to wait for I/O): 
    let result: u32 = future.await;
    ...
}
{% endhighlight %}

I love it :)

$$e^{i \pi} = -1$$

If you want to know how this page is built, check out [here](https://help.github.com/articles/using-jekyll-with-pages/) and [here(it didn't work for me, but you can get the idea here)](https://github.com/barryclark/jekyll-now).

Also a special thank to my colleague [Rasmus Berg Palm](http://rasmusbergpalm.github.io/2016/02/12/dense-codes.html), from whom I copied the blog setup.

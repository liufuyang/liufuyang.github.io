---
layout: post
title:  "Tokio 和 Async IO 到底都是些啥玩意?"
date:   2019-11-10
comments: true
---

我已经关注Rust一段时间了, 也在慢慢自学一些相关内容. 最近
Async IO, 也就是异步IO的一些标准语法也已经包含在了Rust
稳定版本里面比如 `async` 和 `await` 关键字.

可我之前在学习Async IO的过程当中, 一直有些疑惑. 比如那些经常听说的库 `tokio`, `mio`, `futures` 等等, 
到底都是干嘛用的? Rust的 Async IO 和 其他语言, 比如Go的协程, 是不是类似的概念?

后来无意当中搜到了Manish Goregaokar(Mozilla的一名研究工程师, 参与搭建火狐浏览器核心引擎)在2018年1月写的一篇博文:
[What Are Tokio and Async IO All About?](https://manishearth.github.io/blog/2018/01/10/whats-tokio-and-async-io-all-about/). 
读完之后有一种豁然开朗的感觉. 虽然现在已经过了挺久,
有些内容也已经稍微陈旧, 但我觉得还是很好的一篇适合入门者了解
Rust Async IO的博文. 特此尝试翻译成中文, 方便大家学习.
如果您有任何疑问或者建议, 非常欢迎您在本文下方留言.

Translation of Manish's orignal blog has been approved by the original author.
翻译已经获得原作者Manish Goregaokar的许可. 下面开始正文.

---

# What Are Tokio and Async IO All About?
# Tokio 和 Async IO 到底都是些啥玩意?
作者: [Manish Goregaokar](https://manishearth.github.io/blog/2018/01/10/whats-tokio-and-async-io-all-about/).
写于2018-01-10.

近期, Rust社区在“Async IO”上投入大量关注, 很多
工作在这个叫 [tokio](https://github.com/tokio-rs/)
的库上展开. 这非常棒!

但是对社区内部很多不和网络服务器等等打交道的同学来说,
还是挺难搞清楚那些人们在 Async 这块开发是想要达成怎样的目.
当人们在 (Rust) 1.0版本出来的时候谈论与此相关的话题时, 
我也是一头雾水, 从来没有接手过相关的工作.

Async IO 到底都是在搞啥? 协程又是什么东西? 
轻线程(lightweight thread)呢? 这些概念直接有关系吗?

# 我们在想解决怎样的问题?

Rust 的一个卖点就是“并发不可怕”. 但是那种,
需要管理大量输入输出所限的任务的并发 - 那种能在Go, 
Elixir, Erlang 这类语言里看到的并发方式 - 在
Rust 里并不存在.

打个比方, 比如说你想搞个网络服务器之类的东西,
来在任意单独的时刻来管理成千上万的请求(这种问题也被
称作 “[c10k](https://en.wikipedia.org/wiki/C10k_problem)问题”). 总之来说, 我们要解决的问题有着极大量的输入输出 (I/O, 通常是网络I/O) 所相关的任务.

“同时管理N件事儿”非常适合来用线程 thread 来实现. 但是...要创建上千个线程吗? 那听上去有点真太多了. 而且线程还挺“昂贵”: 每一个线程需要分配一个挺大的栈 stack, 建立线程需要经手不少系统调用, 而且系统上下文切换也非常耗资源.

当然了, 上千个线程也不可能真的能在CPU上真正同时运行. 你只有屈指可数的几个内核(core), 在某一时刻, 只有一个线程运行在一个核上.

但对于这种网络服务器类型的系统来说, (如果真去创建大量的线程的话), 其实大部分创建出来的线程都在不工作状态, 它们都会在做大量的网络等待. 这些线程要么在监听来访的请求 (request), 要么在等待回复 (response) 被发送出去.

所以你用这种普通线程的, 来调度这种输入输入的任务的时候, 系统调用把控制交付给操作系统内核, 然后内核并不会立马把控制返回给你, 因为输入或者输出还没有结束. 事实上, 此时系统内核会利用这个机会去换一个另外的线程来运行, 等输入输出操作结束后(比如, 等操作非阻塞 / unblock 的时候), 再换到你之前的线程上运行. 如果没有 Tokio 和那些兄弟类库的话, 你就得用这种本办法来解决这种问题. 创建超级多的线程, 让操作系统去做任务切换.

但是, 正如我们已经知晓的那样, 线程并不能很好的在这种问题上被拓展 (scale). 然而也有例外[见下方索引1].

我们需要比较“轻”的线程.

---
# 轻量级线程 - Lightweight 线程

{% highlight golang %}
listener, err = net.Listen(...)
// handle err
for {
    conn, err := listener.Accept()
    // handle err

    // spawn goroutine:
    go handler(conn)
}
{% endhighlight %}


 🗼
---

索引
1. 值得提醒一下并不是说创建大量线程是完全不可行的方案. 比如 Apache 就采用大量系统线程. 系统线程也经常可以在这种问题上被采用.
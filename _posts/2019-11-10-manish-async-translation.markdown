---
layout: post
title:  "Tokio 和 Async IO 到底都是些啥玩意?"
date:   2019-11-10
comments: true
---

我已经关注 Rust 一段时间了, 也在慢慢自学一些相关内容. 最近
Async IO, 也就是异步IO的一些标准语法也已经包含在了Rust
稳定版本里面比如 `async` 和 `await` 关键字.

可我之前在学习 Async IO的过程当中, 一直有些疑惑. 比如那些经常听说的库 `tokio`, `mio`, `futures` 等等, 
到底都是干嘛用的? Rust的 Async IO 和 其他语言, 比如Go的协程, 是不是类似的概念?

后来无意当中搜到了 Manish Goregaokar (Mozilla的一名研究工程师, 参与搭建 Sevro 浏览器核心引擎)在2018年1月写的一篇博文:
[What Are Tokio and Async IO All About?](https://manishearth.github.io/blog/2018/01/10/whats-tokio-and-async-io-all-about/). 
读完之后有一种豁然开朗的感觉. 虽然现在已经过了挺久,
有些内容也已经稍微陈旧, 但我觉得还是很好的一篇适合入门者了解
Rust Async IO的博文. 特此尝试翻译成中文, 方便大家学习.
如果您有任何疑问或者建议, 非常欢迎您在本文下方留言.

Translation of Manish's orignal blog has been approved by the original author.
翻译已经获得原作者Manish Goregaokar的许可. 下面开始正文.

---
<br>

# **Tokio 和 Async IO 到底都是些啥玩意?**
# What Are Tokio and Async IO All About?

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

# **我们在想解决怎样的问题?**

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

所以你用这种普通线程的, 来调度这种输入输入的任务的时候, 系统调用把控制交付给操作系统内核, 然后内核并不会立马把控制返回给你, 因为输入或者输出还没有结束. 事实上, 此时系统内核会利用这个机会去换一个另外的线程来运行, 等输入输出操作结束后(比如, 等操作非阻塞 / unblock 的时候), 再换到你之前的线程上运行. 如果没有 Tokio 和那些兄弟类库的话, 你就得用这种办法来解决这种问题 - 创建超级多的线程, 让操作系统去做任务切换.

但是, 正如我们已经知晓的那样, 线程并不能很好的在此问题上被拓展 (scale). 然而也有例外[见下方索引1].

我们需要比较“轻”的线程.

---

<br>

# **轻量级线程 - Lightweight 线程**

我认为一种容易理解轻量级线程的方法, 是暂时先不要管 Rust, 而看看 Go 是怎么做的. 因为 Go 在这一点公认做的很好.

Go不采用操作系统线程, 而是定义了一种轻量级线程, 叫做"goroutines". 你用 `go` 关键字来启动这种线程. 比如一个网络服务器可能这样来实现:

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

这里用一个循环, 等待新的 TCP 链接, 之后创建一个 goroutine , 这个 goroutine 会开始运行 `handler` 函数来响应得到的链接. 每一个链接都会成为一个新的 goroutine, 而且这个 goroutine 会在 `handler` 执行完毕之后被销毁. 与此同时, 主循环还在继续运行, 因为它跑在一个不同的 goroutine 上.

那么问题是, 如果并没有"真的"(操作系统)线程, 那这一切是怎样发生的?

Goroutine 是一种"轻量级"线程的实例. 操作系统并不知晓这种线程, 它只看到 N 个自己的线程被 Go 运行时(runtime)所拥有, 而 Go 运行时把 M 个 goroutines 映射到这 N 个操作系统上面 [见下方索引2]. Go 运行时负责来回切换那些 goroutine, 就如操作系统调度器一般. 它之所以能这么做事因为 Go 代码已经可以被中断进而进行垃圾回收, 所以 Go 运行时的调度器总可以让某个 goroutine 停下来. 这个调度器同时也知晓输入输出操作, 当一个 goroutine 等待输入输出时, 它会把自己"交还"(yields)给调度器.

本质上来说, 一个编译出来的 Go 函数会有一堆断点散落在其过程当中, 在每一个点上它会告诉调度器和垃圾回收"你要让我暂停?, 那好吧, CPU归你了" (或者"我正在等待, 你来接手CPU的使用吧")

当一个 goroutine 从系统线程上切走的时候, 一些寄存器会被保存, 程序计数器会切换到新来的 goroutine 上.

但之前这个 goroutine 的栈 stack 怎么办? 操作系统线程有一个很大的栈, 而且你基本上得有个栈才能让你写的函数或者代码来工作.

Go 之前采用的解决方法是用分段的栈(segmanted stacks). 对多数语言来说, 包括 C, 一个线程之所以需要一个很大的栈, 是因为它们需要一个连续的栈. 而且栈不能像那种随意增长的缓存一样被"从新分配 / reallocated", 因为我们需要栈数据保持在原位, 以便那些指向栈上的指针们可以持续保持有效. 因此, 我们预留所有我们觉得需要的的空间在到栈上(大概8MB), 然后就只能寄希望于之后不需要更多了.

但是这种对于栈要连续的需求并不是严格必须的. 在 Go 里, 栈是由很多小的区块组成的. 当开始调用一个函数的时候, Go 会看一个栈上是否还有足够的空间来跑这个函数, 如果不够, 分配一块新的小空间作为栈, 然后让函数跑在上面. 所以如果你有上千个线程, 每个在做一些小量的工作, 它们总共占用着很多很小的小栈, 这就没什么问题.

现如今, Go 实际上采用了不同的一种方式. 它会[复制栈](https://blog.cloudflare.com/how-stacks-are-handled-in-go/). 我上面提到, 由于需要栈数据保持在原位, 栈不能被"从新分配 / reallocated". 但其实这也不一定完全正确 - 因为 Go 还有垃圾回收, 它也知道每一个指针在哪里, 从而可以把指针重新定向到新的栈位置, 如果需要的话.

总之, 不管用分段的栈, 或者复制栈的方法, Go 的丰富的运行时可以让其很好的管理这些事务. Goroutines 非常廉价, 轻量, 你可以创建成千上万的 goroutine, 系统也不会有什么问题.

Rust 早先的时候支持轻量/"绿色"线程(我记得好像是用分段栈的方法). 但是, Rust 非常关心"不用的东西就不要花钱在上面", 所以如果支持轻量线程的话, 所有不需要轻量级线程的代码也得背上这个包袱. 因此 Rust 在1.0版本之前, 去掉了对轻量级线程的支持.

---
<br>

# **异步 I/O - Async I/O**

解决上述问题的一个关键基石就是异步 I/O. 如前所述, 如果采用常规的blocking I/O 的话, 当跟系统发出 I/O 请求的时候, 你的线程将被禁止继续运行(“被阻碍 / blocked”), 直到 I/O 操作最终完成. 这如果是发生在操作系统线程的话就显得没什么问题(因为操作系统调度会帮你完成所有工作), 可如果这是发生在轻量线程上的话, 你得负责将这个blocked轻量线程从操作系统线程上换下来, 换上另一个轻量线程.

这如何实现呢? 你得采用非阻塞 I/O (non-blocking I/O) - 这时当你跟系统发出 I/O 请求的时候, 你的线程可以继续工作而不用停止. 这个 I/O 请求将会在一段时间之后被内核执行. 之后你的线程, 在尝试访问 I/O 结果之前, 需要问问操作系统:“请问我刚才提交的 I/O 请求完成了吗?”

当然了, 要是你一直不断的去询问操作系统好了没好了没, 肯定显得比较啰嗦而且耗费资源. 而这就是为什么会一类像 **`epoll`** 这样的系统调用存在. 采用这种方式, 你可以把一些没有完成的 I/O 请求打成一捆, 然后告诉操作系统, 如果这一捆操作里有任何一个完成之后, 来把我的线程唤醒. 像这样的话, 你就可以用一个线程(一个操作系统线程)来负责换下等待 I/O 操作的轻量级线程. 而且当没事情的时候(比如 I/O 操作都在等待的时候), 这个线程在执行完最后一个 `epoll` 之后就直接进入睡眠状态了, 直到某个 I/O 操作完成之后, 操作系统将其再次唤醒.

(真实的过程很可能比上述的描述复杂很多, 但你现在应该能了解个大概了)

那么好了, 现在我们要把同样但机理引入到 Rust 当中, 一种做法就是通过 Rust 的 [mio](https://github.com/tokio-rs/mio) 库. 这个库提供了一种与平台无关的一组打包好的函数, 其中包括 non-blocking I/O 和 相对于每个平台的异步系统调用, 比如 `epoll/kqueue` 等等. 这个库算是一个组件库, 即便那些在过去在 C 里面用 `epoll` 的人会觉得这个库比较有用, 这个库并没有提供一种像 Go 那样方便的编程模型. 但我们可以通过叠加其他组建来达成我们在 Rust 里也可以方便处理异步输入输出的目标.

---
<br>

# **Futures**

**Futures** 是另一块解决这个问题的基石. 一个 `Future` 好比一个将来总会有值返回的承诺 / promise (事实上, 在Javascript里, 这个概念就直接被叫做了 `Promise`也就是承诺).

比如你请求在网络端口监听, 得到一个 `Future` (事实上, 是一个 `Stream`, 跟 future 差不多但是返回一连串的值). 这个 `Future` 一开始并没有受到任何响应, 但响应来到时它就会知道. 你可以 `wait()` 来等待在 `Future` 上, 这样可以以阻塞但方式等待直到结果返回, 你也可以 `poll()` 它, 问问它有没有结果已经返回了(有的话结果会给到你手里).

Futures 还可以被链接在一起, 因此你可以写些比如这样的东西 `future.then(|result| process(result))`. 那个给 `then` 的闭包自己也可以产生一个 future, 所以你可以继续往后面链接诸如 I/O 之类的操作. 在这些链接起来的 futures 上, 你得用 `poll()` 来取得进展; 每次你调用 poll() 的时候, 如果前一个 future 已经准备好了, 它会跳到下一个 future 上.

这算是在 non-blocking I/O 上的一个很不错的抽象框架.

链接 futures 就合链接 iterators 差不多. 每个 `and_then` (或者其他什么算子) 调用会返回一个包裹内部 future 的结构, 上面也可能含有一个其他的闭包(closure). 闭包 / closures 自己会带有它们的引用和数据, 所以这一整串看上去其实像一个小小的栈.


 # 🗼**Tokio**🗼
Tokio 本质上来说就是包在 mio 之上的一个抽象层, 提供了 futures 在其之上. Tokio 内部实现了一个核心事件循环 (core event loop). 你给它提供代码闭包(closure), 它会返回 future 给你. Tokio 所做的事情其实就是运行你给它的所有闭包, 并采用 mio 来高效地知晓哪个 future 已经准备完毕[注释3], 然后继续运行这些 futures (调用 `poll()` 来继续).

这种模式其实已经和 Go 在概念层面非常相似. 你得自己手动建立 Tokio 事件循环(也就是所谓的“调度器”). 然而建好之后, 你就可以给它多个做 I/O 的任务了, 事件循环会管理任务间的切换, 当一个任务 I/O 等待的时候去跑另一个任务. 重要的一点是 Tokio 是单线程的, 而 Go 的调度器可以利用多个系统线程来执行调度. 然而, 你可以把 CPU 要求高的任务发送到其他操作系统线程之上运行, 用管道来实现这种设计并不是很难.

当在概念层面和 Go 相似的时候, 代码层面还不是特别美观. 比如想这段 Go 代码:
```go
// error handling ignored for simplicity

func foo(...) ReturnType {
    data := doIo()
    result := compute(data)
    moreData = doMoreIo(result)
    moreResult := moreCompute(data)
    // ...
    return someFinalResult
}
```

在 Rust 就可能要写成这个样子:
```rust
// error handling ignored for simplicity

fn foo(...) -> Future<ReturnType, ErrorType> {
    do_io()
    .and_then(|data| do_more_io(compute(data)))
    .and_then(|more_data| do_even_more_io(more_compute(more_data)))
    // ......
}
```

看着不是很美观. [如果再加入分支和循环, 代码会更难看](https://docs.rs/futures/0.1.25/futures/future/fn.loop_fn.html#examples). 出现这种问题的关键是, 在 Go 里我们在代码就直接拥有一个个的中断点, 而在 Rust 里我们得编码这种链接在一起的算子来实现一种状态机. 好吧...

---
<br>

# **生成器和 async/await** - Generators and async/await

这就是为什么我们需要生成器(generator, 或者叫 coroutines).

[Generators](https://doc.rust-lang.org/nightly/unstable-book/language-features/generators.html) 是 Rust 的一个实验功能. 比如这就是一个例子:

```rust
let mut generator = || {
    let i = 0;
    loop {
        yield i;
        i += 1;
    }
};
assert_eq!(generator.resume(), GeneratorState::Yielded(0));
assert_eq!(generator.resume(), GeneratorState::Yielded(1));
assert_eq!(generator.resume(), GeneratorState::Yielded(2));
```

Functions 是那种运行一次有一个返回的东西. 而 generator 可以产生多个返回. 它们会暂停运行来 “yield” (返回)一些值, 而后它们可以继续运行到下一次 yield. 虽然我的例子里没有显示, 但它们也可以像普通 function 一样最终停止运行.

在 Rust 当中闭包(closures)算是[装有捕获数据的语法糖, 外加一个某个`Fn` traits 的实现, 以便能够被调用](http://huonw.github.io/blog/2015/05/finding-closure-in-rust/).

Generators 和这差不多, 除了它们也实现了 `Generator` trait [注释4]. 通常generator会保存一个 enum 来表示不同的状态.

[Unstable book](https://doc.rust-lang.org/nightly/unstable-book/language-features/generators.html#generators-as-state-machines) 这本在线书里有些例子, 来展示 generator 状态 enum 是啥样的.

利用 generator, 我们的代码就变得更像我们需要的了! 现在可以这样写了:
```rust
fn foo(...) -> Future<ReturnType, ErrorType> {
    let generator = || {
        let mut future = do_io();
        let data;
        loop {
            // poll the future, yielding each time it fails,
            // but if it succeeds then move on
            match future.poll() {
                Ok(Async::Ready(d)) => { data = d; break },
                Ok(Async::NotReady(d)) => (),
                Err(..) => ...
            };
            yield future.polling_info();
        }
        let result = compute(data);
        // do the same thing for `doMoreIo()`, etc
    }

    futurify(generator)
}
```

这里 `futurify` 是一个函数, 拿一个 generator 作为输入, 返回一个 future, 这个 future 会在每次 `poll()` 调用的时候去调用 generator 的 `resume()`, 而且会一直返回 `NotReady` 直到 generator 结束执行.

可是, 这样的代码不是显得更恶心了吗? 把之前相对干净的 callback-chaining code 搞成这样是要做甚?

但现在你看, 这段代码现在看上去就是“线性”但了. 我们已经把那种回调代码转成了线性但流程, 就像 Go 的代码一样. 然而目前还有着这个奇怪的 loop-yield 啰嗦代码, 和一个 `futurify` 函数.

这就是为什么我们需要引入 [futures-await](https://github.com/alexcrichton/futures-await) 语法了. `futures-await` 是一个过程宏库, 帮你实现打包上述啰嗦代码的最后一点工作. 采用其之后, 代码就可以写成这样了:

```rust
#[async]
fn foo(...) -> Result<ReturnType, ErrorType> {
    let data = await!(do_io());
    let result = compute(data);
    let more_data = await!(do_more_io());
    // ....
}
```

很干净了吧? 几乎和 Go 一样干净了, 只是我们还得显式调用 `await!(...)`. 这类 await 调用基本上提供了那种 Go 代码隐含有的中断点.

哦, 当然了, 因为下面是用 generator 实现的, 你可以加循环, 加分支, 随你怎样写, 就像一般代码一样, 现在看着就干净了.

---
<br>

# 综上所述 - Tying it together

所以, 在 Rust 当中, futures 可以被连在一起来提供一个轻量的栈样的系统. 再加上 async/await, 你可以简洁的编写这种 future 链. `await` 提供了显式的在每个 I/O 操作上的中断点. Tokio 提供了事件循环 - “调度器”抽象, 来管理你提交的 async functions, 而它自己去调用 mio 给抽象出来的底层原始 I/O 阻塞操作.

这些组建都可以被单独使用 - 你可以使用 Tokio 和 futures 而不用  async/await. 你也可以用 async/await 而不用 Tokio - 比如, 我觉得这可能对 Servo 的网络栈有帮助. 它并不需要做很多并行的 I/O (不会到上千线程那种级别), 所以它尽可以用多个操作系统线程. 然而, 我们还想有个线程池和数据管道, async/await 就可以派上用场.

同事采用上述所有的组建, 将使得我们写出几乎和 Go 一样干净的代码. 而且 generators 和 async/await 和 borrow checker 结合的很好 (因为 generators 其实就是一些 enum 状态机), Rust 的安全机制还可以持续发挥作用, 我们因此而有了 “fearless concurrency”, 再也无需畏惧编写有大量 I/O 操作的多线程程序了!

*感谢 Arshia Mufti, Steve Klabnik, Zaki Manian, 和 Kyle Huey 的审阅工作*

---
<br>

索引
1. 值得提醒一下并不是说创建大量线程是完全不可行的方案. 比如 Apache 就采用大量系统线程. 系统线程也经常可以在这种问题上被采用.
2. 轻量级线程也经常被叫做 M:N 线程 (也被称为"绿色线程/green thread").
3. 总体来说, future 运算子并不知晓 tokio 或者 I/O, 所以, 想要问一个运算子“hey, 你在等待哪种 I/O 操作?” 并不是一件容易的事情. 实际上, 用 Tokio 的时候你是在用一种特殊的 I/O primitives, 这种 primitives 不仅提供 futures, 而且也会注册自己到本地线程状态的调度器上. 这样的话, 当一个 future 在等待 I/O 的时候, Tokio 可以检查最近期的 I/O 操作是哪个, 并把这个操作和这个 future 关联, 所以当这个 I/O 操作的 `epoll` 调用告诉 Tokio 本操作完成的时候, Tokio 就可以把这个 future 再次唤醒了. (2018 12月再次编辑: 这种方式已经改变了, futures 现在有一个内建的 `Waker` 概念, 可以用来向栈上传递东西)
4. `Generator` trait 有个 `resume()` 函数, 你可以多次调用, 每次调用会 yield 数据或者告诉你 generator 已经运行结束了.

作者: [Manish Goregaokar](https://manishearth.github.io/blog/2018/01/10/whats-tokio-and-async-io-all-about/).
写于2018-01-10.

---
<br>

### 我的后记

如今已是 2019 年 11 月. Rust 的 Async/Await 语法已经稳定, Tokio 也刚刚发布了 0.2 版本, 一切似乎都在按计划蓬勃发展. 上文最后提到的那个例子, 现在已经可以写成这样了(而且也无需使用 futures-await) :

```rust
async fn foo(...) -> Result<ReturnType, ErrorType> {
    let data = do_io().await;
    let result = compute(data);
    let more_data = do_more_io().await;
    // ....
}
```

怎么样, 是不是看着越来越 cool 了 :)
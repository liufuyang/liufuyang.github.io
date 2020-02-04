---
layout: post
title:  "Calling a tiny C code in Rust"
date:   2020-02-02
comments: true
---

Recently I had been trying to call some C code in Rust, and found it isn't that easy and there seems no good full scale documentation on it. Thus I would like to write a small note here for anyone who is trying the same thing on the first time. 

Perhaps this can help you understand a few basic things more clear. And I have to admit that I am still confused by many of the points around this topic. So don't take my words below as some standard.

My main source of knowledge comes from the following posts or pages:
* Rust embedded book [A little C with your Rust](https://rust-embedded.github.io/book/interoperability/c-with-rust.html)
* Cargo book: [build scripts](https://doc.rust-lang.org/cargo/reference/build-scripts.html)
* A block post - [Calling a C function from Rust](https://blog.jfo.click/calling-a-c-function-from-rust/) (I steal may code examples from it.)

[My source code can be found here](https://github.com/liufuyang/rust-call-c-demo)

And there is another whole field of calling Rust code in C. In this post we don't talk about it yet. Here we only focus on calling some C in Rust.

## Simple C code Running with Rust
So let's start by looking a some simple C code and run
it in Rust code:
```h
// doubler.h
const int FACTOR = 2;

int doubler(int x);
```

```c
// doubler.c
#include "doubler.h"
extern const int FACTOR;

int doubler(int x) {
        return x * FACTOR;
}
```

Just like in the C world, you can compile these files and get a `shared lib` out of it, which can be linked to other piece of code. So let's make a shared lib (sometimes called dynamic lib) now:
```shell
clang doubler.c -c
```
And this will give you a `doubler.o` file.

Now let's try compile it with a piece of Rust code. Like the C header files, now we need to specify a "Rust version of header file" in Rust. For example, 
let's simply make a file manually called `main.rs`
```rust
// main.rs
extern "C" {
    fn doubler(x: i32) -> i32;
}

fn main() {
    unsafe {
        println!("{}", doubler(1));
    }
}
```

Nothing the stuff in `extern "C" {...}` is basically like a header file in Rust, telling the Rust code there is a function like that exist. 

Now we can compile our Rust main function like this:
```shell
rustc main.rs -l doubler.o -L .
```

This will give you a `main` file. `-l` means "link this shared lib", `-L` means "look for lib files in this path".

Then you can run your Rust application with C shared lib:
```
./main
2
```

It works. 

## Make it more like a proper Rust project

But now we just have a few files created on our own. In a real project you will be needing cargo and you would wish cargo can build and run application for you.

Now you can use `cargo new xxx` to firstly create a proper Rust project, then create our code edited above besides the main.rs file. Including the `doubler.o` lib file.

If you do a simple `cargo run`, then it looks like this:
```
$ cargo run
...
  = note: Undefined symbols for architecture x86_64:
            "_doubler", referenced from:
                rust_call_c_demo::main::h62d9537ad28848a0 in rust_call_c_demo-01cdcb27a5928a37.1heji461v07o2wcf.rcgu.o
          ld: symbol(s) not found for architecture x86_64
```

Which means Cargo don't know where to find that shared lib and link it for you.

This can be solved by using [Build Script](https://doc.rust-lang.org/cargo/reference/build-scripts.html). Basically we need to create a `build.rs` file and leave it at the root folder, besides `Cargo.toml` file.
```
// build.rs, in the project root folder
fn main() {
    println!("cargo:rustc-link-search=all=src");      // works like "rustc -L src ..." 
    println!("cargo:rustc-link-lib=dylib=doubler.o"); // works like "rustc -l doubler.o"
}
```

Now cargo run works:
```
$ cargo run
2
```

## Build C code with cargo

However, the above code only works when that `doubler.o` already exist. For a smaller project perhaps it is fine like this. But for a bigger project and people develop code on different platforms, it wouldn't be convenient to just have a `.o` file committed into the repo. You would want it builds while cargo builds.

We can try achieve this via the [cc-rs](https://github.com/alexcrichton/cc-rs) Rust crate:

```toml
# add in cargo.toml
[build-dependencies]
cc = "1.0"
```

```rust
// update build.rs file as:
extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/doubler.c")
        .compile("libdoubler.a");
}
```

Now you can remove that `doubler.o` file, and do cargo run then it will work, as now cargo will build
your shared lib as `libdoubler.a` before making your Rust application:
```
$ cargo run
2
```

## How about some C++ code?

This seems works well with C, can we try out some C++ code? Let's try change the `doubler.c` file to `doubler.cpp`
```c++
#include "doubler.h"
#include <iostream>

extern const int FACTOR;

int doubler(int x) {
    std::cout << "doubler function runs... \n";
    return x * FACTOR;
}
```

Then cargo run gives you an error:
```
Undefined symbols for architecture x86_64:
            "_doubler", referenced from:
```

As I asked in the Rust Discord channel, people told me it is because C++ will doing some kind of renaming of your defined functions (as Rust does as well). 

Then the linking part would become of a problem because of that. So one solution I was told, is to use `extern "C"` in front of your function:

```h
// doubler.h
const int FACTOR = 2;

extern "C" int doubler(int x);
//
```

```c++
// doubler.cpp
#include "doubler.h"
#include <iostream>

extern const int FACTOR;
extern "C" {
    int doubler(int x) {
        std::cout << "doubler function runs... \n";
        return x * FACTOR;
    }
}
```

Also, you would need to tell `cc-rs` now we are building `C++` code like this:
```rust
// build.rs
extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/doubler.cpp")
        .compile("libdoubler.a");
}
```

Now cargo run works again:
```
$  cargo run
doubler function runs...
2
```

## How to easily write "C header" in Rust

Remember earlier we mentioned that, this piece of code
```rust
extern "C" {
    fn doubler(x: i32) -> i32;
}
```
works as a header file in Rust. We manually wrote this earlier. 

If your C/C++ project is a big one, and you wish to automatically generate this Rust code for you from a C/C++ header file, is that possible?

There is a crate called [rust-bindgen](https://github.com/rust-lang/rust-bindgen) can help you with it. 

It seems very powerful, but I got a bit confused while trying to set it up. So basically I tired out it's command line usage only for my simple program here:

```
$ cargo install bindgen
$ bindgen src/doubler.h

/* automatically generated by rust-bindgen */

pub const FACTOR: ::std::os::raw::c_int = 2;
extern "C" {
    pub fn doubler(x: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}
```

(Note that, I am not sure why, while using it, I had to remove the `extern "C"` from the header file, otherwise it won't work for me.)

Now you can replace the code in the `main.rs` and do a cargo run again it will work as well.

I hope this somehow helps. If you encounter any problems, you are welcome to comment below :)
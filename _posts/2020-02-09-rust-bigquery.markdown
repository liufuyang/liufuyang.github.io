---
layout: post
title:  "Calling Rust functions from BigQuery with WASM"
date:   2020-02-09
comments: true
---

We had a hack week at the place I am working, and as a Rust lover I tried to make a small Rust lib that basically do some string encoding so that in the future if we need to work on something more fancy in Rust, we could have this lib at hand to conveniently changing some `gid` such as `dbc3d5ebe344484da3e2448712a02213` to a base 62 encoded `uri` such as `6GGODyP2LIdbxIfYxy5UbN`. 

With some help from some awesome colleagues, writing a lib like this in Rust is pretty easy and fun. 

Then I thought: **If I have this lib in Rust, then can I use it in our BigQuery's UDF via WASM to replace a similar function written in JS, so to speed up things?**

Well, the answer is:
* Yes, you can call Rust code in BigQuery via WASM
* And in my case, it is not faster than JS code

I will show you how to achieve this later in the post. But let me explain why for my case, replacing JS code with the Rust/WASM code didn't help speeding up things.

## Why Rust/WASM not faster than JS code?

While I was trying to get it all work, a nice guy [Pauan](https://github.com/pauan) (who seem to be some core developer on Rust-WASM stuff) gave me some great help. 

Pauan also mentioned that what I was trying to do might not gain speed improvement. Simply citing Pauan's words:

>As for performance, I wouldn't expect Rust to be any faster than JS in this case because it has to do a full copy of the string, do the operations in Rust, and then do a second full copy of the string whereas JS can just use the string directly, without any copying Rust is a lot faster than JS for most things, but not for strings.
>
>Because even though Rust strings are faster than JS strings, it has to do a full copy when converting from a JS string to a Rust string, or a Rust string to a JS string. So that negates the performance benefits


If you gets a bit confused reading this and would like to know more about JS String to Rust WASM, I think you can [take a glance of this post](https://stackoverflow.com/questions/49014610/passing-a-javascript-string-to-a-rust-function-compiled-to-webassembly) 

Anyway, here I show you how you could basically do this. And it might be a valid user case if you have some BigQuery user defined functions, in JavaScript code, that is taking a number in and do some heavy calculation and return a number out. In that case, the String copy cost won't be in your way. So you could potentially gain some speed improvement.

---

## So how to call Rust functions from BigQuery with WASM

Basically I got this idea after reading [Francesc's post on "Calling C functions from BigQuery with Web Assembly"](https://medium.com/@sourcedtech/calling-c-functions-from-bigquery-with-web-assembly-c19c023fc755), and a follow up post from [Felipe Hoffa's post on "BigQuery beyond SQL and JS: Running C and Rust code at scale with wasm"](https://medium.com/@hoffa/bigquery-beyond-sql-and-js-running-c-and-rust-code-at-scale-33021763ee1f)

However they are all a bit out dated, for now the Rust WASM community have come up with many convenient tools to help you achieve things like this.  There are two tools we will use for our Rust WASM code:
* [wasm-bidgen](https://github.com/rustwasm/wasm-bindgen) - something help you write Rust code can be easilly called in JS
* [wasm-pack](https://github.com/rustwasm/wasm-pack) - a cli tool to help you build Rust wasm

### The Rust code - rb62

Let's start with a Rust lib function `rb62::get_b62` - just suppose this is some Rust lib out there that you want to use in WASM, and eventually run on BigQuery. I am using this [`rb62 lib repo`](https://github.com/liufuyang/rb62) as my example.

So with a lib like that, you can write code doing stuff like:
```rust
use rb62;
use std::str;

fn main() {
    let hex = "dbc3d5ebe344484da3e2448712a02213";
    let b62 = rb62::get_b62(hex).unwrap();
    println!("Input hex {}, output b62 {:?}", hex, str::from_utf8(&b62).unwrap());
}
```
And the output is:
```
Input hex dbc3d5ebe344484da3e2448712a02213, output b62 6GGODyP2LIdbxIfYxy5UbN
```

Very simple, String in and String out.

### Prepare a wasm project

Then lets create a new Cargo project to make a WASM that provide this Rust function. Let's call just call it `rb62-wasm`
([Example code repe is here](https://github.com/liufuyang/rb62-wasm))
```
cargo install wasm-pack

cargo new --lib rb62-wasm 
```

Then configure the wasm lib's `Cargo.toml` file as below
```
[package.metadata.wasm-pack.profile.release]
wasm-opt = false

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2.58"
rb62 = { git = 'https://github.com/liufuyang/rb62.git' }
```

Now we can happily write our wasm Rust function `get_b62`:
```rust
use wasm_bindgen::prelude::*;
use rb62;

#[wasm_bindgen]
pub fn get_b62(hex: &str) -> String {
    match rb62::get_b62(hex) {
        Some(b62) => {
            std::str::from_utf8(&b62).unwrap().to_string()
        },
        None => {
            "Not Valid Input       ".to_string()
        },
    }
}
```

Basically it is wrapping a function with tag `#[wasm_bindgen]`. And what `#[wasm_bindgen]` will do is that it will help you generate a bunch of helper code so that a String can be memory copied from the JS vm to WASM's linear memory. Without it, you would have to write lots of code to do this manually, [as this outdated post have described](https://stackoverflow.com/questions/49014610/passing-a-javascript-string-to-a-rust-function-compiled-to-webassembly). (Even though it is outdated, I suggest you still read about it to understand what is happening under the hood)

Then after we have the code, we can now build our Rust wasm package very simply like this:
```
wasm-pack build --release --target no-modules
```
which will generate a bunch of files under a folder called `pkg`
```
ls pkg

rb62_wasm.js rb62_wasm_bg.wasm ...
```

Those are the wasm files that has the code of our Rust function.

Then, in order to use these conveniently in a Javascript environment, [Pauan](https://github.com/pauan) has helped making me a JS script to basically combine those to files to a single JS file to be used.

---

## Prepare a glue JS script and run it

The script he made is this one `generate.js`
```js
const fs = require('fs');

const glue = fs.readFileSync("./pkg/rb62_wasm.js", { encoding: "utf8" });
const buffer = fs.readFileSync("./pkg/rb62_wasm_bg.wasm");

const bytes = Array.from(new Uint8Array(buffer.buffer));

fs.writeFileSync("base62.js", `\
${glue}
self.wasm = wasm_bindgen(new Uint8Array(${JSON.stringify(bytes)}));
`);
```

Simply running it:
```
node generate.js
```
which will create the [`base62.js`](https://github.com/liufuyang/rb62-wasm/blob/master/base62.js) - a **golden single file** include the JS helper code and the wasm byte code in it!

If you take a glance of it, you would spot all the helper code that is more or less the same with the manually written code in the outdated Stackoverflow question I linked above.

Now, before try out this code, you need to a manual thing:
```
In the generated base62.js file, manually replace the word `self` as `this`
```

This has to be done because in BigQuery JS environment it doesn't have the word `self`. Also, as the webpack generated code are not meant to be used on places like BigQuery, it means what we are doing here is a very special and even a bit hacky-ish thing. So to make it work, for now, you just need to manually replace the 2 `self` as `this` in `base62.js`.

And that's it.

Now you can try it out locally with some simple JS code:
```js
// test.js
const RB62 = require('./base62.js')
RB62.wasm.then(() => {
    console.log(wasm_bindgen.get_integer("dbc3d5ebe344484da3e2448712a02213"))
});
```
Running it to verify:
```
node test.js

6GGODyP2LIdbxIfYxy5UbN
```

It works! (Or you can even play it out in your browser from here [https://liufuyang.github.io/rb62-wasm/](https://liufuyang.github.io/rb62-wasm/))

---

## Use this on BigQuery

Now we can use this on BigQuery, firstly, upload this code 
onto some GCS place, in my case I put it at `"gs://liufuyang/public/rb62-wasm/base62.js"`

Then you can run query like:
```sql
CREATE TEMP FUNCTION `hex_to_b62`(hex STRING) RETURNS STRING LANGUAGE js AS '''
 
return wasm.then(() => {
        return wasm_bindgen.get_b62(hex);
    });
'''
OPTIONS (
  library=[
    "gs://fh-bigquery/js/inexorabletash.encoding.js",
    "gs://liufuyang/public/rb62-wasm/base62.js"
  ]
);


SELECT hex_to_b62(hex) b62
FROM (
  select 'dbc3d5ebe344484da3e2448712a02213' as hex
  union all
  select 'ffffffffffffffffffffffffffffffff' as hex
)
```

And the output looks like this:
![BigQuery](/assets/2020-02-09-rust-bigquery/bq_1.png)*Rust code running in BigQuery UDF*

Note that `"gs://fh-bigquery/js/inexorabletash.encoding.js"` is needed as it provides us the needed `TextEncoder` and `TextDecoder` used in `base62.js`, see [this stackoverflow answer from Felipe Hoffa](https://stackoverflow.com/questions/60094731/can-i-use-textencoder-in-bigquery-js-udf)

### Grouping input to speed things up
As you may have already noticed or not, if each row of the select query is calling once this UDF, then each time a WASM runtime is initiated. And it takes time to do that.

So in practice, you really want to group some input together and define an UDF that takes in an array of Strings and return an array of Strings. Basically it is the idea I borrowed from [Felipe Hoffa's post](https://medium.com/@hoffa/bigquery-beyond-sql-and-js-running-c-and-rust-code-at-scale-33021763ee1f).

An example would be like this:
```sql
CREATE TEMP FUNCTION `hex_to_b62`(hex ARRAY<STRING>) RETURNS ARRAY<STRING>
  LANGUAGE js
  OPTIONS (
    library=[
        "gs://fh-bigquery/js/inexorabletash.encoding.js",
        "gs://liufuyang/public/rb62-wasm/base62.js"
    ]
  )
  AS '''
    return wasm.then(() => {
        return hex.map((val) => {
            return wasm_bindgen.get_b62(val);
        });
    });
  ''';

SELECT b62
FROM (
SELECT FLOOR(RAND()*10) grp, hex_to_b62(ARRAY_AGG(gid)) b62_array
FROM(
    SELECT gid FROM `some-table`
    LIMIT 1000000
  )
GROUP BY grp
) , UNNEST(b62_array) b62

```

### A comment on speed:

As mentioned in the beginning, this is not really helping speed things up as firstly there is time needed to load wasm runtime, and mostly importantly it just cost too much to copy JS string to Rust wasm string and back-forth...

 For some specific queries , the wasm version is around 23-27 seconds to finish, while directly use the js library it uses 10-13 seconds, both with the grouping idea mentioned above. Without grouping it is much much slower for the wasm version.

I hope this post helps anyone trying out similar or related stuff out there.

---
# UPDATE 2020-02-15 - No Grouping is really needed

As Felipe mentioned to me and you can [see some discussion on this StackOverflow page](https://stackoverflow.com/questions/59430104/bigquery-javascript-udf-process-per-row-or-per-processing-node/), this means that, in the BigQuery UDF JavaScript function you defined:
```js
return wasm.then(() => { ...
```
that `wasm` is actually not loaded again and again for each row. The same process was reused multiple times, and variables such as `wasm` were kept around in between calls. Though this is not officially documented anywhere yet, according to Felipe's answer.

Then this means we can just used our WASM function as if it is a JS function directly and let's run it again and do some speed comparison on some fairly large amount of data:

This one is for `b62_to_hex` comparison:
```sql
CREATE TEMP FUNCTION `b62_to_hex`(b62 STRING) RETURNS STRING
  LANGUAGE js
  OPTIONS (
        library=[
        "gs://fh-bigquery/js/inexorabletash.encoding.js",
        "gs://liufuyang/public/rb62-wasm/base62.js",
        "gs://bq-udfs/latest/base62.js" -- Internal JS base62 libarary, providing Base62.toHex function
    ]
  )
  AS '''
    return wasm.then(() => wasm_bindgen.get_integer(b62));  // 26.8 sec, total 138328004 values
    // return Base62.toHex(b62);                            // 20.2 sec, total 138328004 values
  ''';

SELECT b62_to_hex(b62_array[ORDINAL(ARRAY_LENGTH(b62_array))]) hex
FROM(
    SELECT  SPLIT(uri, ':') b62_array
    FROM `spotify-entities.track.20200205`
    # LIMIT 1000000
  )
```

And this one for `hex_to_b62` comparison:
```sql
CREATE TEMP FUNCTION `hex_to_b62`(hex STRING) RETURNS STRING
  LANGUAGE js
  OPTIONS (
    library=[
        "gs://fh-bigquery/js/inexorabletash.encoding.js",
        "gs://liufuyang/public/rb62-wasm/base62.js",
        "gs://bq-udfs/latest/base62.js" -- Internal JS base62 libarary - providing Base62.fromHex function
    ]
  )
  AS '''
    return wasm.then(() => wasm_bindgen.get_b62(hex));  // 22.7 sec total of 138328003 element
    // return Base62.fromHex(hex);                      // 15.4 sec total of 138328003 element
  ''';

 SELECT hex_to_b62(track.gid) AS hex 
 FROM `spotify-entities.entities.entities20200205` 
 WHERE track.gid IS NOT NULL
 # LIMIT 1000000
```

## Performance result comparison on direct calling (no grouping)

|function|JavaScript|Rust/WASM|
|--|--|--|
|**b62_to_hex**| **20.2** sec| 26.8 sec|
|**hex_to_b62**| **15.4** sec| 22.7 sec|

Tested each function running once on a data set having 138,328,003 rows.

I would conclude that the result is not much different than the previous test with grouping. But indeed the SQL code can be much simpler now.

It is still a bottle neck of the String copies from JS to WASM. Looking forward to someday on BigQuery you can run native code, perhaps via WASM or WASI, then that would be pretty cool and Rust will be very helpful in that case.

---

And special thanks to [Pauan](https://github.com/pauan) and [Felipe Hoffa](https://medium.com/@hoffa) for the great help and support on me trying these random stuff out :)
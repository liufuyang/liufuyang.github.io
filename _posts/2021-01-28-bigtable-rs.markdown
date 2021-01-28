---
layout: post
title:  "bigtable_rs - A Bigtable client lib in Rust"
date:   2021-01-28
comments: true
---

I am happy to announce that I published 
a quite simple Rust library **[bigtable_rs](https://crates.io/crates/bigtable_rs)** - a client 
for talking with Google Bigtable service.

Repo is [here](https://github.com/liufuyang/bigtable_rs), or if you wish to take a look at [the doc](https://docs.rs/bigtable_rs/0.1.3/bigtable_rs/)

With it, you can assemble read or write requests based on [Google Bigtable V2 protobuf schema](https://github.com/googleapis/googleapis/blob/master/google/bigtable/v2/bigtable.proto) and send the requests via the client, which is build on top of tonic gRPC over HTTP/2.

Google service account key authentication is also provided.

A simple read example could be like this,
with filters such as family name filter, 
qualifier filter and so on.

```rust
use bigtable_rs::bigtable;
use bigtable_rs::google::bigtable::v2::row_filter::{Chain, Filter};
use bigtable_rs::google::bigtable::v2::row_range::{EndKey, StartKey};
use bigtable_rs::google::bigtable::v2::{ReadRowsRequest, RowFilter, RowRange, RowSet};
use env_logger;
use std::error::Error;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let project_id = "project-id";
    let instance_name = "instance-1";
    let table_name = "table-1";
    let channel_size = 4;
    let timeout = Duration::from_secs(10);

    let key_start: String = "key1".to_owned();
    let key_end: String = "key4".to_owned();

    // make a bigtable client
    let connection = bigtable::BigTableConnection::new(
        project_id,
        instance_name,
        true,
        channel_size,
        Some(timeout),
    )
        .await?;
    let mut bigtable = connection.client();

    // prepare a ReadRowsRequest
    let request = ReadRowsRequest {
        app_profile_id: "default".to_owned(),
        table_name: bigtable.get_full_table_name(table_name),
        rows_limit: 10,
        rows: Some(RowSet {
            row_keys: vec![], // use this field to put keys for reading specific rows
            row_ranges: vec![RowRange {
                start_key: Some(StartKey::StartKeyClosed(key_start.into_bytes())),
                end_key: Some(EndKey::EndKeyOpen(key_end.into_bytes())),
            }],
        }),
        filter: Some(RowFilter {
            filter: Some(Filter::Chain(Chain {
                filters: vec![
                    RowFilter {
                        filter: Some(Filter::FamilyNameRegexFilter("cf1".to_owned())),
                    },
                    RowFilter {
                        filter: Some(Filter::ColumnQualifierRegexFilter("c1".as_bytes().to_vec())),
                    },
                    RowFilter {
                        filter: Some(Filter::CellsPerColumnLimitFilter(1)),
                    },
                ],
            })),
        }),
        ..ReadRowsRequest::default()
    };

    // calling bigtable API to get results
    let response = bigtable.read_rows(request).await?;

    // simply print results for example usage
    response.into_iter().for_each(|(key, data)| {
        println!("------------\n{}", String::from_utf8(key.clone()).unwrap());
        data.into_iter().for_each(|row_cell| {
            println!(
                "    [{}:{}] \"{}\" @ {}",
                row_cell.family_name,
                String::from_utf8(row_cell.qualifier).unwrap(),
                String::from_utf8(row_cell.value).unwrap(),
                row_cell.timestamp_micros
            )
        })
    });

    Ok(())
}
```

You can easily play with it together with the Bigtable Emulator.
See the [repo](https://github.com/liufuyang/bigtable_rs) for details.

There is [another example](https://github.com/liufuyang/bigtable_rs/tree/main/examples/src/http_server) showing you how to use this client in a Http (or any other type of)
server like environment. Enjoy it.

Hopefully someone might found this crate helpful? Any help to improve on it is appreciated :)

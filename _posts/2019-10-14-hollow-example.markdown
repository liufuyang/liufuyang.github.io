---
layout: post
title:  "Hollow Example (with Ktor)"
date:   2019-10-14
comments: true
---


# Step 1 - Make a simple Ktor service

With https://ktor.io/quickstart/generator.html# you can simply generate a Ktor project.

Otherwise you are welcome to use the code I have for this blog post at 
https://github.com/liufuyang/hollow-example-ktor

Firstly when you create a Ktor project, in the main file you will see something like these:
```kotlin
    routing {
        get("/") {
            call.respondText("HELLO WORLD!", contentType = ContentType.Text.Plain)
        }

        get("/json/gson") {
            call.respond(mapOf("hello" to "world"))
        }
    }
```

And know you can start the server and do query on it via curl or browser, you can get text return or a json return.

## Hold an object in memory

```kotlin
package com.example

import com.netflix.hollow.core.write.objectmapper.HollowPrimaryKey

@HollowPrimaryKey(fields=["id"])
data class InfoEntity(
    val id: Int,
    val name: String,
    val status: String,
    val timeUpdated: Long
)
```

```kotlin
...
fun Application.module(testing: Boolean = false) {
    install(ContentNegotiation) {
        gson {
        }
    }

    val entityList = listOf(
        InfoEntity(1, "News 1", "NEW", 0),
        InfoEntity(2, "News 2", "NEW", 1),
        InfoEntity(3, "News 3", "NEW", 2)
    ).toMutableList()

    routing {

        get("/info") {
            call.respond(entityList)
        }

        post("/info") {
            val newInfo = call.receive<InfoEntity>()
            println(newInfo)
            entityList.add(newInfo)
            call.respond(HttpStatusCode.Created)
        }
    }
}
```

```
curl localhost:8080/info

[
  {
    "id": 1,
    "name": "News 1",
    "status": "NEW",
    "timeUpdated": 0
  },
  ...
]
```

```
curl -i  -X POST localhost:8080/info -H "Content-Type: application/json" -d '{"id":4, "name":"A new item", "status":"NEW", "timeUpdated":3}'

curl localhost:8080/info

[
  {
    "id": 1,
    "name": "News 1",
    "status": "NEW",
    "timeUpdated": 0
  },
  ...
  {
    "id": 4,
    "name": "A new item",
    "status": "NEW",
    "timeUpdated": 3
  }
]
```

# Step 2 - Make service as a Hollow Producer

```kotlin
// producer code
...
fun Application.module(testing: Boolean = false) {
    install(ContentNegotiation) {
        gson {
        }
    }

    val entityList = listOf(
        InfoEntity(1, "News 1", "NEW", 0),
        InfoEntity(2, "News 2", "NEW", 1),
        InfoEntity(3, "News 3", "NEW", 2)
    ).toMutableList()

    /** Hollow setup **/
    val localPublishDir = File("target/publish")

    val publisher = HollowFilesystemPublisher(localPublishDir.toPath())
    val announcer = HollowFilesystemAnnouncer(localPublishDir.toPath())

    val producer = HollowProducer
        .withPublisher(publisher)
        .withAnnouncer(announcer)
        .buildIncremental()

    producer.runIncrementalCycle { state ->
        for (e in entityList)
            state.addIfAbsent(e)
    }

    routing {
        get("/info") {
            call.respond(entityList)
        }

        post("/info") {
            val newInfo = call.receive<InfoEntity>()
            println(newInfo)
            entityList.add(newInfo)

            producer.runIncrementalCycle { state ->
                state.addIfAbsent(newInfo)
            }
            call.respond(HttpStatusCode.Created)
        }
    }
}
```

# Step 3 - Create a consumer that reconstruct entities in memory

## Generate Hollow source code (Hollow API) for consumer to use

Before we move on to build our consumer, we first need to let Hollow generate
some source code (the so-called Hollow API) that can be used for our consumer later.

It is very simple to generate these code, simply make a main function like the code
below and run it directly via ide, then the API code will be generated:
```kotlin
// GenerateHollowSource.kt
import com.netflix.hollow.api.codegen.HollowAPIGenerator
import com.netflix.hollow.core.write.objectmapper.HollowObjectMapper
import com.netflix.hollow.core.write.HollowWriteStateEngine

fun main() {
    val writeEngine = HollowWriteStateEngine()
    val mapper = HollowObjectMapper(writeEngine)
    mapper.initializeTypeState(InfoEntity::class.java)

    val generator = HollowAPIGenerator.Builder().withAPIClassname("InfoEntityAPI")
        .withPackageName("com.example.hollow.generated")
        .withDataModel(writeEngine)
        .build()

    generator.generateFiles("../hollow-ktor-consumer/src/")
}
```

The only thing worth noticing is that the generatedFiles is set to `../hollow-ktor-consumer/src/`, 
as my producer and consumer project is side by side so I can generate the 
code directly into the consumer project, saving me some time to copy 
files around.

After running the main function above, you should see those Java files 
are generated in consumer project:

![Generated code](/assets/2019-10-14-hollow-example/generated_code.png)

Another super good thing of Kotlin I have to mention is that you can use 
Kotlin together with Java code in the same project. So even Hollow generated 
only Java files, we don't have to do anything else, we can directly use
them in our Kotlin consumer project :)

## Writing consumer code

```kotlin
// consumer code
...
fun Application.module(testing: Boolean = false) {
    install(ContentNegotiation) {
        gson {
            registerTypeAdapter(InfoEntity::class.java, InfoEntityAdapter())
        }
    }

    /** Hollow setup **/
    val localPublishDir = File("../hollow-ktor-producer/target/publish")

    val blobRetriever = HollowFilesystemBlobRetriever(localPublishDir.toPath())
    val announcementWatcher = HollowFilesystemAnnouncementWatcher(localPublishDir.toPath())

    val consumer = HollowConsumer.withBlobRetriever(blobRetriever)
        .withAnnouncementWatcher(announcementWatcher)
        .withGeneratedAPIClass(InfoEntityAPI::class.java)
        .build()

    consumer.triggerRefresh()

    // Setting up an id index
    val uniqueIndex = InfoEntity.uniqueIndex(consumer)
    consumer.addRefreshListener(uniqueIndex)

    // Setting up an Hash Index

    /** End of Hollow setup **/

    routing {

        get("/info") {
            val infoEntityAPI = consumer.getAPI() as InfoEntityAPI
            call.respond(infoEntityAPI.getAllInfoEntity().stream().collect(Collectors.toList()))
        }

        get("/info/{id}") {
            val id = call.parameters["id"]?.toInt() ?: throw IllegalStateException("Must provide id");
            val infoEntity = uniqueIndex.findMatch(id)
            if (infoEntity != null) {
                return@get call.respond(infoEntity)
            } else {
                return@get call.respond(HttpStatusCode.NoContent)
            }
        }
    }
}
```

You would also need a `InfoEntityAdapter` on the consumer side to help gson turn the
Hollow generated `InfoEntity` into Json format.

```kotlin
import com.example.hollow.generated.InfoEntity
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonSerializationContext
import com.google.gson.JsonSerializer
import java.lang.reflect.Type


class InfoEntityAdapter : JsonSerializer<InfoEntity> {

    override fun serialize(
        src: InfoEntity, typeOfSrc: Type,
        context: JsonSerializationContext
    ): JsonElement {

        val obj = JsonObject()
        obj.addProperty("id", src.id)
        obj.addProperty("name", src.name.value)
        obj.addProperty("status", src.status.value)
        obj.addProperty("timeUpdated", src.timeUpdated)

        return obj
    }
}
```
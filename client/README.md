#### The client

This is just a website, your can use any web server, just serve all the content under **client/web**.

If you use windows, click on **client/client.exe**. It's a single executable that packages everything.

For Linux and Mac, or other Unix, the server can be built with:

```
$ cd adversarial-driving/model
$ go install github.com/gobuffalo/packr/v2@v2.8.3
$ go build
$ ./client
```

The web page is available at: http://localhost:3333/

<img src="../doc/client.png"  width="100%"/>

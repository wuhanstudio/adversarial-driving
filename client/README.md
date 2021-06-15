#### 3. Setup the browser

This is just a website, your can use any web server, just serve all the content under **client/web**.

If you use windows, click on **client/client.exe**. It's a single executable that packages everything.

For Linux and Mac, or other Unix, the server can be built with:

```
go get -u github.com/gobuffalo/packr/packr
packr build
```


The web page will be available at: http://localhost:3333/

<img src="../doc/client.png"  width="100%"/>

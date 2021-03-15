package main

import (
  "net/http"
  "fmt"
  "github.com/gobuffalo/packr"
)

func main() {
  box := packr.NewBox("./web")

  http.Handle("/", http.FileServer(box))
  fmt.Println("Listening on port 3333")
  http.ListenAndServe(":3333", nil)
}

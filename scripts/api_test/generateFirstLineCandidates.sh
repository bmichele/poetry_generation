#!/usr/bin/env bash

URL=http://0.0.0.0
PORT=8000

curl -v -H "key: supersecret" -H "Content-Type: application/json" -X POST "${URL}:${PORT}/generator/first_line" \
  -d '{
        "language": "en",
        "style": "",
        "keywords": "table sun"
      }'

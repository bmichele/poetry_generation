#!/usr/bin/env bash

URL=http://0.0.0.0
PORT=8000

curl -v -H "key: supersecret" -H "Content-Type: application/json" -X POST "${URL}:${PORT}/generator/next_line" \
  -d '{
        "language": "en",
        "style": "",
        "poem_state":["The test and the example of mankind,","To guard the virtues of their noblest minds,"]
      }'

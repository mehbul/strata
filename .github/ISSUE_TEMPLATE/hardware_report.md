---
name: Hardware report
about: Tell us what Strata does on your card
title: '[hardware] '
labels: ''
assignees: ''
---

Strata is developed and measured on one machine. Reports from other cards are
genuinely useful, because hardware detection is meant to work on anything
llama.cpp's ROCm build enumerates.

## Your machine

```
GPU:
VRAM:
RAM:
CPU:
OS:
```

## `strata info`

```
paste here
```

## Does the runtime see your GPU?

```powershell
.\runtime\llama-server.exe --list-devices
```

```
paste here
```

## What you ran, and what you got

<!-- Model, quantisation, context, and the tok/s you measured if you have it.
     `strata tune --model <m> --save` produces a table worth pasting. -->

## Anything surprising

<!-- Including "it just worked", which is also useful to know. -->

---
name: Bug report
about: Something is not working
title: ''
labels: bug
assignees: ''
---

## What happened

<!-- What you expected, and what happened instead. -->

## How to reproduce

```powershell
# the exact command you ran
```

## Your machine

<!-- `strata info` output is ideal. At minimum: GPU, VRAM, RAM, OS. -->

```
GPU:
VRAM:
RAM:
OS:
```

## Does the runtime see your GPU?

```powershell
.\runtime\llama-server.exe --list-devices
```

<!-- Paste the output. An empty list means a backend library is missing, which
     makes everything fall back to CPU with nothing in the log. -->

## Model

<!-- File name, quantisation, and roughly how large. -->

## Anything else

<!-- Logs, screenshots, the console's Inspector values. -->

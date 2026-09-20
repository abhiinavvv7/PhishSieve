# PhishSieve

PhishSieve is a web app that checks whether a URL looks safe or suspicious using a machine learning model.

It uses URL-based features to detect phishing patterns, makes a prediction with a trained Random Forest model, and shows a simple result with clear guidance for the user.

## Overview

This project analyzes a submitted URL, extracts useful signals such as URL length, `@` usage, suspicious host patterns, and other phishing indicators, then classifies it as:

- Safe
- Phishing

The interface is lightweight, simple, and designed for quick checks without exposing unnecessary technical detail.

## Features

- URL-based phishing detection
- Random Forest model for live predictions
- Simple browser UI
- Light and dark theme support
- Clear user-facing result and guidance
- Local Python backend with no heavy framework setup

## Project structure

```
.
├── app.py
├── index.html
├── phishing.csv
├── requirements.txt
├── README.md
├── docs/
├── images/
└── Report and PPT/
```

## Getting started

### Prerequisites

- Python 3.9+

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Then open:

```text
http://localhost:8000/
```

## How it works

1. Enter a URL into the app.
2. The backend extracts phishing-related features from the URL.
3. The trained model predicts whether the URL is safe or suspicious.
4. The UI shows the result and next-step advice.

## Notes

This project is intended for local use and demo purposes. It is a personal project and is not tied to any original external owner or repository attribution.

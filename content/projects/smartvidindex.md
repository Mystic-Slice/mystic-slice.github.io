---
title: "Smart Video Index"
description: "Finds the most relevant moments in a video collection for your question."
descriptionOnCover: true
cover:
    image: "images/smartvidindex/cover.png"
tags: ["Project", "Python", "Langchain", "Qdrant", "OpenAI", "Ollama", "Webapp"]
draft: false
comments: true
math: true
---

## Overview
This app helps users find the most relevant moments in a video collection by analyzing video transcripts. It breaks videos into segments and uses an advanced retrieval system to match user questions with the most suitable video segments, delivering precise answers.

When you ask a question, the app searches the video summaries and segments, then shows the most relevant video parts and answers. It's perfect for targeted searches, like finding specific explanations in educational videos.

For the project code and more info, visit [Smart Video Index](https://github.com/Mystic-Slice/smart-vid-index).

## Demo
### Want to see it in action?
Check out the demo video below to see how the app quickly finds relevant video moments based on user questions.

<video width=100% controls>
    <source src="/vids/smartvidindex/Demo.mp4" type="video/mp4">
    Your browser does not support the video tag.  
</video>

## Examples
Here are some more examples of the app being used to answer questions in real-time from a video collection.

<video width=100% controls>
    <source src="/vids/smartvidindex/Examples.mp4" type="video/mp4">
    Your browser does not support the video tag.  
</video>

**Tech Used:** Langchain, OpenAI API, Qdrant, Flask, Next.js, and shadcn.
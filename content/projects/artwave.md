---
title: "ArtWave - Find the missing piece"
summary: "This blog explores ArtWave, an AI-powered matchmaking app designed to connect artists across diverse fields by analyzing the emotional and thematic essence of their work."
description: "Discover artists and works that resonate most closely with yours"
descriptionOnCover: true
cover:
    image: "images/artwave/logo.png"
tags: ["Project", "Python", "LlamaIndex", "Qdrant", "OpenAI", "Webapp", "GenAI", "Flask"]
date: 2024-11-11T00:00:00
draft: false
comments: true
math: true
showtoc: true
---

## Introduction 

ArtWave was created at [**HackSC**](https://socal-tech-week-2024.devpost.com/), the flagship hackathon of **SoCal Tech Week 2024**. This hackathon presented an exciting opportunity to explore solutions for creative collaboration, and our team was inspired to address a common problem: artistic collaboration can be transformative, yet finding the right collaborator—especially across different art forms—often feels like a daunting challenge. While social media platforms connect people broadly, they rarely cater to the unique needs of artists seeking partners in new mediums. ArtWave fills this gap, providing a dedicated space for artists across disciplines to connect, collaborate, and bring diverse creative visions to life.

Our inspiration came from seeing how artists often work in silos, limited by access to others who share similar artistic vibes or complementary skills. ArtWave aims to break these barriers, helping artists transcend their creative limits by matching them with like-minded collaborators whose work resonates on an emotional and thematic level. Whether you're a filmmaker looking for a musician, a poet seeking a visual artist, or any creator wanting to cross into new fields, ArtWave connects you with the ideal partner to help realize your vision.

## What It Does

ArtWave is an AI-powered app that intelligently matches you with like-minded creatives with similar artistic styles across different artistic fields.  Imagine you're a filmmaker with a romantic story - you have the perfect script and poster, but need music that captures your vision's emotional essence. Simply upload your movie poster or story synopsis to ArtWave, and our AI analyzes its emotional elements - from tender hope to gentle melancholy. The system then matches you with musicians whose compositions evoke similar emotions, providing a curated list of potential collaborators (and their works) whose style naturally aligns with your vision. No matter your medium, our app helps you find that perfect spark of synergy to bring your vision to life. Currently the app supports matching across visual media (image) like paintings, movie posters and audio media like music. But the underlying framework can easily be extended to other art forms like poetry, dance, and more.

## Demo
<video width=100% controls>
    <source src="/vids/artwave/Demo.mp4" type="video/mp4">
    Your browser does not support the video tag.  
</video>

## Technical Details

### Artist Matching System
The biggest challenge in this idea is capturing the emotional essence of an artwork. But once this is achieved, matching different people across different artistic modalities is a relatively straightforward task. Comparing artworks of different modalities requires a unified representation that can capture the non-verbal elements like emotions, themes and styles. This representation has to be malleable enough to capture the essence of different art forms, yet robust enough to allow for meaningful comparisons using traditional search algorithms. I chose to go with text as the medium of representation.

### Artwork to Unified Representation
I built a three stage pipeline to convert different artworks into the unified representation (i.e. text). 
1. **Convert to text**: Converting non-verbal media like paintings and music to text is a non-trivial task. For the MVP, I utilized OpenAI's [Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) model to transcribe the lyrics of the song from its audio. For paintings and other visual artworks, I used Meta's [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct), a visual question answering model to describe the artwork in words.
2. **Detail Description Generation**: The initial text obtained from the previous text is objective and doesn't contain any information about the "meaning" behind the artwork. To capture the non-obvious details, which usually are the more important aspect that distinguishes artists, I used GPT-4o to generate a detailed description of the artwork. I prompted the model to guess the artist's intent behind the artwork, the emotions it evokes, and the themes it explores.
3. **Conversion to Modality-agnostic Unified Representation**: The detailed description generated in the previous step is then fed to GPT-4o to summarize and specifically remove any modality-specific information. This step ensures that the representation is modality-agnostic and only captures the essence of the artwork. For example, while describing an artwork, the model might mention colours and their meanings but these are not relevant for music. This step ensures that such information is removed while the emotions behind the colours are retained.
4. **Tag Generation**: The unified textual representation of the artwork is then tagged with the emotions, themes and styles it captures. This is done again using GPT-4o. The tags are stored as metadata associated with each artwork in a vector database and then are used to match the artwork with other artworks.

The unified textual representation of the artwork is then compared with the representations of other artworks to find the most similar ones. The similarity is calculated using cosine similarity on a vector database and metadata filtering.

**Note:** The multi-step approach is required to ensure that the accuracy of the generation by GPT-4o is high. When it comes to LLMs, the more task you burden them with in a single step, the more likely they are to make mistakes. By breaking down the task into smaller steps, we ensure that the model is not overwhelmed and can focus on the specific task at hand.

### Other Technical Details

I built the frontend of the application using Next.js (React, Tailwind, shadcn). I built the AI workflow using LlamaIndex, Qdrant, OpenAI API, Huggingface and Kindo AI API (to access GPT-4o). The backend which connects the frontend to the AI workflow was built by my teammate Aditya Sharma, using Flask, Firebase and MongoDb. 

## Future Extension
I could think of a few ways the app could be extended in the future:
1. **More Art Forms**: The app currently supports matching across visual and audio media (for the sake of time constraint). The underlying framework can be extended to other art forms like poetry, dance, and more.
2. **Improving Accuracy of Unified Representation**: For example, the current approach I took to describe a song is just to use the lyrics of the song. But I want to expand this to include information about the melody, the rhythm, the instruments used, etc so that all music, including the ones without lyrics, are understood and represented better by the system. This would require a more sophisticated model that can understand music. 
3. **Incorporating the Artists' own interpretation**: While an advanced model could be better at describing music, it will never be able to capture the exact intent and emotion of the artist that went into their work. I want to incorporate a feature where the artist can describe their work in their own words and this description would be combined with the model's description to get a more accurate unified representation.
4. **Community-building features**: The app could be extended to include features that help artists collaborate better. For example, a chat feature, a shared workspace, user rating-based reputation system, etc.

## Code
The hackathon project page is: [ArtWave](https://devpost.com/software/artwave).

The code for the entire application can be found in these repositories:
- [Frontend](https://github.com/Mystic-Slice/artwave-frontend)
- [Backend](https://github.com/Mystic-Slice/artwave-backend)

I had a lot of fun working on this project with my team [Aditya Sharma](https://www.linkedin.com/in/adityasharma98/), [Prathmesh Lonkar](https://www.linkedin.com/in/prathmesh-lonkar/) and [Pratik Dilip Dhende](https://www.linkedin.com/in/pratik-dhende/). This project is one of the most impactful uses of generative AI I've developed so far. It was amazing to see how far LLMs could (or maybe pretend to) understand and describe human art. I was also pleasantly surprised by how quickly we brought this project together. The entire application came together in less than 24 hours.

<div style="display: block; width: 100%; justify-items: center;">
<img src="/images/artwave/team_pic.jpg" style="display: block;margin: auto;"/><em>My team at the hackthon</em>
</div>

While I built this project quickly for the hackathon, I believe that the underlying idea has a lot of potential and I am excited to see where it goes in the future. If you believe in this project's potential and would like to collaborate on its development, please reach out! I'd also love to hear any ideas you might have for enhancing ArtWave.
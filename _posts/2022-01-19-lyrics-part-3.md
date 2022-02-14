---
layout: post
date: 2022-01-21
title: "Analysis of Heavy Metal Lyrics - Part 3: Word Clouds"
categories: jekyll update
permalink: /projects/heavy-metal-analysis/lyrics-part-3
summary: |
  Word cloud graphics for the most popular genres and bands.
---

<pre style="margin-left: 50px; margin-right: 50px; font-size: 13px">
Explicit/NSFW content warning: this project features examples of heavy metal lyrics and song/album/band names.
These often contain words and themes that some may find offensive/inappropriate.
</pre>

This article is the third part of the lyrical analysis [heavy metal lyrics](/projects/heavy-metal-analysis.html).
If you're interested in seeing the full code (a lot is omitted here), check out the
[original notebook](https://github.com/pdqnguyen/metallyrics/blob/main/analyses/lyrics/notebooks/lyrics-part-3-word-clouds.ipynb).
In the [next article](./lyrics-part-4.html) we'll prototype machine learning models for lyric-based genre classification.

Word clouds are a fun and oftentimes helpful technique for visualizing natural language data.
They can show words scaled by any metric, although term frequency (TF) and
term-frequency-inverse-document-frequency
([TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting))
are the most common metrics.
For a multi-class or multi-label classification problem,
word clouds can highlight the similarities and differences between separate classes by treating each class as its own document to compare with all others.
The word clouds seen here were made with the `WordCloud` generator by
[amueller](https://github.com/amueller/word_cloud),
with pre-processing done via `gensim` and `nltk`.

## Genre word clouds

First I split the full dataframe by genre, so each document consists of all the lyrics for that genre.
Since bands can belong to multiple genres, there's a lot of overlap in the vocabularies of some commonly-associated genres,
so some words appear in a few different word clouds.

The TF-IDF word clouds do quite well at picking out the words that are unique to a genre:
black metal lyrics deal with topics like the occult, religion, and nature;
death metal focuses on the obscene and horrifying;
heavy metal revolves around themes more familiar to rock and pop;
power metal adopts the vocabulary of fantasies and histories;
and thrash metal sings of violence, war, and... beer?
The full corpus word cloud shows themes common to all heavy metal genres.

![genre_clouds](/assets/images/heavy-metal-lyrics/wordclouds/genres.png)

## Band word clouds

Here are word clouds for the ten most-reviewed bands in on Metal-Archives.
If you'd like to produce a word cloud for a particular artist (among the top-200 most reviewed),
you can do so by heading over to the [lyrics data dashboard](https://metal-lyrics-feature-plots.herokuapp.com/){:target="_blank"})
mentioned in the previous articles and clicking on any of the artists shown.
The TF-IDF parameters I implemented there are a little different, so the results may vary.

![band_clouds](/assets/images/heavy-metal-lyrics/wordclouds/bands.png)

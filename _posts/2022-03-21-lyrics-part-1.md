---
layout: post
date: 2022-03-21
title: "Analysis of Heavy Metal Lyrics - Part 1: Overview"
categories: jekyll update
permalink: /projects/heavy-metal-analysis/lyrics-part-1
summary: |
  A quantitative overview of the vocabulary of heavy metal lyrics extracted from DarkLyrics.
---

<pre style="margin-left: 50px; margin-right: 50px; font-size: 13px">
Explicit/NSFW content warning: this project features examples of heavy metal lyrics and song/album/band names.
These often contain words and themes that some may find offensive/inappropriate.
</pre>


This article is a part of my [heavy metal lyrics project](/projects/heavy-metal-analysis.html).
It provides a top-level overview of the song lyrics dataset.
Below is a dashboard I've put together
([click here for full-size version](https://metal-lyrics-feature-plots.herokuapp.com/){:target="_blank"})
to visualize the complete dataset using the metrics discussed here and in the other articles.
Since I did the analyses here before building the dashboard, some plots will look different.
If you're interested in seeing the full code (a lot is omitted here), check out the
[original notebook](https://github.com/pdqnguyen/metallyrics/blob/main/analyses/lyrics/notebooks/lyrics-part-1-overview.ipynb).
In the [next article](./lyrics-part-2.html)
we'll dive much deeper into evaluating lyrical complexity using various lexical diversity measures.

<span style="font-size: 14px">Note: Dashboard may take a minute to load</span>

<script>
  function resizeIframe(obj) {
    obj.style.height = obj.contentWindow.document.documentElement.scrollHeight + 'px';
  }
</script>

<div style="overflow: scroll; width:100%; height:800px">
<iframe src="https://metal-lyrics-feature-plots.herokuapp.com" title="Dataset dashboard" scrolling="no" 
style="width: 1600px; height: 1200px; border: 0px"></iframe>
</div>


## Summary

**Things we'll do:**

* Clean the data and prepare it for analyses here and elsewhere.
* Use basic statistics to compare heavy metal lyrics at the song, album, and band levels.
* Rank songs, albums, and bands by words per song, unique words per song, words per second, and unique words per second.
* Produce a swarm plot in the style of [Matt Daniels' hip hop lyrics analysis](https://pudding.cool/projects/vocabulary/index.html)
* Compare lyrical statistics between different heavy metal genres.

**Conclusions:**
* <span style="color:#ebc634; font-weight:bold">Lyrical datasets can be much more varied in structure than typical text datasets!</span>
Different lyrical styles make conventional methods of text comparison difficult.
Heavy metal lyrics can range from no words at all, to spoken word passages over a thousand words long.  
* Due to the stylistic diversity in the data, small changes in the methods used to quantify vocabulary can lead
to noticeably different outcomes.
* Bands with the highest overall word counts typically belong to the less "extreme" genres like (traditional) heavy metal and power metal.
This is due to having long, word-dense songs, often with a focus on narrative, coupled with higher album output.
* Short pieces by grindcore and death metal bands often provide the highest density of unique words.
* Using Matt Daniels' metric (unique words in first X words), we see a cluster of outliers at the top end of the distribution:
Cryptopsy, Napalm Death, Cattle Decapitation, Cradle of Filth, Deeds of Flesh, Dying Fetus, and Exhumed.
(The plot here has fewer bands shown than in the dashboard, in order to fit all the names on the figure.)
* <span style="color:#ebc634; font-weight:bold">Word count distributions are correlated with genres in the way you'd expect, but the stylistic diversity in each
genre blurs that correlation, suggesting that attempts at lyrics-based genre classification are going to be a challenge.</span>

## Table of Contents
1. [Module imports](#module-imports)
2. [Dataset](#dataset)
3. [Cleanup song lyrics](#cleanup-song-lyrics)
4. [Reduced dataset](#reduced-dataset)
5. [Word counts by song](#word-counts-by-song)
6. [Word counts by album](#word-counts-by-album)
7. [Word counts by band](#word-counts-by-band)
8. [Word counts among the most popular bands](#word-counts-among-the-most-popular-bands)
9. [Ranking artists by the number of unique words in their first 10,000 words](#ranking-artists-by-the-number-of-unique-words-in-their-first-10000-words)
10. [Word counts by genre](#word-counts-by-genre)

## Imports


<details>
<summary>Show code</summary>
{% highlight python %}
import re
import sys
from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
plt.style.use('seaborn')
import seaborn as sns
sns.set(font_scale=2)
from nltk.corpus import words as nltk_words
from scipy.stats import linregress

from adjustText import adjust_text


sys.path.append('../scripts/')

from nlp import tokenize
{% endhighlight %}
</details><br>


## Data

The dataset used here is the table of artist/album/song info and lyrics for every song in the core dataset.


<details>
<summary>Show code</summary>
{% highlight python %}
df = pd.read_csv('../songs.csv', low_memory=False)
df = df[~df.song_darklyrics.isnull()]
df = df[df.song_darklyrics.str.strip().apply(len) > 0]
{% endhighlight %}
</details><br>


#### Cleanup song lyrics

There were some issues when parsing lyrics. They are handled here since it isn't quite worth it to rescrape all of darklyrics again with a new scraper.


<details>
<summary>Show code</summary>
{% highlight python %}
# Remove songs that are mostly non-English

min_english = 0.6  # A little higher than 50% to include songs with translations, whose lyrics typically include original and translated text

rows = []
song_words = []
for i, row in df.iterrows():
    text = row.song_darklyrics.strip()
    words = tokenize(text)
    english_words = tokenize(text, english_only=True)
    is_english = len(english_words) > min_english * len(words)
    if is_english:
        rows.append(row)
        song_words.append(english_words)
df = pd.DataFrame(rows, columns=df.columns)
df['song_words'] = song_words
{% endhighlight %}
</details><br>

<details>
<summary>Show code</summary>
{% highlight python %}
# Remove songs that were copyright claimed

copyrighted = df.song_darklyrics.str.contains('lyrics were removed due to copyright holder\'s request')
df = df[~copyrighted]
{% endhighlight %}
</details><br>

#### Reduced dataset

For lyrical analyses the data is reduced to just a column of lyrics
(which will become the feature vector upon some transformation to a quantitative representation)
for each song and columns for the most popular genres (the target/label vectors).
These are the genres that appear at least once in isolation, i.e. not accompanied by any other genre,
and that appear in some minimum percentage of songs.
For example, the “black” metal label can appear on bands with or without other genres,
but a label like “atmospheric” never appears on its own despite being fairly popular,
usually because it is more of an adjective to denote subgenres like atmospheric black metal;
thus “black” is included in the reduced label space but “atmospheric” is not.
This reduces the genres to a more manageable set: five genres if the minimum occurrence requirement is set to 10%,
and thirteen if set to 1%.

A five-genre set would be easier to handle but leaves quite a few holes in the label space,
because doom metal, metalcore, folk metal, and many other fairly popular genres are being omitted that may not be
covered by any of the five labels. The larger label set covers just about all the most important genres,
but because eight of them occur in fewer than 10% of all songs, they will force greater class imbalance which
will adversely affect attempts at applying binary classification models later on. For the sake of comparison,
both reduced datasets are saved here, but the rest of this exploratory analysis only looks at the 1% dataset,
while the 10% dataset is reserved for modeling. Each dataset is saved in its raw form and in a truncated (ML-ready)
form containing only the lyrics and genre columns.


<details>
<summary>Show code</summary>
{% highlight python %}
def process_genre(genre):
    # Find words (including hyphenated words) not in parentheses
    out = re.findall('[\w\-]+(?![^(]*\))', genre.lower())
    out = [s for s in out if s != 'metal']
    return out


song_genres = df.band_genre.apply(process_genre)
genres = sorted(set(song_genres.sum()))
genre_cols = {f'genre_{genre}': song_genres.apply(lambda x: int(genre in x)) for genre in genres}
df = df.join(pd.DataFrame(genre_cols))

def get_top_genres(data, min_pct):
    isolated = (data.sum(axis=1) == 1)
    isolated_cols = sorted(set(data[isolated].idxmax(axis=1)))
    top_cols = [col for col in isolated_cols if data[col][isolated].mean() >= min_pct]
    top_genres = [re.sub(r"^genre\_", "", col) for col in top_cols]
    return top_genres

top_genres_1pct = get_top_genres(df[genre_cols.keys()], 0.01)
print(top_genres_1pct)
df_r = df.copy()
drop_cols = [col for col in df.columns if ('genre_' in col) and (re.sub(r"^genre\_", "", col) not in top_genres_1pct)]
df_r.drop(drop_cols, axis=1, inplace=True)

# Only lyrics and genre are relevant for ML later
df_r_ml = pd.DataFrame(index=range(df.shape[0]), columns=['lyrics'] + top_genres_1pct)
df_r_ml['lyrics'] = df['song_darklyrics'].reset_index(drop=True)
df_r_ml[top_genres_1pct] = df[[f"genre_{genre}" for genre in top_genres_1pct]].reset_index(drop=True)
{% endhighlight %}
</details><br>

<pre class="code-output">
['black', 'death', 'deathcore', 'doom', 'folk', 'gothic', 'grindcore', 'heavy', 'metalcore', 'power',
 'progressive', 'symphonic', 'thrash']
</pre>

<details>
<summary>Show code</summary>
{% highlight python %}
top_genres_10pct = get_top_genres(df[genre_cols.keys()], 0.1)
print(top_genres_10pct)
df_rr = df.copy()
drop_cols = [col for col in df.columns if ('genre_' in col) and (re.sub(r"^genre\_", "", col) not in top_genres_10pct)]
df_rr.drop(drop_cols, axis=1, inplace=True)

# Only lyrics and genre are relevant for ML later
df_rr_ml = pd.DataFrame(index=range(df.shape[0]), columns=['lyrics'] + top_genres_10pct)
df_rr_ml['lyrics'] = df['song_darklyrics'].reset_index(drop=True)
df_rr_ml[top_genres_10pct] = df[[f"genre_{genre}" for genre in top_genres_10pct]].reset_index(drop=True)
{% endhighlight %}
</details><br>

<pre class="code-output">
['black', 'death', 'heavy', 'power', 'thrash']
</pre>



## Basic lyrical properties

This section compares looks at word counts and unique word counts, in absolute counts as well as counts per minute, between different songs, albums, bands, and genres. [Part 3](./lyrics2.ipynb) dives much deeper into evaluating lyrical complexity using various lexical diversity measures from the literature.

Song lyrics are tokenized using a custom `tokenize()` function in `nlp.py`.


<details>
<summary>Show code</summary>
{% highlight python %}
df_r = pd.read_csv('../songs-1pct.csv')
df_r['song_words'] = df_r['song_words'].str.split()
top_genres_1pct = [c for c in df_r.columns if 'genre_' in c]

df_rr = pd.read_csv('../songs-10pct.csv')
df_rr['song_words'] = df_rr['song_words'].str.split()
top_genres_10pct = [c for c in df_rr.columns if 'genre_' in c]
{% endhighlight %}
</details><br>


## Word counts by song


<details>
<summary>Show code</summary>
{% highlight python %}
def to_seconds(data):
    """Convert a time string (MM:ss or HH:MM:ss) to seconds
    """
    out = pd.Series(index=data.index, dtype=int)
    for i, x in data.items():
        if isinstance(x, str):
            xs = x.split(':')
            if len(xs) < 3:
                xs = [0] + xs
            seconds = int(xs[0]) * 3600 + int(xs[1]) * 60 + int(xs[2])
        else:
            seconds = 0
        out[i] = seconds
    return out


def get_words_per_second(data):
    out = pd.DataFrame(index=data.index, dtype=float)
    out['words_per_second'] = (data['word_count'] / data['seconds']).round(2)
    out['words_per_second'][out['words_per_second'] == np.inf] = 0
    out['unique_words_per_second'] = (data['unique_word_count'] / data['seconds']).round(2)
    out['unique_words_per_second'][out['unique_words_per_second'] == np.inf] = 0
    return pd.concat([data, out], axis=1)

df_r_songs = df_r[['band_name', 'album_name', 'song_name', 'band_genre', 'song_words', 'song_length']].copy()
df_r_songs = df_r_songs.rename(columns={'band_name': 'band', 'album_name': 'album', 'song_name': 'song', 'band_genre': 'genre', 'song_words': 'words'})
df_r_songs['seconds'] = to_seconds(df_r_songs['song_length'])
df_r_songs = df_r_songs.drop('song_length', axis=1)
df_r_songs['word_count'] = df_r_songs['words'].apply(len)
df_r_songs['unique_word_count'] = df_r_songs['words'].apply(lambda x: len(set(x)))
df_r_songs = get_words_per_second(df_r_songs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

df_r_songs['word_count'].hist(bins=np.logspace(1, 3, 30), ax=ax1)
ax1.set_xscale('log')
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.set_xlabel("word count")
ax1.set_ylabel("number of songs")
ax1.set_title("Words per song")

df_r_songs['unique_word_count'].hist(bins=np.logspace(1, 3, 30), ax=ax2)
ax2.set_xscale('log')
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.set_xlabel("unique word count")
ax2.set_title("Unique words per song")

plt.show()
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics1/lyrics-part-1-overview_24_0.png)

    


#### Songs with highest word counts

The honor of highest word count in a single song goes to the [Bal-Sagoth's "The Obsidian Crown Unbound"](https://youtu.be/xizMG4nI2dk) at over two thousand words. However, most of those words are not sung in the actual song: Bal-Sagoth lyrics typically include the massive collection of narrative text that accompanies their songs. Although the lyrics they sing are still plentiful, there are nowhere near two thousand words spoken in the six-minute symphonic black metal track.

This makes the forty-minute prog metal epic [Crimson by Edge of Sanity](https://youtu.be/St6lJaiHYIc) a better contender for most verbose song. Still, such a claim might be challenged by the fact that the digital edition of the album, which a listener would find on Spotify for instance, divides the single-track album into eight parts. That said, DarkLyrics keeps the original one-track format.

At third place is another multi-part song, [Mirror of Souls](https://youtu.be/y6n1kMsLbc8) by the Christian progressive/power metal group Theocracy. This is less contentious since the official track listing considers this a single track.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>album</th>
      <th>song</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>358</td>
      <td>2259</td>
      <td>897</td>
      <td>6.31</td>
      <td>2.51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Edge of Sanity</td>
      <td>Crimson</td>
      <td>Crimson</td>
      <td>Progressive Death Metal</td>
      <td>2400</td>
      <td>1948</td>
      <td>658</td>
      <td>0.81</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Theocracy</td>
      <td>Mirror of Souls</td>
      <td>Mirror of Souls</td>
      <td>Epic Progressive Power Metal</td>
      <td>1346</td>
      <td>1556</td>
      <td>457</td>
      <td>1.16</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cephalectomy</td>
      <td>Sign of Chaos</td>
      <td>Unto the Darkly Shining Abyss</td>
      <td>Experimental Death Metal/Grindcore</td>
      <td>383</td>
      <td>1338</td>
      <td>391</td>
      <td>3.49</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>To Dethrone the Witch-Queen of Mytos K'unn (Th...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>405</td>
      <td>1306</td>
      <td>548</td>
      <td>3.22</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Seventh Wonder</td>
      <td>The Great Escape</td>
      <td>The Great Escape</td>
      <td>Progressive Metal</td>
      <td>1814</td>
      <td>1306</td>
      <td>504</td>
      <td>0.72</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Unfettering the Hoary Sentinels of Karnak</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>262</td>
      <td>1237</td>
      <td>560</td>
      <td>4.72</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bal-Sagoth</td>
      <td>Battle Magic</td>
      <td>Blood Slakes the Sand at the Circus Maximus</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>533</td>
      <td>1186</td>
      <td>530</td>
      <td>2.23</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Redemption</td>
      <td>Redemption</td>
      <td>Something Wicked This Way Comes</td>
      <td>Progressive Metal</td>
      <td>1466</td>
      <td>1114</td>
      <td>439</td>
      <td>0.76</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cephalectomy</td>
      <td>Sign of Chaos</td>
      <td>Gates to the Spheres of Astral Frost</td>
      <td>Experimental Death Metal/Grindcore</td>
      <td>233</td>
      <td>1065</td>
      <td>339</td>
      <td>4.57</td>
      <td>1.45</td>
    </tr>
  </tbody>
</table>
</div>




#### Songs with highest unique word counts





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>album</th>
      <th>song</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>358</td>
      <td>2259</td>
      <td>897</td>
      <td>6.31</td>
      <td>2.51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Edge of Sanity</td>
      <td>Crimson</td>
      <td>Crimson</td>
      <td>Progressive Death Metal</td>
      <td>2400</td>
      <td>1948</td>
      <td>658</td>
      <td>0.81</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Unfettering the Hoary Sentinels of Karnak</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>262</td>
      <td>1237</td>
      <td>560</td>
      <td>4.72</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>To Dethrone the Witch-Queen of Mytos K'unn (Th...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>405</td>
      <td>1306</td>
      <td>548</td>
      <td>3.22</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bal-Sagoth</td>
      <td>Battle Magic</td>
      <td>Blood Slakes the Sand at the Circus Maximus</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>533</td>
      <td>1186</td>
      <td>530</td>
      <td>2.23</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Seventh Wonder</td>
      <td>The Great Escape</td>
      <td>The Great Escape</td>
      <td>Progressive Metal</td>
      <td>1814</td>
      <td>1306</td>
      <td>504</td>
      <td>0.72</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Theocracy</td>
      <td>Mirror of Souls</td>
      <td>Mirror of Souls</td>
      <td>Epic Progressive Power Metal</td>
      <td>1346</td>
      <td>1556</td>
      <td>457</td>
      <td>1.16</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Redemption</td>
      <td>Redemption</td>
      <td>Something Wicked This Way Comes</td>
      <td>Progressive Metal</td>
      <td>1466</td>
      <td>1114</td>
      <td>439</td>
      <td>0.76</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>The Splendour of a Thousand Swords Gleaming Be...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>363</td>
      <td>977</td>
      <td>429</td>
      <td>2.69</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>443</td>
      <td>1018</td>
      <td>427</td>
      <td>2.3</td>
      <td>0.96</td>
    </tr>
  </tbody>
</table>
</div>




#### Songs with highest word density

Unfortunately songs by number of words per seconds or unique words per second yields mostly songs with commentary/not-sung lyrics...



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>album</th>
      <th>song</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Nanowar of Steel</td>
      <td>Into Gay Pride Ride</td>
      <td>Karkagnor's Song - In the Forest</td>
      <td>Heavy/Power Metal/Hard Rock</td>
      <td>10</td>
      <td>286</td>
      <td>159</td>
      <td>28.6</td>
      <td>15.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cripple Bastards</td>
      <td>Your Lies in Check</td>
      <td>Rending Aphthous Fevers</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>7</td>
      <td>81</td>
      <td>51</td>
      <td>11.57</td>
      <td>7.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cripple Bastards</td>
      <td>Desperately Insensitive</td>
      <td>Bomb ABC No Rio</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>63</td>
      <td>482</td>
      <td>286</td>
      <td>7.65</td>
      <td>4.54</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hortus Animae</td>
      <td>Waltzing Mephisto</td>
      <td>A Lifetime Obscurity</td>
      <td>Progressive Black Metal</td>
      <td>57</td>
      <td>418</td>
      <td>193</td>
      <td>7.33</td>
      <td>3.39</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gore</td>
      <td>Consumed by Slow Decay</td>
      <td>Inquisitive Corporeal Recremation</td>
      <td>Goregrind</td>
      <td>16</td>
      <td>103</td>
      <td>66</td>
      <td>6.44</td>
      <td>4.12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>358</td>
      <td>2259</td>
      <td>897</td>
      <td>6.31</td>
      <td>2.51</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cripple Bastards</td>
      <td>Variante alla morte</td>
      <td>Karma del riscatto</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>62</td>
      <td>388</td>
      <td>197</td>
      <td>6.26</td>
      <td>3.18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Trans-Siberian Orchestra</td>
      <td>The Christmas Attic</td>
      <td>The Ghosts of Christmas Eve</td>
      <td>Orchestral/Progressive Rock/Metal</td>
      <td>135</td>
      <td>815</td>
      <td>311</td>
      <td>6.04</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cripple Bastards</td>
      <td>Desperately Insensitive</td>
      <td>The Mushroom Diarrhoea</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>47</td>
      <td>276</td>
      <td>135</td>
      <td>5.87</td>
      <td>2.87</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cripple Bastards</td>
      <td>Desperately Insensitive</td>
      <td>When Immunities Fall</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>90</td>
      <td>527</td>
      <td>260</td>
      <td>5.86</td>
      <td>2.89</td>
    </tr>
  </tbody>
</table>
</div>




#### Songs with highest unique word density




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>album</th>
      <th>song</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Nanowar of Steel</td>
      <td>Into Gay Pride Ride</td>
      <td>Karkagnor's Song - In the Forest</td>
      <td>Heavy/Power Metal/Hard Rock</td>
      <td>10</td>
      <td>286</td>
      <td>159</td>
      <td>28.6</td>
      <td>15.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cripple Bastards</td>
      <td>Your Lies in Check</td>
      <td>Rending Aphthous Fevers</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>7</td>
      <td>81</td>
      <td>51</td>
      <td>11.57</td>
      <td>7.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cripple Bastards</td>
      <td>Desperately Insensitive</td>
      <td>Bomb ABC No Rio</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>63</td>
      <td>482</td>
      <td>286</td>
      <td>7.65</td>
      <td>4.54</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gore</td>
      <td>Consumed by Slow Decay</td>
      <td>Inquisitive Corporeal Recremation</td>
      <td>Goregrind</td>
      <td>16</td>
      <td>103</td>
      <td>66</td>
      <td>6.44</td>
      <td>4.12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Napalm Death</td>
      <td>Scum</td>
      <td>You Suffer</td>
      <td>Hardcore Punk (early); Grindcore/Death Metal (...</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cripple Bastards</td>
      <td>Your Lies in Check</td>
      <td>Intelligence Means...</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>19</td>
      <td>93</td>
      <td>70</td>
      <td>4.89</td>
      <td>3.68</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cripple Bastards</td>
      <td>Desperately Insensitive</td>
      <td>Idiots Think Slower</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>30</td>
      <td>154</td>
      <td>105</td>
      <td>5.13</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Wormrot</td>
      <td>Dirge</td>
      <td>You Suffer but Why Is It My Problem</td>
      <td>Grindcore</td>
      <td>4</td>
      <td>14</td>
      <td>14</td>
      <td>3.5</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hortus Animae</td>
      <td>Waltzing Mephisto</td>
      <td>A Lifetime Obscurity</td>
      <td>Progressive Black Metal</td>
      <td>57</td>
      <td>418</td>
      <td>193</td>
      <td>7.33</td>
      <td>3.39</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Circle of Dead Children</td>
      <td>Human Harvest</td>
      <td>White Trash Headache</td>
      <td>Brutal Death Metal, Grindcore</td>
      <td>6</td>
      <td>21</td>
      <td>20</td>
      <td>3.5</td>
      <td>3.33</td>
    </tr>
  </tbody>
</table>
</div>




## Word counts by album

Grouping song lyrics by album shows Blind Guardian's 75-minute [Twilight Orchestra: Legacy of the Dark Lands](https://en.wikipedia.org/wiki/Legacy_of_the_Dark_Lands) coming out on top as the album with the highest word count, even outstripping all of Bal-Sagoth's albums. Not counting Bal-Sagoth, Cradle of Filth's [Darkly, Darkly, Venus Aversa](https://en.wikipedia.org/wiki/Darkly,_Darkly,_Venus_Aversa)
has the highest number of unique words. Unfortunately most of the highest-ranking albums by words per second are albums with unsung lyrics as well.


<details>
<summary>Show code</summary>
{% highlight python %}
df_r_albums = pd.concat([
    df_r_songs.groupby(['band', 'album'])['genre'].first(),
    df_r_songs.groupby(['band', 'album'])['words'].sum(),
    df_r_songs.groupby(['band', 'album'])['seconds'].sum(),
], axis=1)
df_r_albums['word_count'] = df_r_albums['words'].apply(len)
df_r_albums['unique_word_count'] = df_r_albums['words'].apply(lambda x: len(set(x)))
df_r_albums = get_words_per_second(df_r_albums)
df_r_albums = df_r_albums.reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

df_r_albums['word_count'].hist(bins=np.logspace(2, 4, 30), ax=ax1)
ax1.set_xscale('log')
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.set_xlabel("word count")
ax1.set_ylabel("number of albums")
ax1.set_title("Words per album")

df_r_albums['unique_word_count'].hist(bins=np.logspace(2, 4, 30), ax=ax2)
ax2.set_xscale('log')
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.set_xlabel("unique word count")
ax2.set_title("Unique words per album")

plt.show()
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics1/lyrics-part-1-overview_39_0.png)




#### Albums with highest word counts





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>album</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Blind Guardian</td>
      <td>Twilight Orchestra: Legacy of the Dark Lands</td>
      <td>Speed Metal (early); Power Metal (later)</td>
      <td>8210</td>
      <td>8812</td>
      <td>1010</td>
      <td>1.07</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>3639</td>
      <td>6979</td>
      <td>2073</td>
      <td>1.92</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Gentle Storm</td>
      <td>The Diary</td>
      <td>Symphonic/Progressive Metal/Rock</td>
      <td>6832</td>
      <td>6896</td>
      <td>810</td>
      <td>1.01</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>3157</td>
      <td>6500</td>
      <td>1634</td>
      <td>2.06</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cephalectomy</td>
      <td>Sign of Chaos</td>
      <td>Experimental Death Metal/Grindcore</td>
      <td>2264</td>
      <td>6392</td>
      <td>958</td>
      <td>2.82</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kenn Nardi</td>
      <td>Dancing with the Past</td>
      <td>Progressive Metal/Rock</td>
      <td>9007</td>
      <td>6174</td>
      <td>1200</td>
      <td>0.69</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mayan</td>
      <td>Dhyana</td>
      <td>Symphonic Death Metal</td>
      <td>7394</td>
      <td>5618</td>
      <td>718</td>
      <td>0.76</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Vulvodynia</td>
      <td>Cognizant Castigation</td>
      <td>Deathcore/Brutal Death Metal</td>
      <td>4928</td>
      <td>5400</td>
      <td>854</td>
      <td>1.1</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cradle of Filth</td>
      <td>Darkly, Darkly, Venus Aversa</td>
      <td>Death Metal (early); Symphonic Black Metal (mi...</td>
      <td>4557</td>
      <td>5374</td>
      <td>1719</td>
      <td>1.18</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>3218</td>
      <td>5264</td>
      <td>1033</td>
      <td>1.64</td>
      <td>0.32</td>
    </tr>
  </tbody>
</table>
</div>




#### Albums with highest unique word counts





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>album</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>3639</td>
      <td>6979</td>
      <td>2073</td>
      <td>1.92</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cradle of Filth</td>
      <td>Darkly, Darkly, Venus Aversa</td>
      <td>Death Metal (early); Symphonic Black Metal (mi...</td>
      <td>4557</td>
      <td>5374</td>
      <td>1719</td>
      <td>1.18</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>3157</td>
      <td>6500</td>
      <td>1634</td>
      <td>2.06</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cradle of Filth</td>
      <td>Midian</td>
      <td>Death Metal (early); Symphonic Black Metal (mi...</td>
      <td>3217</td>
      <td>3821</td>
      <td>1474</td>
      <td>1.19</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cradle of Filth</td>
      <td>The Manticore and Other Horrors</td>
      <td>Death Metal (early); Symphonic Black Metal (mi...</td>
      <td>3416</td>
      <td>3498</td>
      <td>1359</td>
      <td>1.02</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Lascaille's Shroud</td>
      <td>The Roads Leading North</td>
      <td>Progressive Death Metal</td>
      <td>7720</td>
      <td>4800</td>
      <td>1359</td>
      <td>0.62</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Enfold Darkness</td>
      <td>Adversary Omnipotent</td>
      <td>Technical/Melodic Black/Death Metal</td>
      <td>3750</td>
      <td>4540</td>
      <td>1321</td>
      <td>1.21</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The Agonist</td>
      <td>Prisoners</td>
      <td>Melodic Death Metal/Metalcore</td>
      <td>3213</td>
      <td>3223</td>
      <td>1265</td>
      <td>1.0</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dying Fetus</td>
      <td>Wrong One to Fuck With</td>
      <td>Brutal Death Metal/Grindcore</td>
      <td>3239</td>
      <td>2731</td>
      <td>1264</td>
      <td>0.84</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cripple Bastards</td>
      <td>Desperately Insensitive</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>1213</td>
      <td>3368</td>
      <td>1260</td>
      <td>2.78</td>
      <td>1.04</td>
    </tr>
  </tbody>
</table>
</div>




#### Albums with highest word density




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>album</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Cephalectomy</td>
      <td>Sign of Chaos</td>
      <td>Experimental Death Metal/Grindcore</td>
      <td>2264</td>
      <td>6392</td>
      <td>958</td>
      <td>2.82</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cripple Bastards</td>
      <td>Desperately Insensitive</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>1213</td>
      <td>3368</td>
      <td>1260</td>
      <td>2.78</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cripple Bastards</td>
      <td>Misantropo a senso unico</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>1459</td>
      <td>3710</td>
      <td>1069</td>
      <td>2.54</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Samsas Traum</td>
      <td>Heiliges Herz - Das Schwert deiner Sonne</td>
      <td>Gothic/Avant-garde Black Metal (early); Neocla...</td>
      <td>47</td>
      <td>99</td>
      <td>59</td>
      <td>2.11</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>3157</td>
      <td>6500</td>
      <td>1634</td>
      <td>2.06</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fuck the Facts</td>
      <td>Mullet Fever</td>
      <td>Grindcore</td>
      <td>41</td>
      <td>84</td>
      <td>60</td>
      <td>2.05</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Melvins</td>
      <td>Prick</td>
      <td>Sludge Metal, Various</td>
      <td>257</td>
      <td>504</td>
      <td>193</td>
      <td>1.96</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>3639</td>
      <td>6979</td>
      <td>2073</td>
      <td>1.92</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Municipal Waste</td>
      <td>Waste 'Em All</td>
      <td>Thrash Metal/Crossover</td>
      <td>848</td>
      <td>1615</td>
      <td>630</td>
      <td>1.9</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cripple Bastards</td>
      <td>Variante alla morte</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>1699</td>
      <td>3211</td>
      <td>1019</td>
      <td>1.89</td>
      <td>0.6</td>
    </tr>
  </tbody>
</table>
</div>




#### Albums with highest unique word density




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>album</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Fuck the Facts</td>
      <td>Mullet Fever</td>
      <td>Grindcore</td>
      <td>41</td>
      <td>84</td>
      <td>60</td>
      <td>2.05</td>
      <td>1.46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Samsas Traum</td>
      <td>Heiliges Herz - Das Schwert deiner Sonne</td>
      <td>Gothic/Avant-garde Black Metal (early); Neocla...</td>
      <td>47</td>
      <td>99</td>
      <td>59</td>
      <td>2.11</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cripple Bastards</td>
      <td>Desperately Insensitive</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>1213</td>
      <td>3368</td>
      <td>1260</td>
      <td>2.78</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Suicidal Tendencies</td>
      <td>Controlled by Hatred / Feel like Shit... Deja Vu</td>
      <td>Thrash Metal/Crossover, Hardcore Punk</td>
      <td>196</td>
      <td>364</td>
      <td>162</td>
      <td>1.86</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Haggard</td>
      <td>Tales of Ithiria</td>
      <td>Progressive Death Metal (early); Classical/Orc...</td>
      <td>240</td>
      <td>299</td>
      <td>195</td>
      <td>1.25</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gridlink</td>
      <td>Orphan</td>
      <td>Technical Grindcore</td>
      <td>728</td>
      <td>1249</td>
      <td>565</td>
      <td>1.72</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Frightmare</td>
      <td>Midnight Murder Mania</td>
      <td>Death/Thrash Metal/Grindcore</td>
      <td>276</td>
      <td>349</td>
      <td>208</td>
      <td>1.26</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Melvins</td>
      <td>Prick</td>
      <td>Sludge Metal, Various</td>
      <td>257</td>
      <td>504</td>
      <td>193</td>
      <td>1.96</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Municipal Waste</td>
      <td>Waste 'Em All</td>
      <td>Thrash Metal/Crossover</td>
      <td>848</td>
      <td>1615</td>
      <td>630</td>
      <td>1.9</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Gridlink</td>
      <td>Amber Gray</td>
      <td>Technical Grindcore</td>
      <td>710</td>
      <td>967</td>
      <td>522</td>
      <td>1.36</td>
      <td>0.74</td>
    </tr>
  </tbody>
</table>
</div>




## Word counts by band

Surprisingly, Bal-Sagoth’s inflated lyric counts do not matter much when comparing entire bands, perhaps due to how short their discography is. The bands with the highest word counts typically have massive discographies, and are usually power metal or heavy metal bands. [Saxon](https://en.wikipedia.org/wiki/Saxon_(band)) rank highest in raw word counts, with nearly over 39,000 words spanning nearly sixteen hours of music, while [Cradle of Filth](https://en.wikipedia.org/wiki/Cradle_of_Filth) throughout their eleven-hour-long discography have used the greatest number of unique words.


<details>
<summary>Show code</summary>
{% highlight python %}
df_r_bands = pd.concat([
    df_r_albums.groupby('band')['genre'].first(),
    df_r_albums.groupby('band')['words'].sum(),
    df_r_albums.groupby('band')['seconds'].sum(),
], axis=1)
df_r_bands['word_count'] = df_r_bands['words'].apply(len)
df_r_bands['unique_word_count'] = df_r_bands['words'].apply(lambda x: len(set(x)))
df_r_bands = get_words_per_second(df_r_bands)
df_r_bands = df_r_bands.reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

df_r_bands['word_count'].hist(bins=np.logspace(2, 4, 30), ax=ax1)
ax1.set_xscale('log')
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.set_xlabel("word count")
ax1.set_ylabel("number of bands")
ax1.set_title("Words per band")

df_r_bands['unique_word_count'].hist(bins=np.logspace(2, 4, 30), ax=ax2)
ax2.set_xscale('log')
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.set_xlabel("unique word count")
ax2.set_title("Unique words per band")

plt.show()
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics1/lyrics-part-1-overview_52_0.png)

    



#### Bands with highest word counts





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Saxon</td>
      <td>NWOBHM, Heavy Metal</td>
      <td>56651</td>
      <td>39019</td>
      <td>2732</td>
      <td>0.69</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iron Maiden</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>58093</td>
      <td>38525</td>
      <td>3377</td>
      <td>0.66</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cradle of Filth</td>
      <td>Death Metal (early); Symphonic Black Metal (mi...</td>
      <td>39655</td>
      <td>37932</td>
      <td>6054</td>
      <td>0.96</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rage</td>
      <td>Heavy/Speed/Power Metal</td>
      <td>58844</td>
      <td>36476</td>
      <td>2959</td>
      <td>0.62</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Blind Guardian</td>
      <td>Speed Metal (early); Power Metal (later)</td>
      <td>38090</td>
      <td>34834</td>
      <td>2415</td>
      <td>0.91</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Overkill</td>
      <td>Thrash Metal, Thrash/Groove Metal</td>
      <td>47662</td>
      <td>33728</td>
      <td>3173</td>
      <td>0.71</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cannibal Corpse</td>
      <td>Death Metal</td>
      <td>34966</td>
      <td>32700</td>
      <td>4567</td>
      <td>0.94</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Helloween</td>
      <td>Power/Speed Metal</td>
      <td>46600</td>
      <td>31223</td>
      <td>2699</td>
      <td>0.67</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Judas Priest</td>
      <td>Heavy Metal</td>
      <td>52478</td>
      <td>31097</td>
      <td>3551</td>
      <td>0.59</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Accept</td>
      <td>Heavy Metal</td>
      <td>40939</td>
      <td>29029</td>
      <td>2655</td>
      <td>0.71</td>
      <td>0.06</td>
    </tr>
  </tbody>
</table>
</div>




#### Bands with highest unique word counts



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Cradle of Filth</td>
      <td>Death Metal (early); Symphonic Black Metal (mi...</td>
      <td>39655</td>
      <td>37932</td>
      <td>6054</td>
      <td>0.96</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Napalm Death</td>
      <td>Hardcore Punk (early); Grindcore/Death Metal (...</td>
      <td>36761</td>
      <td>22260</td>
      <td>5081</td>
      <td>0.61</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cannibal Corpse</td>
      <td>Death Metal</td>
      <td>34966</td>
      <td>32700</td>
      <td>4567</td>
      <td>0.94</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Skyclad</td>
      <td>Folk Metal</td>
      <td>25903</td>
      <td>26651</td>
      <td>4179</td>
      <td>1.03</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The Black Dahlia Murder</td>
      <td>Melodic Death Metal</td>
      <td>19546</td>
      <td>21928</td>
      <td>4132</td>
      <td>1.12</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Dying Fetus</td>
      <td>Brutal Death Metal/Grindcore</td>
      <td>16783</td>
      <td>15110</td>
      <td>3930</td>
      <td>0.9</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Elephant</td>
      <td>Epic Heavy Metal</td>
      <td>19168</td>
      <td>13893</td>
      <td>3865</td>
      <td>0.72</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sodom</td>
      <td>Black/Speed Metal (early); Thrash Metal (later)</td>
      <td>35072</td>
      <td>26202</td>
      <td>3824</td>
      <td>0.75</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>16021</td>
      <td>21458</td>
      <td>3730</td>
      <td>1.34</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cattle Decapitation</td>
      <td>Progressive Death Metal/Grindcore</td>
      <td>17320</td>
      <td>14363</td>
      <td>3624</td>
      <td>0.83</td>
      <td>0.21</td>
    </tr>
  </tbody>
</table>
</div>




#### Bands with highest word density



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Cephalectomy</td>
      <td>Experimental Death Metal/Grindcore</td>
      <td>4081</td>
      <td>9410</td>
      <td>1415</td>
      <td>2.31</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cripple Bastards</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>8256</td>
      <td>16210</td>
      <td>3538</td>
      <td>1.96</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Archagathus</td>
      <td>Grindcore (early); Goregrind (later)</td>
      <td>1149</td>
      <td>2113</td>
      <td>792</td>
      <td>1.84</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gridlink</td>
      <td>Technical Grindcore</td>
      <td>1438</td>
      <td>2216</td>
      <td>943</td>
      <td>1.54</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Crumbsuckers</td>
      <td>Thrash Metal/Crossover</td>
      <td>1368</td>
      <td>2066</td>
      <td>516</td>
      <td>1.51</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Archspire</td>
      <td>Technical Death Metal</td>
      <td>7084</td>
      <td>10633</td>
      <td>2344</td>
      <td>1.5</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Embalmer</td>
      <td>Death Metal</td>
      <td>266</td>
      <td>392</td>
      <td>145</td>
      <td>1.47</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Municipal Waste</td>
      <td>Thrash Metal/Crossover</td>
      <td>7479</td>
      <td>10587</td>
      <td>2167</td>
      <td>1.42</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Varg</td>
      <td>Melodic Death/Black Metal/Metalcore</td>
      <td>467</td>
      <td>645</td>
      <td>205</td>
      <td>1.38</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Blood Freak</td>
      <td>Death Metal/Grindcore</td>
      <td>4447</td>
      <td>6123</td>
      <td>1778</td>
      <td>1.38</td>
      <td>0.4</td>
    </tr>
  </tbody>
</table>
</div>




#### Bands with highest unique word density



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>genre</th>
      <th>seconds</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>words_per_second</th>
      <th>unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Archagathus</td>
      <td>Grindcore (early); Goregrind (later)</td>
      <td>1149</td>
      <td>2113</td>
      <td>792</td>
      <td>1.84</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gridlink</td>
      <td>Technical Grindcore</td>
      <td>1438</td>
      <td>2216</td>
      <td>943</td>
      <td>1.54</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Samsas Traum</td>
      <td>Gothic/Avant-garde Black Metal (early); Neocla...</td>
      <td>307</td>
      <td>394</td>
      <td>200</td>
      <td>1.28</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Embalmer</td>
      <td>Death Metal</td>
      <td>266</td>
      <td>392</td>
      <td>145</td>
      <td>1.47</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Acrania</td>
      <td>Brutal Deathcore</td>
      <td>1674</td>
      <td>2282</td>
      <td>902</td>
      <td>1.36</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Revenance</td>
      <td>Brutal Death Metal</td>
      <td>218</td>
      <td>241</td>
      <td>115</td>
      <td>1.11</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Illusion Force</td>
      <td>Power Metal</td>
      <td>306</td>
      <td>386</td>
      <td>161</td>
      <td>1.26</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hiroshima Will Burn</td>
      <td>Technical Deathcore</td>
      <td>691</td>
      <td>737</td>
      <td>360</td>
      <td>1.07</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The County Medical Examiners</td>
      <td>Goregrind</td>
      <td>1794</td>
      <td>1662</td>
      <td>932</td>
      <td>0.93</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Decrypt</td>
      <td>Technical Death Metal/Grindcore</td>
      <td>1775</td>
      <td>2208</td>
      <td>911</td>
      <td>1.24</td>
      <td>0.51</td>
    </tr>
  </tbody>
</table>
</div>




#### Word counts among the most popular bands

To pick out the most popular bands,
we can filter out artists with fewer than a certain number of reviews.
Plotting out their full-discography unique word counts,
we find that there is a generally linear relationship between the
number of unique words and overall discography length, which is not surprising.
Cradle of Filth sits farthest from the trend line,
with about twice as many words and unique words in their lyrics than expected.
Opeth seems like an outlier on the flip side,
probably due their songs being very heavily instrumental
(Dream Theater probably incorporates more instrumentals but the narrative
nature of their lyrics results in them falling much more in line with heavy/power metal bands).


<details>
<summary>Show code</summary>
{% highlight python %}
min_reviews = 20

bands_popular = sorted(set(df_r[df_r['album_review_num'] > min_reviews].band_name))
df_r_bands_popular = df_r_bands[df_r_bands.band.isin(bands_popular)].set_index('band', drop=True)

plt.figure(figsize=(14, 8))
xlist, ylist = [], []
for band, row in df_r_bands_popular.iterrows():
    x = row['seconds'] / 3600.0
    y = row['word_count'] / 1000.0
    xlist.append(x)
    ylist.append(y)
    plt.plot(x, y, 'r.')

res = linregress(df_r_bands.seconds / 3600.0, df_r_bands.word_count / 1000.0)
xline = np.linspace(0, df_r_bands_popular.seconds.max() / 3600.0)
yline = xline * res.slope + res.intercept
plt.plot(xline, yline, label='full dataset linear fit')

texts = []
for x, y, band in zip(xlist, ylist, df_r_bands_popular.index):
    texts.append(plt.text(x, y, band, fontsize=12))
adjust_text(texts)

plt.xlabel('Discography length (hours)')
plt.ylabel('Total word count (thousands)')
plt.legend(fontsize=14)
plt.show()
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics1/lyrics-part-1-overview_64_0.png)

    



<details>
<summary>Show code</summary>
{% highlight python %}
plt.figure(figsize=(14, 8))
xlist, ylist = [], []
for band, row in df_r_bands_popular.iterrows():
    x = row['seconds'] / 3600.0
    y = row['unique_word_count'] / 1000.0
    xlist.append(x)
    ylist.append(y)
    plt.plot(x, y, 'r.')

res = linregress(df_r_bands.seconds / 3600.0, df_r_bands.unique_word_count / 1000.0)
xline = np.linspace(0, df_r_bands_popular.seconds.max() / 3600.0)
yline = xline * res.slope + res.intercept
plt.plot(xline, yline, label='full dataset linear fit')

texts = []
for x, y, band in zip(xlist, ylist, df_r_bands_popular.index):
    texts.append(plt.text(x, y, band, fontsize=12))
adjust_text(texts)

plt.xlabel('Discography length (hours)')
plt.ylabel('Unique word count (thousands)')
plt.legend(fontsize=14)
plt.show()
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics1/lyrics-part-1-overview_65_0.png)

    


## Ranking artists by the number of unique words in their first 10,000 words

A few years ago, Matt Daniels of The Pudding wrote up [an article](https://pudding.cool/projects/vocabulary/index.html)
comparing the number of unique words used by several famous rappers in their first 35,000 words.
A similar comparison can be done with the metal lyrics here,
although since heavy metal tends to have more instrumentals and metal musicians don't put out as many songs as rappers do,
I chose to look at each artist's first 10,000 words.
Here, for clarity only the top 100 bands by number of album reviews are shown
but the full plot at the top of the page shows the top 200.
Interestingly, there's a gap between the cluster of highest unique words and the main field of artists.
Every band in the outlier cluster is associated with death metal, hinting at a correlation in genre.
On the dashboard you can filter by genres to see where on the swarm plot those bands lie.


<details>
<summary>Show code</summary>
{% highlight python %}
,"""Copied from https://stackoverflow.com/questions/55005272/get-bounding-boxes-of-individual-elements-of-a-pathcollection-from-plt-scatter
"""

from matplotlib.path import get_path_collection_extents

def getbb(sc, ax):
    """ Function to return a list of bounding boxes in data coordinates
        for a scatter plot """
    ax.figure.canvas.draw() # need to draw before the transforms are set.
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            # for usual scatters you have one path, but several offsets
            paths = [paths[0]]*len(offsets)
        if len(transforms) < len(offsets):
            # often you may have a single scatter size, but several offsets
            transforms = [transforms[0]]*len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t],
                [o], transOffset.frozen())
            bboxes.append(result.transformed(ax.transData.inverted()))

    return bboxes


def plot_swarm(data, names):
    fig = plt.figure(figsize=(25, 12), facecolor='black')
    ax = sns.swarmplot(x=data, size=50, zorder=1)

    # Get bounding boxes of scatter points
    cs = ax.collections[0]
    boxes = getbb(cs, ax)

    # Add text to circles
    for i, box in enumerate(boxes):
        x = box.x0 + box.width / 2
        y = box.y0 + box.height / 2
        s = names.iloc[i].replace(' ', '\n')
        txt = ax.text(x, y, s, color='white', va='center', ha='center')

        # Shrink font size until text fits completely in circle
        for fs in range(10, 1, -1):
            txt.set_fontsize(fs)
            tbox = txt.get_window_extent().transformed(ax.transData.inverted())
            if (
                    abs(tbox.width) < np.cos(0.5) * abs(box.width)
                    and abs(tbox.height) < np.cos(0.5) * abs(box.height)
            ):
                break

    ax.xaxis.tick_top()
    ax.set_xlabel('')
    ax.tick_params(axis='both', colors='white')

    return fig


band_words = pd.concat(
    (
        df_r.groupby('band_id')['band_name'].first(),
        df_r.groupby(['band_id', 'album_name'])['album_review_num'].first().groupby('band_id').sum(),
        df_r.groupby('band_id')['song_words'].sum()
    ),
    axis=1
)
band_words.columns = ['name', 'reviews', 'words']
{% endhighlight %}
</details><br>





<details>
<summary>Show code</summary>
{% highlight python %}
num_bands = 100
num_words = 10000

band_filt_words = band_words[band_words['words'].apply(len) >= num_words].sort_values('reviews')[-num_bands:]
band_filt_words['unique_first_words'] = band_filt_words['words'].apply(lambda x: len(set(x[:num_words])))
band_filt_words = band_filt_words.sort_values('unique_first_words')

fig = plot_swarm(band_filt_words['unique_first_words'], band_filt_words['name'])
fig.suptitle(f"# of unique words in first {num_words:,.0f} of artist's lyrics", color='white', fontsize=25)
plt.show()
{% endhighlight %}
</details><br>


![png](/assets/images/heavy-metal-lyrics/lyrics1/lyrics-part-1-overview_72_1.png)

    


# Word counts by genre

Although there are some noticeable trends in the word counts of genres,
overall the distributions of word counts and song lengths per genre are quite broad.
The overlap means lyrical complexity is likely not a sufficient means of distinguishing between genres.
In the next article we'll expand on this, using more sophisticated lexical diversity measures
to quantify the complexity of different genres.


<details>
<summary>Show code</summary>
{% highlight python %}
df_genre_songs = df_r[['band_name', 'album_name', 'song_name'] + top_genres_1pct].copy()
df_genre_songs['word_count'] = df_r_songs.word_count
df_genre_songs['seconds'] = df_r_songs.seconds

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
violindata = []
for genre in top_genres_1pct:
    df_genre = df_genre_songs[df_genre_songs[genre] == 1]
    violindata.append((genre, df_genre['word_count'].values))
violindata.sort(key=lambda x: -np.median(x[1]))
sns.boxplot(data=[x[1] for x in violindata], orient='h', showfliers=False)
ax.set_yticklabels([x[0] for x in violindata])
ax.set_xlim
# ax.set_xlim(0, 500)
ax.set_title("Words per song")
ax.set_xlabel("word count")
plt.show()
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics1/lyrics-part-1-overview_79_0.png)



    



<details>
<summary>Show code</summary>
{% highlight python %}
plt.figure(figsize=(14, 8))
xlist, ylist = [], []
for genre in top_genres_1pct:
    df_genre = df_genre_songs[df_genre_songs[genre] == 1].copy()
    x = df_genre['seconds'].mean() / 60.0
    y = df_genre['word_count'].mean()
    xlist.append(x)
    ylist.append(y)
    plt.plot(x, y, 'r.', ms=10, label=genre)
texts = []
for x, y, genre in zip(xlist, ylist, top_genres_1pct):
    texts.append(plt.text(x, y , genre, fontsize=12))
adjust_text(texts)
plt.xlabel('Average song length (minutes)')
plt.ylabel('Average words per song')
plt.show()
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics1/lyrics-part-1-overview_81_0.png)


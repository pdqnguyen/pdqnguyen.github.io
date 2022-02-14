---
layout: post
date: 2022-01-20
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
9. [Ranking artists by the number of unique words in their first 15,000 words](#ranking-artists-by-the-number-of-unique-words-in-their-first-15000-words)
10. [Word counts by genre](#word-counts-by-genre)


## Module imports


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
</details>

## Dataset

The dataset used here is the table of artist/album/song info and lyrics for every song in the core dataset.


{% highlight python %}
df = pd.read_csv('../songs.csv', low_memory=False)
df = df[~df.song_darklyrics.isnull()]
df = df[df.song_darklyrics.str.strip().apply(len) > 0]
print(df.columns)
{% endhighlight %}
<pre class="output">
Index(['band_name', 'band_id', 'band_url', 'band_country_of_origin',
       'band_location', 'band_status', 'band_formed_in', 'band_genre',
       'band_lyrical_themes', 'band_last_label', 'band_years_active',
       'album_name', 'album_type', 'album_year', 'album_review_num',
       'album_review_avg', 'album_url', 'album_reviews_url', 'song_name',
       'song_length', 'song_url', 'song_darklyrics', 'song_darklyrics_url',
       'band_current_label'],
      dtype='object')
</pre>


## Cleanup song lyrics

<details>
<summary>Show section</summary>
<br>
There were some issues when parsing lyrics.
They are handled here since it isn't quite worth it to rescrape all of darklyrics again with a new scraper.

{% highlight python %}
print('Number of songs', len(df))
{% endhighlight %}

<pre class="code-output">Number of songs 60964</pre>

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
print('Non-English songs removed: ', len(df) - len(rows))
df = pd.DataFrame(rows, columns=df.columns)
df['song_words'] = song_words
{% endhighlight %}

<pre class="code-output">Non-English songs removed:  2724</pre>

{% highlight python %}
# Remove songs that were copyright claimed

copyrighted = df.song_darklyrics.str.contains('lyrics were removed due to copyright holder\'s request')
print('Songs with lyrics removed: ', len(df[copyrighted]))
df = df[~copyrighted]
{% endhighlight %}

<pre class="code-output">Songs with lyrics removed:  66</pre>

</details>


## Reduced dataset

For lyrical analyses the data is reduced to just a column of lyrics
(which will become the feature vector upon some transformation to a quantitative representation)
for each song and columns for the most popular genres (the target/label vectors).
These are the genres that appear at least once in isolation, i.e. not accompanied by any other genre,
and that appear in some minimum percentage of songs.
For example, the "black" metal label can appear on bands with or without other genres,
but a label like "atmospheric" never appears on its own despite being fairly popular,
usually because it is more of an adjective to denote subgenres like atmospheric black metal;
thus "black" is included in the reduced label space but "atmospheric" is not.
This reduces the genres to a more manageable set: five genres if the minimum occurrence requirement is set to 10%,
and thirteen if set to 1%.

A five-genre set would be easier to handle but leaves quite a few holes in the label space, because doom metal,
metalcore, folk metal, and many other fairly popular genres are being omitted that may not be covered by any of 
the five labels. The larger label set covers just about all the most important genres, but because eight of them
occur in fewer than 10% of all songs, they will force greater class imbalance which will adversely affect attempts
at applying binary classification models later on. For the sake of comparison, both reduced datasets are saved here,
but the rest of this exploratory analysis only looks at the 1% dataset, while the 10% dataset is reserved for modeling.
Each dataset is saved in its raw form and in a truncated (ML-ready) form containing only the lyrics and genre columns.

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
genre_cols = [f'genre_{genre}' for genre in genres]
for genre, col in zip(genres, genre_cols):
    df[col] = song_genres.apply(lambda x: int(genre in x))
{% endhighlight %}


{% highlight python %}
def get_top_genres(data, min_pct):
    isolated = (data.sum(axis=1) == 1)
    isolated_cols = sorted(set(data[isolated].idxmax(axis=1)))
    top_cols = [col for col in isolated_cols if data[col][isolated].mean() >= min_pct]
    top_genres = [re.sub(r"^genre\_", "", col) for col in top_cols]
    return top_genres
{% endhighlight %}


{% highlight python %}
top_genres_1pct = get_top_genres(df[genre_cols], 0.01)
print(top_genres_1pct)
df_r = df.copy()
drop_cols = [col for col in df.columns if ('genre_' in col) and (re.sub(r"^genre\_", "", col) not in top_genres_1pct)]
df_r.drop(drop_cols, axis=1, inplace=True)

# Only lyrics and genre are relevant for ML later
df_r_ml = pd.DataFrame(index=range(df.shape[0]), columns=['lyrics'] + top_genres_1pct)
df_r_ml['lyrics'] = df['song_darklyrics'].reset_index(drop=True)
df_r_ml[top_genres_1pct] = df[[f"genre_{genre}" for genre in top_genres_1pct]].reset_index(drop=True)
{% endhighlight %}

<pre class="code-output">['black', 'death', 'deathcore', 'doom', 'folk', 'gothic', 'grindcore', 'heavy', 'metalcore', 'power', 'progressive', 'symphonic',
 'thrash']
</pre>


{% highlight python %}
top_genres_10pct = get_top_genres(df[genre_cols], 0.1)
print(top_genres_10pct)
df_rr = df.copy()
drop_cols = [col for col in df.columns if ('genre_' in col) and (re.sub(r"^genre\_", "", col) not in top_genres_10pct)]
df_rr.drop(drop_cols, axis=1, inplace=True)

# Only lyrics and genre are relevant for ML later
df_rr_ml = pd.DataFrame(index=range(df.shape[0]), columns=['lyrics'] + top_genres_10pct)
df_rr_ml['lyrics'] = df['song_darklyrics'].reset_index(drop=True)
df_rr_ml[top_genres_10pct] = df[[f"genre_{genre}" for genre in top_genres_10pct]].reset_index(drop=True)
{% endhighlight %}

<pre class="code-output">['black', 'death', 'heavy', 'power', 'thrash']</pre>
    


{% highlight python %}
df_r = pd.read_csv('../songs-1pct.csv')
df_r['song_words'] = df_r['song_words'].apply(literal_eval)
top_genres_1pct = [c for c in df_r.columns if 'genre_' in c]

df_rr = pd.read_csv('../songs-10pct.csv')
df_rr['song_words'] = df_rr['song_words'].apply(literal_eval)
top_genres_10pct = [c for c in df_rr.columns if 'genre_' in c]
{% endhighlight %}

</details>


## Word counts by song

<details>
<summary>Show code</summary>
{% highlight python %}
song_word_counts = df_r.song_words.apply(len)
song_unique_word_counts = df_r.song_words.apply(lambda x: len(set(x)))

def to_seconds(data):
    """Convert a time string (MM:ss or HH:MM:ss) to seconds
    """
    out = pd.Series(index=data.index, dtype=int)
    for i, x in data.song_length.items():
        if isinstance(x, str):
            xs = x.split(':')
            if len(xs) < 3:
                xs = [0] + xs
            seconds = int(xs[0]) * 3600 + int(xs[1]) * 60 + int(xs[2])
        else:
            seconds = 0
        out[i] = seconds
    return out

song_seconds = to_seconds(df_r)
song_words_per_second = song_word_counts / song_seconds
song_words_per_second[song_words_per_second == np.inf] = 0
song_unique_words_per_second = song_unique_word_counts / song_seconds
song_unique_words_per_second[song_unique_words_per_second == np.inf] = 0

df_r_songs = df_r[['band_name', 'album_name', 'song_name', 'band_genre']].copy()
df_r_songs['song_word_count'] = song_word_counts
df_r_songs['song_unique_word_count'] = song_unique_word_counts
df_r_songs['song_seconds'] = song_seconds
df_r_songs['song_unique_words_per_second'] = song_unique_words_per_second
{% endhighlight %}

{% highlight python %}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

song_word_counts.hist(bins=np.logspace(1, 3, 30), ax=ax1)
ax1.set_xscale('log')
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.set_xlabel("word count")
ax1.set_ylabel("number of songs")
ax1.set_title("Words per song")

song_unique_word_counts.hist(bins=np.logspace(1, 3, 30), ax=ax2)
ax2.set_xscale('log')
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.set_xlabel("unique word count")
ax2.set_title("Unique words per song")

plt.show()
{% endhighlight %}
</details>
<br>

![png](/assets/images/heavy-metal-lyrics/word_count_histogram.png)

#### Songs with highest word counts

The honor of highest word count in a single song goes to the
[Bal-Sagoth's "The Obsidian Crown Unbound"](https://youtu.be/xizMG4nI2dk) at over two thousand words.
However, most of those words are not sung in the actual song:
Bal-Sagoth lyrics typically include the massive collection of narrative text that accompanies their songs.
Although the lyrics they sing are still plentiful, there are nowhere near two thousand words spoken in the six-minute
symphonic black metal track.

This makes the forty-minute prog metal epic ["Crimson" by Edge of Sanity](https://youtu.be/St6lJaiHYIc)
a better contender for most verbose song.
Still, such a claim might be challenged by the fact that the digital edition of the album,
which a listener would find on Spotify for instance, divides the single-track album into eight parts.
That said, DarkLyrics keeps the original one-track format.

At third place is another multi-part song, [Mirror of Souls](https://youtu.be/y6n1kMsLbc8)
by the Christian progressive/power metal group Theocracy.
This is less contentious since the official track listing considers this a single track.


<details>
<summary>Show table</summary>
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
      <th>7</th>
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
      <th>8</th>
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
      <th>9</th>
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
    <tr>
      <th>10</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Six Score and Ten Oblations to a Malefic Avatar</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>368</td>
      <td>793</td>
      <td>415</td>
      <td>2.15</td>
      <td>1.13</td>
    </tr>
  </tbody>
</table>
</div>
</details>

#### Songs with highest unique word counts

<details>
<summary>Show table</summary>
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
      <th>7</th>
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
      <th>8</th>
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
      <th>9</th>
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
    <tr>
      <th>10</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Six Score and Ten Oblations to a Malefic Avatar</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>368</td>
      <td>793</td>
      <td>415</td>
      <td>2.15</td>
      <td>1.13</td>
    </tr>
  </tbody>
</table>
</div>
</details>

#### Songs with highest word density

Again "The Obsidian Crown Unbound" tops the charts for highest number of words per second, however at second place
is ["The Ghosts of Christmas Eve"](https://youtu.be/bT4ruFp5U2w),
the two-minute intro track to The Christmas Attic by Trans-Siberian Orchestra.
Most of the other tracks on this table are short, typically less than a minute.

<details>
<summary>Show table</summary>
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
      <th>3</th>
      <td>Macabre</td>
      <td>Gloom</td>
      <td>I Need to Kill</td>
      <td>Thrash/Death Metal/Grindcore</td>
      <td>36</td>
      <td>199</td>
      <td>77</td>
      <td>5.53</td>
      <td>2.14</td>
    </tr>
    <tr>
      <th>4</th>
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
      <th>5</th>
      <td>Putrid Pile</td>
      <td>Paraphiliac Perversions</td>
      <td>Toxic Shock Therapy</td>
      <td>Brutal Death Metal</td>
      <td>4</td>
      <td>18</td>
      <td>3</td>
      <td>4.5</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>6</th>
      <td>S.O.D.</td>
      <td>Bigger than the Devil</td>
      <td>Charlie Don't Cheat</td>
      <td>Hardcore/Crossover/Thrash Metal</td>
      <td>25</td>
      <td>105</td>
      <td>74</td>
      <td>4.2</td>
      <td>2.96</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Napalm Death</td>
      <td>Scum</td>
      <td>You Suffer</td>
      <td>Hardcore Punk (early), Grindcore/Death Metal (...</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Welcome</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>131</td>
      <td>490</td>
      <td>230</td>
      <td>3.74</td>
      <td>1.76</td>
    </tr>
    <tr>
      <th>9</th>
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
    <tr>
      <th>10</th>
      <td>Suicidal Tendencies</td>
      <td>Free Your Soul... and Save My Mind</td>
      <td>Cyco Speak</td>
      <td>Thrash Metal/Crossover, Hardcore Punk</td>
      <td>183</td>
      <td>640</td>
      <td>224</td>
      <td>3.5</td>
      <td>1.22</td>
    </tr>
  </tbody>
</table>
</div>
</details>

#### Songs with highest unique word density

This metric tends to favor songs with the lowest word counts, hence the prevalence of more
death/thrash-adjacent styles. The one-second ["You Suffer" by Napalm Death](https://youtu.be/ybGOT4d2Hs8)
squeezes in four unique words--although you can't convince me they're audible--and paying homage to this
masterpiece is [Wormrot's "You Suffer But Why Is It My Problem"](https://youtu.be/2SCjTPlMcPw).

<details>
<summary>Show table</summary>
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
      <td>Napalm Death</td>
      <td>Scum</td>
      <td>You Suffer</td>
      <td>Hardcore Punk (early), Grindcore/Death Metal (...</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
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
      <th>3</th>
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
    <tr>
      <th>4</th>
      <td>S.O.D.</td>
      <td>Bigger than the Devil</td>
      <td>Charlie Don't Cheat</td>
      <td>Hardcore/Crossover/Thrash Metal</td>
      <td>25</td>
      <td>105</td>
      <td>74</td>
      <td>4.2</td>
      <td>2.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Municipal Waste</td>
      <td>Waste 'Em All</td>
      <td>I Want to Kill the President</td>
      <td>Thrash Metal/Crossover</td>
      <td>17</td>
      <td>54</td>
      <td>44</td>
      <td>3.18</td>
      <td>2.59</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Insect Warfare</td>
      <td>World Extermination</td>
      <td>Street Sweeper</td>
      <td>Grindcore</td>
      <td>13</td>
      <td>43</td>
      <td>33</td>
      <td>3.31</td>
      <td>2.54</td>
    </tr>
    <tr>
      <th>7</th>
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
      <th>8</th>
      <td>Corrosion of Conformity</td>
      <td>Eye for an Eye</td>
      <td>No Drunk</td>
      <td>Crossover/Sludge/Southern Metal</td>
      <td>22</td>
      <td>74</td>
      <td>52</td>
      <td>3.36</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Deliverance</td>
      <td>What a Joke</td>
      <td>Happy Star</td>
      <td>Speed/Thrash Metal, Industrial</td>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>2.33</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>10</th>
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
  </tbody>
</table>
</div>
</details>


## Word counts by album

Power metal fans rejoice! Grouping song lyrics by album shows Blind Guardian's 75-minute
[Twilight Orchestra: Legacy of the Dark Lands](https://en.wikipedia.org/wiki/Legacy_of_the_Dark_Lands)
coming out on top, even outstripping all of Bal-Sagoth's albums on raw word counts.


#### Albums with highest word counts

<details>
<summary>Show table</summary>
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
      <td>Speed Metal (early), Power Metal (later)</td>
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
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>3218</td>
      <td>5264</td>
      <td>1033</td>
      <td>1.64</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ayreon</td>
      <td>The Human Equation</td>
      <td>Progressive Metal/Rock</td>
      <td>5950</td>
      <td>4917</td>
      <td>805</td>
      <td>0.83</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Therion</td>
      <td>Beloved Antichrist</td>
      <td>Death Metal (early), Symphonic/Operatic Metal ...</td>
      <td>9110</td>
      <td>4859</td>
      <td>1008</td>
      <td>0.53</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Trans-Siberian Orchestra</td>
      <td>The Christmas Attic</td>
      <td>Orchestral/Progressive Rock/Metal</td>
      <td>4066</td>
      <td>4794</td>
      <td>847</td>
      <td>1.18</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Blind Guardian</td>
      <td>A Night at the Opera</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>4024</td>
      <td>4630</td>
      <td>879</td>
      <td>1.15</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Machine Head</td>
      <td>Catharsis</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>4457</td>
      <td>4623</td>
      <td>1131</td>
      <td>1.04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cage</td>
      <td>Ancient Evil</td>
      <td>Heavy/Power Metal</td>
      <td>4477</td>
      <td>4569</td>
      <td>1204</td>
      <td>1.02</td>
      <td>0.27</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Albums with highest unique word counts

<details>
<summary>Show table</summary>
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
      <th>3</th>
      <td>Cradle of Filth</td>
      <td>Midian</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>3217</td>
      <td>3816</td>
      <td>1471</td>
      <td>1.19</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cradle of Filth</td>
      <td>Darkly, Darkly, Venus Aversa</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>3462</td>
      <td>4235</td>
      <td>1444</td>
      <td>1.22</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cradle of Filth</td>
      <td>Godspeed on the Devil's Thunder</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>4275</td>
      <td>3646</td>
      <td>1382</td>
      <td>0.85</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cradle of Filth</td>
      <td>Damnation and a Day</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>3995</td>
      <td>3836</td>
      <td>1381</td>
      <td>0.96</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cradle of Filth</td>
      <td>The Manticore and Other Horrors</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>3416</td>
      <td>3498</td>
      <td>1359</td>
      <td>1.02</td>
      <td>0.4</td>
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
      <td>Exhumed</td>
      <td>Anatomy Is Destiny</td>
      <td>Death Metal/Grindcore</td>
      <td>2346</td>
      <td>3296</td>
      <td>1259</td>
      <td>1.4</td>
      <td>0.54</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Albums with highest word density

<details>
<summary>Show table</summary>
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
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>3157</td>
      <td>6500</td>
      <td>1634</td>
      <td>2.06</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>5</th>
      <td>Origin</td>
      <td>Informis Infinitas Inhumanitas</td>
      <td>Technical Brutal Death Metal</td>
      <td>1712</td>
      <td>3022</td>
      <td>942</td>
      <td>1.77</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Archspire</td>
      <td>Relentless Mutation</td>
      <td>Technical Death Metal</td>
      <td>1837</td>
      <td>3158</td>
      <td>984</td>
      <td>1.72</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Korpiklaani</td>
      <td>Noita</td>
      <td>Folk Metal</td>
      <td>178</td>
      <td>293</td>
      <td>90</td>
      <td>1.65</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>3218</td>
      <td>5264</td>
      <td>1033</td>
      <td>1.64</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Municipal Waste</td>
      <td>Hazardous Mutation</td>
      <td>Thrash Metal/Crossover</td>
      <td>1425</td>
      <td>2246</td>
      <td>840</td>
      <td>1.58</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Nekrogoblikon</td>
      <td>Heavy Meta</td>
      <td>Melodic Death/Folk Metal</td>
      <td>2151</td>
      <td>3376</td>
      <td>780</td>
      <td>1.57</td>
      <td>0.36</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Albums with highest unique word density

Municipal Waste takes up the top two positions here, with [Waste 'Em All](https://en.wikipedia.org/wiki/Waste_%27Em_All)
and [Hazardous Mutation](https://en.wikipedia.org/wiki/Hazardous_Mutation).
All the non-Bal Sagoth albums here are pretty short, none reaching the forty-minute mark.

<details>
<summary>Show table</summary>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
      <td>Helrunar</td>
      <td>Frostnacht</td>
      <td>Pagan Black Metal</td>
      <td>92</td>
      <td>107</td>
      <td>59</td>
      <td>1.16</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Absurd</td>
      <td>Werwolfthron</td>
      <td>Black Metal/RAC, Pagan Black Metal</td>
      <td>122</td>
      <td>145</td>
      <td>76</td>
      <td>1.19</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Despised Icon</td>
      <td>Consumed by Your Poison</td>
      <td>Deathcore</td>
      <td>639</td>
      <td>591</td>
      <td>375</td>
      <td>0.92</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Municipal Waste</td>
      <td>Hazardous Mutation</td>
      <td>Thrash Metal/Crossover</td>
      <td>1425</td>
      <td>2246</td>
      <td>840</td>
      <td>1.58</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Soilent Green</td>
      <td>Confrontation</td>
      <td>Sludge/Death Metal/Grindcore</td>
      <td>1730</td>
      <td>2511</td>
      <td>1006</td>
      <td>1.45</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Archspire</td>
      <td>The Lucid Collective</td>
      <td>Technical Death Metal</td>
      <td>1661</td>
      <td>2495</td>
      <td>961</td>
      <td>1.5</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>3639</td>
      <td>6979</td>
      <td>2073</td>
      <td>1.92</td>
      <td>0.57</td>
    </tr>
  </tbody>
</table>
</div>
</details>


## Word counts by band

Surprisingly, Bal-Sagoth's inflated lyric counts do not matter much when comparing entire bands,
perhaps due to how short their discography is.
The bands with the highest word counts typically have massive discographies,
and are usually power metal or heavy metal bands.
That said, Cradle of Filth are a huge outlier, with nearly 42,000 words spanning twelve hours of music,
setting them well above the rest in both total word count and total unique word count.


#### Bands with highest word counts

<details>
<summary>Show table</summary>
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
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>44097</td>
      <td>41815</td>
      <td>6415</td>
      <td>0.95</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Saxon</td>
      <td>NWOBHM, Heavy Metal</td>
      <td>53755</td>
      <td>36759</td>
      <td>2665</td>
      <td>0.68</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Iron Maiden</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>52673</td>
      <td>34843</td>
      <td>3241</td>
      <td>0.66</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Blind Guardian</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>38090</td>
      <td>34836</td>
      <td>2416</td>
      <td>0.91</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rage</td>
      <td>Heavy/Speed/Power Metal</td>
      <td>56064</td>
      <td>34314</td>
      <td>2874</td>
      <td>0.61</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Overkill</td>
      <td>Thrash Metal; Thrash/Groove Metal</td>
      <td>47540</td>
      <td>32485</td>
      <td>3119</td>
      <td>0.68</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Helloween</td>
      <td>Power/Speed Metal</td>
      <td>48991</td>
      <td>32472</td>
      <td>2769</td>
      <td>0.66</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tankard</td>
      <td>Thrash Metal</td>
      <td>38493</td>
      <td>30652</td>
      <td>3710</td>
      <td>0.8</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cannibal Corpse</td>
      <td>Death Metal</td>
      <td>32398</td>
      <td>30596</td>
      <td>4377</td>
      <td>0.94</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Judas Priest</td>
      <td>Heavy Metal</td>
      <td>51177</td>
      <td>30143</td>
      <td>3506</td>
      <td>0.59</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Bands with highest unique word counts

<details>
<summary>Show table</summary>
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
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>44097</td>
      <td>41815</td>
      <td>6415</td>
      <td>0.95</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Napalm Death</td>
      <td>Hardcore Punk (early), Grindcore/Death Metal (...</td>
      <td>34363</td>
      <td>20338</td>
      <td>4833</td>
      <td>0.59</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cannibal Corpse</td>
      <td>Death Metal</td>
      <td>32398</td>
      <td>30596</td>
      <td>4377</td>
      <td>0.94</td>
      <td>0.14</td>
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
      <td>17441</td>
      <td>20061</td>
      <td>3945</td>
      <td>1.15</td>
      <td>0.23</td>
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
      <td>Sodom</td>
      <td>Black/Speed Metal (early), Thrash Metal (later)</td>
      <td>34897</td>
      <td>26202</td>
      <td>3741</td>
      <td>0.75</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>16021</td>
      <td>21458</td>
      <td>3730</td>
      <td>1.34</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Tankard</td>
      <td>Thrash Metal</td>
      <td>38493</td>
      <td>30652</td>
      <td>3710</td>
      <td>0.8</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Judas Priest</td>
      <td>Heavy Metal</td>
      <td>51177</td>
      <td>30143</td>
      <td>3506</td>
      <td>0.59</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Bands with highest word density

Again, thrash and death metal bands with short, lyric-heavy songs dominate the words-per-second list.
It's probably not even remotely surprising to tech death fans that [Archspire](https://en.wikipedia.org/wiki/Archspire)
tops this list, with nearly one-and-a-half words per second throughout their discography.

<details>
<summary>Show table</summary>
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
      <td>Archspire</td>
      <td>Technical Death Metal</td>
      <td>5189</td>
      <td>7454</td>
      <td>1970</td>
      <td>1.44</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Municipal Waste</td>
      <td>Thrash Metal/Crossover</td>
      <td>7479</td>
      <td>10587</td>
      <td>2167</td>
      <td>1.42</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Blood Freak</td>
      <td>Death Metal/Grindcore</td>
      <td>4447</td>
      <td>6123</td>
      <td>1778</td>
      <td>1.38</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Acrania</td>
      <td>Brutal Deathcore</td>
      <td>1674</td>
      <td>2282</td>
      <td>902</td>
      <td>1.36</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>16021</td>
      <td>21458</td>
      <td>3730</td>
      <td>1.34</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The Berzerker</td>
      <td>Industrial Death Metal/Grindcore</td>
      <td>8290</td>
      <td>10562</td>
      <td>1383</td>
      <td>1.27</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Animosity</td>
      <td>Death Metal/Metalcore/Grindcore</td>
      <td>4677</td>
      <td>5795</td>
      <td>1302</td>
      <td>1.24</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Absurd</td>
      <td>Black Metal/RAC, Pagan Black Metal</td>
      <td>122</td>
      <td>145</td>
      <td>76</td>
      <td>1.19</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Black Dahlia Murder</td>
      <td>Melodic Death Metal</td>
      <td>17441</td>
      <td>20061</td>
      <td>3945</td>
      <td>1.15</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Exhumed</td>
      <td>Death Metal/Grindcore</td>
      <td>13765</td>
      <td>15669</td>
      <td>3492</td>
      <td>1.14</td>
      <td>0.25</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Bands with highest unique word density

<details>
<summary>Show table</summary>
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
      <td>Absurd</td>
      <td>Black Metal/RAC, Pagan Black Metal</td>
      <td>122</td>
      <td>145</td>
      <td>76</td>
      <td>1.19</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Acrania</td>
      <td>Brutal Deathcore</td>
      <td>1674</td>
      <td>2282</td>
      <td>902</td>
      <td>1.36</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The County Medical Examiners</td>
      <td>Goregrind</td>
      <td>1794</td>
      <td>1662</td>
      <td>932</td>
      <td>0.93</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Peste Noire</td>
      <td>Black Metal</td>
      <td>389</td>
      <td>345</td>
      <td>186</td>
      <td>0.89</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Regurgitate</td>
      <td>Goregrind</td>
      <td>739</td>
      <td>575</td>
      <td>349</td>
      <td>0.78</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Glittertind</td>
      <td>Viking/Folk Metal (early); Indie/Folk Rock (la...</td>
      <td>394</td>
      <td>413</td>
      <td>173</td>
      <td>1.05</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Insect Warfare</td>
      <td>Grindcore</td>
      <td>1253</td>
      <td>1233</td>
      <td>539</td>
      <td>0.98</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Intestinal Disgorge</td>
      <td>Noise/Grindcore (early), Brutal Death Metal/No...</td>
      <td>1493</td>
      <td>1628</td>
      <td>616</td>
      <td>1.09</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Blood Freak</td>
      <td>Death Metal/Grindcore</td>
      <td>4447</td>
      <td>6123</td>
      <td>1778</td>
      <td>1.38</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Archspire</td>
      <td>Technical Death Metal</td>
      <td>5189</td>
      <td>7454</td>
      <td>1970</td>
      <td>1.44</td>
      <td>0.38</td>
    </tr>
  </tbody>
</table>
</div>
</details>


## Word counts among the most popular bands

To pick out the most popular bands, we can filter out artists with fewer than a certain number of reviews.
Plotting out their full-discography unique word counts, we find that there is a generally linear relationship
between the number of unique words and overall discography length, which is not surprising. Cradle of Filth,
however, is a huge outlier, with about twice as many words and unique words in their lyrics than expected.
Opeth seems like an outlier on the flip side, probably due their songs being very heavily instrumental
(Dream Theater probably incorporates more instrumentals but the narrative nature of their lyrics results
in them falling much more in line with heavy/power metal bands). 

<details>
<summary>Show code</summary>
{% highlight python %}
min_reviews = 20

bands_popular = sorted(set(df_r[df_r['album_review_num'] > min_reviews].band_name))
df_r_bands_popular = df_r_bands[df_r_bands.band_name.isin(bands_popular)].set_index('band_name', drop=True)

plt.figure(figsize=(14, 8))
xlist, ylist = [], []
for band, row in df_r_bands_popular.iterrows():
    x = row['band_seconds'] / 3600.0
    y = row['band_word_count'] / 1000.0
    xlist.append(x)
    ylist.append(y)
    plt.plot(x, y, 'r.')

res = linregress(df_r_bands.band_seconds / 3600.0, df_r_bands.band_word_count / 1000.0)
xline = np.linspace(0, df_r_bands_popular.band_seconds.max() / 3600.0)
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
</details>
<br>
    
![png](/assets/images/heavy-metal-lyrics/words_vs_length.png)
<br>
<br>

<details>
<summary>Show code</summary>
{% highlight python %}
plt.figure(figsize=(14, 8))
xlist, ylist = [], []
for band, row in df_r_bands_popular.iterrows():
    x = row['band_seconds'] / 3600.0
    y = row['band_unique_word_count'] / 1000.0
    xlist.append(x)
    ylist.append(y)
    plt.plot(x, y, 'r.')

res = linregress(df_r_bands.band_seconds / 3600.0, df_r_bands.band_unique_word_count / 1000.0)
xline = np.linspace(0, df_r_bands_popular.band_seconds.max() / 3600.0)
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
</details>
<br>
    
![png](/assets/images/heavy-metal-lyrics/unique_words_vs_length.png)


## Ranking artists by the number of unique words in their first 15,000 words

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
"""Copied from https://stackoverflow.com/questions/55005272/get-bounding-boxes-of-individual-elements-of-a-pathcollection-from-plt-scatter
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
{% endhighlight %}


{% highlight python %}
def plot_swarm(data, names):
    fig = plt.figure(figsize=(25, 12))
    ax = sns.swarmplot(x=data, size=30, zorder=1)

    # Get bounding boxes of scatter points
    cs = ax.collections[0]
    boxes = getbb(cs, ax)

    # Add text to circles
    for i, box in enumerate(boxes):
        x = box.x0 + box.width / 2
        y = box.y0 + box.height / 2
        s = names.iloc[i].replace(' ', '\n')
        txt = ax.text(x, y, s, va='center', ha='center')

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

    return fig
{% endhighlight %}


{% highlight python %}
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


{% highlight python %}
num_bands = 100
num_words = 10000

band_filt_words = band_words[band_words['words'].apply(len) > num_words].sort_values('reviews')[-num_bands:]
band_filt_words['unique_first_words'] = band_filt_words['words'].apply(lambda x: len(set(x[:num_words])))
band_filt_words = band_filt_words.sort_values('unique_first_words')
print(len(band_filt_words))

fig = plot_swarm(band_filt_words['unique_first_words'], band_filt_words['name'])
fig.suptitle(f"# of unique words in first {num_words:,.0f} of artist's lyrics", fontsize=25)
plt.show()
{% endhighlight %}
<pre class="code-output">123</pre>
</details>

<br>

![png](/assets/images/heavy-metal-lyrics/swarm_2.png)


## Word counts by genre

Although there are some noticeable trends in the word counts of genres,
overall the distributions of word counts and song lengths per genre are quite broad.
The overlap means lyrical complexity is likely not a sufficient means of distinguishing between genres.
In the next article we'll expand on this, using more sophisticated lexical diversity measures
to quantify the complexity of different genres.

#### Words per song

<details>
<summary>Show code</summary>
{% highlight python %}
df_genre_songs = df_r[['band_name', 'album_name', 'song_name'] + [f"genre_{genre}" for genre in top_genres_1pct]].copy()
df_genre_songs['song_word_count'] = df_r_songs.song_word_count
df_genre_songs['song_seconds'] = df_r_songs.song_seconds

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
violindata = []
for genre in top_genres_1pct:
    df_genre = df_genre_songs[df_genre_songs['genre_' + genre] == 1]
    violindata.append((genre, df_genre['song_word_count']))
violindata.sort(key=lambda x: -np.median(x[1]))
sns.violinplot(data=[x[1] for x in violindata], cut=0, orient='h', color='c')
ax.set_yticklabels([x[0] for x in violindata])
ax.set_xlim(0, 500)
ax.set_title("Words per song")
ax.set_xlabel("word count")
plt.show()
{% endhighlight %}
</details>
<br>

![png](/assets/images/heavy-metal-lyrics/word_count_genre.png)

#### Words per second


<details>
<summary>Show code</summary>
{% highlight python %}
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
violindata = []
for genre in top_genres_1pct:
    df_genre = df_genre_songs[df_genre_songs['genre_' + genre] == 1].copy()
    df_genre['song_words_per_second'] = df_genre['song_word_count'] / df_genre['song_seconds']
    df_genre.loc[df_genre['song_words_per_second'] == np.inf, 'song_words_per_second'] = 0
    violindata.append((genre, df_genre['song_words_per_second']))
violindata.sort(key=lambda x: -np.median(x[1]))
sns.violinplot(data=[x[1] for x in violindata], cut=0, orient='h', color='c')
ax.set_yticklabels([x[0] for x in violindata])
ax.set_title("Words per second")
ax.set_xlabel("word count")
plt.show()
{% endhighlight %}
</details>
<br>

![png](/assets/images/heavy-metal-lyrics/word_rate_genre.png)
    
#### Scatter plot

<details>
<summary>Show code</summary>
{% highlight python %}
plt.figure(figsize=(14, 8))
xlist, ylist = [], []
for genre in top_genres_1pct:
    df_genre = df_genre_songs[df_genre_songs['genre_' + genre] == 1].copy()
    x = df_genre['song_seconds'].mean() / 60.0
    y = df_genre['song_word_count'].mean()
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
</details>
<br>

![png](/assets/images/heavy-metal-lyrics/words_vs_length_genre.png)


---
layout: post
title: "Analysis of heavy metal lyrics - Part 1: Overview"
categories: jekyll update
hidden: true
---

<pre style="margin-left: 50px; margin-right: 50px; font-size: 13px">
Explicit/NSFW content warning: this project features examples of heavy metal lyrics and song/album/band names.
These often contain words and themes that some may find offensive/inappropriate.
</pre>


This article is a part of my [heavy metal lyrics project](/jekyll/update/2022/01/19/heavy-metal-lyrics.html).
It provides a top-level overview of the song lyrics dataset.
Below is a dashboard I've put together
([click here for full-size version](https://metal-lyrics-feature-plots.herokuapp.com/){:target="_blank"})
to visualize the complete dataset using the metrics discussed here and in the other articles.
Since I did the analyses here before building the dashboard, some plots will look different.
If you're interested in seeing the full code (a lot is omitted here), check out the
[original notebook](https://github.com/pdqnguyen/metallyrics/blob/main/analyses/lyrics/notebooks/lyrics-part-1-overview.ipynb).
In the [next article](/jekyll/update/2022/01/19/lyrics-part-2-lexical-diversity.html)
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
* Using Matt Daniels' metric (unique words in first X words), a few bands make up the top end of the distribution:
Cryptopsy, Napalm Death, Cattle Decapitation, Cradle of Filth, Deeds of Flesh, Dying Fetus, and Exhumed.
(The plot here has fewer bands shown than in the dashboard, in order to fit all the names on the figure.)
* <span style="color:#ebc634; font-weight:bold">Word count distributions are correlated with genres in the way you'd expect, but the stylistic diversity in each
genre blurs that correlation, suggesting that attempts at lyrics-based genre classification are going to be a challenge.</span>


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
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>album_name</th>
      <th>song_name</th>
      <th>band_genre</th>
      <th>song_word_count</th>
      <th>song_seconds</th>
      <th>song_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11862</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>2259</td>
      <td>358</td>
      <td>6.310056</td>
    </tr>
    <tr>
      <th>35657</th>
      <td>Edge of Sanity</td>
      <td>Crimson</td>
      <td>Crimson</td>
      <td>Progressive Death Metal</td>
      <td>1948</td>
      <td>2400</td>
      <td>0.811667</td>
    </tr>
    <tr>
      <th>100430</th>
      <td>Theocracy</td>
      <td>Mirror of Souls</td>
      <td>Mirror of Souls</td>
      <td>Epic Progressive Power Metal</td>
      <td>1556</td>
      <td>1346</td>
      <td>1.156018</td>
    </tr>
    <tr>
      <th>11822</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>To Dethrone the Witch-Queen of Mytos K'unn (Th...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1306</td>
      <td>405</td>
      <td>3.224691</td>
    </tr>
    <tr>
      <th>11866</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Unfettering the Hoary Sentinels of Karnak</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1237</td>
      <td>262</td>
      <td>4.721374</td>
    </tr>
    <tr>
      <th>11838</th>
      <td>Bal-Sagoth</td>
      <td>Battle Magic</td>
      <td>Blood Slakes the Sand at the Circus Maximus</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1186</td>
      <td>533</td>
      <td>2.225141</td>
    </tr>
    <tr>
      <th>81809</th>
      <td>Redemption</td>
      <td>Redemption</td>
      <td>Something Wicked This Way Comes</td>
      <td>Progressive Metal</td>
      <td>1114</td>
      <td>1466</td>
      <td>0.759891</td>
    </tr>
    <tr>
      <th>15057</th>
      <td>Blind Guardian</td>
      <td>A Night at the Opera</td>
      <td>And Then There Was Silence</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>1037</td>
      <td>846</td>
      <td>1.225768</td>
    </tr>
    <tr>
      <th>45056</th>
      <td>Green Carnation</td>
      <td>Light of Day, Day of Darkness</td>
      <td>Light of Day, Day of Darkness</td>
      <td>Death Metal (early); Gothic/Progressive Metal/...</td>
      <td>1028</td>
      <td>3606</td>
      <td>0.285080</td>
    </tr>
  </tbody>
</table>
</div>
</details>

#### Songs with highest unique word counts

<details>
<summary>Show table</summary>
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>album_name</th>
      <th>song_name</th>
      <th>band_genre</th>
      <th>song_word_count</th>
      <th>song_unique_word_count</th>
      <th>song_seconds</th>
      <th>song_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11862</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>2259</td>
      <td>897</td>
      <td>358</td>
      <td>2.505587</td>
    </tr>
    <tr>
      <th>35657</th>
      <td>Edge of Sanity</td>
      <td>Crimson</td>
      <td>Crimson</td>
      <td>Progressive Death Metal</td>
      <td>1948</td>
      <td>658</td>
      <td>2400</td>
      <td>0.274167</td>
    </tr>
    <tr>
      <th>11866</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Unfettering the Hoary Sentinels of Karnak</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1237</td>
      <td>560</td>
      <td>262</td>
      <td>2.137405</td>
    </tr>
    <tr>
      <th>11822</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>To Dethrone the Witch-Queen of Mytos K'unn (Th...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1306</td>
      <td>548</td>
      <td>405</td>
      <td>1.353086</td>
    </tr>
    <tr>
      <th>11838</th>
      <td>Bal-Sagoth</td>
      <td>Battle Magic</td>
      <td>Blood Slakes the Sand at the Circus Maximus</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1186</td>
      <td>530</td>
      <td>533</td>
      <td>0.994371</td>
    </tr>
    <tr>
      <th>100430</th>
      <td>Theocracy</td>
      <td>Mirror of Souls</td>
      <td>Mirror of Souls</td>
      <td>Epic Progressive Power Metal</td>
      <td>1556</td>
      <td>457</td>
      <td>1346</td>
      <td>0.339525</td>
    </tr>
    <tr>
      <th>81809</th>
      <td>Redemption</td>
      <td>Redemption</td>
      <td>Something Wicked This Way Comes</td>
      <td>Progressive Metal</td>
      <td>1114</td>
      <td>439</td>
      <td>1466</td>
      <td>0.299454</td>
    </tr>
    <tr>
      <th>11826</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>The Splendour of a Thousand Swords Gleaming Be...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>977</td>
      <td>429</td>
      <td>363</td>
      <td>1.181818</td>
    </tr>
    <tr>
      <th>11824</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1018</td>
      <td>427</td>
      <td>443</td>
      <td>0.963883</td>
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
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>album_name</th>
      <th>song_name</th>
      <th>band_genre</th>
      <th>song_word_count</th>
      <th>song_seconds</th>
      <th>song_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11862</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>2259</td>
      <td>358</td>
      <td>6.310056</td>
    </tr>
    <tr>
      <th>103209</th>
      <td>Trans-Siberian Orchestra</td>
      <td>The Christmas Attic</td>
      <td>The Ghosts of Christmas Eve</td>
      <td>Orchestral/Progressive Rock/Metal</td>
      <td>815</td>
      <td>135</td>
      <td>6.037037</td>
    </tr>
    <tr>
      <th>61709</th>
      <td>Macabre</td>
      <td>Gloom</td>
      <td>I Need to Kill</td>
      <td>Thrash/Death Metal/Grindcore</td>
      <td>199</td>
      <td>36</td>
      <td>5.527778</td>
    </tr>
    <tr>
      <th>11866</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Unfettering the Hoary Sentinels of Karnak</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1237</td>
      <td>262</td>
      <td>4.721374</td>
    </tr>
    <tr>
      <th>80005</th>
      <td>Putrid Pile</td>
      <td>Paraphiliac Perversions</td>
      <td>Toxic Shock Therapy</td>
      <td>Brutal Death Metal</td>
      <td>18</td>
      <td>4</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>84364</th>
      <td>S.O.D.</td>
      <td>Bigger than the Devil</td>
      <td>Charlie Don't Cheat</td>
      <td>Hardcore/Crossover/Thrash Metal</td>
      <td>105</td>
      <td>25</td>
      <td>4.200000</td>
    </tr>
    <tr>
      <th>70012</th>
      <td>Napalm Death</td>
      <td>Scum</td>
      <td>You Suffer</td>
      <td>Hardcore Punk (early), Grindcore/Death Metal (...</td>
      <td>4</td>
      <td>1</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>85949</th>
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Welcome</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>490</td>
      <td>131</td>
      <td>3.740458</td>
    </tr>
    <tr>
      <th>23586</th>
      <td>Circle of Dead Children</td>
      <td>Human Harvest</td>
      <td>White Trash Headache</td>
      <td>Brutal Death Metal, Grindcore</td>
      <td>21</td>
      <td>6</td>
      <td>3.500000</td>
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
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>album_name</th>
      <th>song_name</th>
      <th>band_genre</th>
      <th>song_word_count</th>
      <th>song_unique_word_count</th>
      <th>song_seconds</th>
      <th>song_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70012</th>
      <td>Napalm Death</td>
      <td>Scum</td>
      <td>You Suffer</td>
      <td>Hardcore Punk (early), Grindcore/Death Metal (...</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>111972</th>
      <td>Wormrot</td>
      <td>Dirge</td>
      <td>You Suffer but Why Is It My Problem</td>
      <td>Grindcore</td>
      <td>14</td>
      <td>14</td>
      <td>4</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>23586</th>
      <td>Circle of Dead Children</td>
      <td>Human Harvest</td>
      <td>White Trash Headache</td>
      <td>Brutal Death Metal, Grindcore</td>
      <td>21</td>
      <td>20</td>
      <td>6</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>84364</th>
      <td>S.O.D.</td>
      <td>Bigger than the Devil</td>
      <td>Charlie Don't Cheat</td>
      <td>Hardcore/Crossover/Thrash Metal</td>
      <td>105</td>
      <td>74</td>
      <td>25</td>
      <td>2.960000</td>
    </tr>
    <tr>
      <th>68940</th>
      <td>Municipal Waste</td>
      <td>Waste 'Em All</td>
      <td>I Want to Kill the President</td>
      <td>Thrash Metal/Crossover</td>
      <td>54</td>
      <td>44</td>
      <td>17</td>
      <td>2.588235</td>
    </tr>
    <tr>
      <th>52038</th>
      <td>Insect Warfare</td>
      <td>World Extermination</td>
      <td>Street Sweeper</td>
      <td>Grindcore</td>
      <td>43</td>
      <td>33</td>
      <td>13</td>
      <td>2.538462</td>
    </tr>
    <tr>
      <th>11862</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>2259</td>
      <td>897</td>
      <td>358</td>
      <td>2.505587</td>
    </tr>
    <tr>
      <th>24474</th>
      <td>Corrosion of Conformity</td>
      <td>Eye for an Eye</td>
      <td>No Drunk</td>
      <td>Crossover/Sludge/Southern Metal</td>
      <td>74</td>
      <td>52</td>
      <td>22</td>
      <td>2.363636</td>
    </tr>
    <tr>
      <th>30283</th>
      <td>Deliverance</td>
      <td>What a Joke</td>
      <td>Happy Star</td>
      <td>Speed/Thrash Metal, Industrial</td>
      <td>7</td>
      <td>7</td>
      <td>3</td>
      <td>2.333333</td>
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
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>album_name</th>
      <th>band_genre</th>
      <th>album_word_count</th>
      <th>album_unique_word_count</th>
      <th>album_seconds</th>
      <th>album_words_per_second</th>
      <th>album_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>882</th>
      <td>Blind Guardian</td>
      <td>Twilight Orchestra: Legacy of the Dark Lands</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>8812</td>
      <td>4361</td>
      <td>8210</td>
      <td>1.073325</td>
      <td>0.531181</td>
    </tr>
    <tr>
      <th>645</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6979</td>
      <td>3437</td>
      <td>3639</td>
      <td>1.917835</td>
      <td>0.944490</td>
    </tr>
    <tr>
      <th>644</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6500</td>
      <td>2959</td>
      <td>3157</td>
      <td>2.058917</td>
      <td>0.937282</td>
    </tr>
    <tr>
      <th>5257</th>
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>5264</td>
      <td>2268</td>
      <td>3218</td>
      <td>1.635799</td>
      <td>0.704786</td>
    </tr>
    <tr>
      <th>628</th>
      <td>Ayreon</td>
      <td>The Human Equation</td>
      <td>Progressive Metal/Rock</td>
      <td>4917</td>
      <td>2057</td>
      <td>5950</td>
      <td>0.826387</td>
      <td>0.345714</td>
    </tr>
    <tr>
      <th>6118</th>
      <td>Therion</td>
      <td>Beloved Antichrist</td>
      <td>Death Metal (early), Symphonic/Operatic Metal ...</td>
      <td>4859</td>
      <td>3026</td>
      <td>9110</td>
      <td>0.533370</td>
      <td>0.332162</td>
    </tr>
    <tr>
      <th>6250</th>
      <td>Trans-Siberian Orchestra</td>
      <td>The Christmas Attic</td>
      <td>Orchestral/Progressive Rock/Metal</td>
      <td>4794</td>
      <td>2035</td>
      <td>4066</td>
      <td>1.179046</td>
      <td>0.500492</td>
    </tr>
    <tr>
      <th>872</th>
      <td>Blind Guardian</td>
      <td>A Night at the Opera</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>4630</td>
      <td>1730</td>
      <td>4024</td>
      <td>1.150596</td>
      <td>0.429920</td>
    </tr>
    <tr>
      <th>3696</th>
      <td>Machine Head</td>
      <td>Catharsis</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>4623</td>
      <td>2077</td>
      <td>4457</td>
      <td>1.037245</td>
      <td>0.466009</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Albums with highest unique word counts

<details>
<summary>Show table</summary>
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>album_name</th>
      <th>band_genre</th>
      <th>album_word_count</th>
      <th>album_unique_word_count</th>
      <th>album_seconds</th>
      <th>album_words_per_second</th>
      <th>album_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>882</th>
      <td>Blind Guardian</td>
      <td>Twilight Orchestra: Legacy of the Dark Lands</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>8812</td>
      <td>4361</td>
      <td>8210</td>
      <td>1.073325</td>
      <td>0.531181</td>
    </tr>
    <tr>
      <th>645</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6979</td>
      <td>3437</td>
      <td>3639</td>
      <td>1.917835</td>
      <td>0.944490</td>
    </tr>
    <tr>
      <th>6118</th>
      <td>Therion</td>
      <td>Beloved Antichrist</td>
      <td>Death Metal (early), Symphonic/Operatic Metal ...</td>
      <td>4859</td>
      <td>3026</td>
      <td>9110</td>
      <td>0.533370</td>
      <td>0.332162</td>
    </tr>
    <tr>
      <th>644</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6500</td>
      <td>2959</td>
      <td>3157</td>
      <td>2.058917</td>
      <td>0.937282</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>Dream Theater</td>
      <td>The Astonishing</td>
      <td>Progressive Metal</td>
      <td>4366</td>
      <td>2682</td>
      <td>7098</td>
      <td>0.615103</td>
      <td>0.377853</td>
    </tr>
    <tr>
      <th>1020</th>
      <td>Cage</td>
      <td>Ancient Evil</td>
      <td>Heavy/Power Metal</td>
      <td>4569</td>
      <td>2551</td>
      <td>4477</td>
      <td>1.020549</td>
      <td>0.569801</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>Cradle of Filth</td>
      <td>Darkly, Darkly, Venus Aversa</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>4235</td>
      <td>2394</td>
      <td>3462</td>
      <td>1.223281</td>
      <td>0.691508</td>
    </tr>
    <tr>
      <th>894</th>
      <td>Blood Freak</td>
      <td>Live Fast, Die Young... and Leave a Flesh-Eati...</td>
      <td>Death Metal/Grindcore</td>
      <td>3575</td>
      <td>2353</td>
      <td>2558</td>
      <td>1.397576</td>
      <td>0.919859</td>
    </tr>
    <tr>
      <th>1343</th>
      <td>Cradle of Filth</td>
      <td>Damnation and a Day</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>3836</td>
      <td>2306</td>
      <td>3995</td>
      <td>0.960200</td>
      <td>0.577222</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Albums with highest word density

<details>
<summary>Show table</summary>
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>album_name</th>
      <th>band_genre</th>
      <th>album_word_count</th>
      <th>album_unique_word_count</th>
      <th>album_seconds</th>
      <th>album_words_per_second</th>
      <th>album_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>644</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6500</td>
      <td>2959</td>
      <td>3157</td>
      <td>2.058917</td>
      <td>0.937282</td>
    </tr>
    <tr>
      <th>3909</th>
      <td>Melvins</td>
      <td>Prick</td>
      <td>Sludge Metal, Various</td>
      <td>504</td>
      <td>193</td>
      <td>257</td>
      <td>1.961089</td>
      <td>0.750973</td>
    </tr>
    <tr>
      <th>645</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6979</td>
      <td>3437</td>
      <td>3639</td>
      <td>1.917835</td>
      <td>0.944490</td>
    </tr>
    <tr>
      <th>4144</th>
      <td>Municipal Waste</td>
      <td>Waste 'Em All</td>
      <td>Thrash Metal/Crossover</td>
      <td>1615</td>
      <td>1043</td>
      <td>848</td>
      <td>1.904481</td>
      <td>1.229953</td>
    </tr>
    <tr>
      <th>4561</th>
      <td>Origin</td>
      <td>Informis Infinitas Inhumanitas</td>
      <td>Technical Brutal Death Metal</td>
      <td>3022</td>
      <td>1605</td>
      <td>1712</td>
      <td>1.765187</td>
      <td>0.937500</td>
    </tr>
    <tr>
      <th>428</th>
      <td>Archspire</td>
      <td>Relentless Mutation</td>
      <td>Technical Death Metal</td>
      <td>3158</td>
      <td>1502</td>
      <td>1837</td>
      <td>1.719107</td>
      <td>0.817637</td>
    </tr>
    <tr>
      <th>3410</th>
      <td>Korpiklaani</td>
      <td>Noita</td>
      <td>Folk Metal</td>
      <td>293</td>
      <td>90</td>
      <td>178</td>
      <td>1.646067</td>
      <td>0.505618</td>
    </tr>
    <tr>
      <th>5257</th>
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>5264</td>
      <td>2268</td>
      <td>3218</td>
      <td>1.635799</td>
      <td>0.704786</td>
    </tr>
    <tr>
      <th>4140</th>
      <td>Municipal Waste</td>
      <td>Hazardous Mutation</td>
      <td>Thrash Metal/Crossover</td>
      <td>2246</td>
      <td>1453</td>
      <td>1425</td>
      <td>1.576140</td>
      <td>1.019649</td>
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
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>album_name</th>
      <th>band_genre</th>
      <th>album_word_count</th>
      <th>album_unique_word_count</th>
      <th>album_seconds</th>
      <th>album_words_per_second</th>
      <th>album_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4144</th>
      <td>Municipal Waste</td>
      <td>Waste 'Em All</td>
      <td>Thrash Metal/Crossover</td>
      <td>1615</td>
      <td>1043</td>
      <td>848</td>
      <td>1.904481</td>
      <td>1.229953</td>
    </tr>
    <tr>
      <th>4140</th>
      <td>Municipal Waste</td>
      <td>Hazardous Mutation</td>
      <td>Thrash Metal/Crossover</td>
      <td>2246</td>
      <td>1453</td>
      <td>1425</td>
      <td>1.576140</td>
      <td>1.019649</td>
    </tr>
    <tr>
      <th>2702</th>
      <td>Haggard</td>
      <td>Tales of Ithiria</td>
      <td>Progressive Death Metal (early); Classical/Orc...</td>
      <td>299</td>
      <td>228</td>
      <td>240</td>
      <td>1.245833</td>
      <td>0.950000</td>
    </tr>
    <tr>
      <th>645</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6979</td>
      <td>3437</td>
      <td>3639</td>
      <td>1.917835</td>
      <td>0.944490</td>
    </tr>
    <tr>
      <th>4561</th>
      <td>Origin</td>
      <td>Informis Infinitas Inhumanitas</td>
      <td>Technical Brutal Death Metal</td>
      <td>3022</td>
      <td>1605</td>
      <td>1712</td>
      <td>1.765187</td>
      <td>0.937500</td>
    </tr>
    <tr>
      <th>644</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6500</td>
      <td>2959</td>
      <td>3157</td>
      <td>2.058917</td>
      <td>0.937282</td>
    </tr>
    <tr>
      <th>894</th>
      <td>Blood Freak</td>
      <td>Live Fast, Die Young... and Leave a Flesh-Eati...</td>
      <td>Death Metal/Grindcore</td>
      <td>3575</td>
      <td>2353</td>
      <td>2558</td>
      <td>1.397576</td>
      <td>0.919859</td>
    </tr>
    <tr>
      <th>5579</th>
      <td>Soilent Green</td>
      <td>Confrontation</td>
      <td>Sludge/Death Metal/Grindcore</td>
      <td>2511</td>
      <td>1584</td>
      <td>1730</td>
      <td>1.451445</td>
      <td>0.915607</td>
    </tr>
    <tr>
      <th>3132</th>
      <td>Intestinal Disgorge</td>
      <td>Vagina</td>
      <td>Noise/Grindcore (early), Brutal Death Metal/No...</td>
      <td>1628</td>
      <td>1274</td>
      <td>1493</td>
      <td>1.090422</td>
      <td>0.853315</td>
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
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>band_genre</th>
      <th>band_word_count</th>
      <th>band_unique_word_count</th>
      <th>band_seconds</th>
      <th>band_words_per_second</th>
      <th>band_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>281</th>
      <td>Cradle of Filth</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>41815</td>
      <td>24367</td>
      <td>44097</td>
      <td>0.948250</td>
      <td>0.552577</td>
    </tr>
    <tr>
      <th>1040</th>
      <td>Saxon</td>
      <td>NWOBHM, Heavy Metal</td>
      <td>36759</td>
      <td>14778</td>
      <td>53755</td>
      <td>0.683825</td>
      <td>0.274914</td>
    </tr>
    <tr>
      <th>650</th>
      <td>Iron Maiden</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>34843</td>
      <td>15254</td>
      <td>52673</td>
      <td>0.661496</td>
      <td>0.289598</td>
    </tr>
    <tr>
      <th>181</th>
      <td>Blind Guardian</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>34836</td>
      <td>15901</td>
      <td>38090</td>
      <td>0.914571</td>
      <td>0.417459</td>
    </tr>
    <tr>
      <th>973</th>
      <td>Rage</td>
      <td>Heavy/Speed/Power Metal</td>
      <td>34314</td>
      <td>16389</td>
      <td>56064</td>
      <td>0.612051</td>
      <td>0.292327</td>
    </tr>
    <tr>
      <th>915</th>
      <td>Overkill</td>
      <td>Thrash Metal; Thrash/Groove Metal</td>
      <td>32485</td>
      <td>14982</td>
      <td>47540</td>
      <td>0.683319</td>
      <td>0.315145</td>
    </tr>
    <tr>
      <th>578</th>
      <td>Helloween</td>
      <td>Power/Speed Metal</td>
      <td>32472</td>
      <td>14181</td>
      <td>48991</td>
      <td>0.662816</td>
      <td>0.289461</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>Tankard</td>
      <td>Thrash Metal</td>
      <td>30652</td>
      <td>13472</td>
      <td>38493</td>
      <td>0.796301</td>
      <td>0.349986</td>
    </tr>
    <tr>
      <th>225</th>
      <td>Cannibal Corpse</td>
      <td>Death Metal</td>
      <td>30596</td>
      <td>16355</td>
      <td>32398</td>
      <td>0.944379</td>
      <td>0.504815</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Bands with highest unique word counts

<details>
<summary>Show table</summary>
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>band_genre</th>
      <th>band_word_count</th>
      <th>band_unique_word_count</th>
      <th>band_seconds</th>
      <th>band_words_per_second</th>
      <th>band_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>281</th>
      <td>Cradle of Filth</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>41815</td>
      <td>24367</td>
      <td>44097</td>
      <td>0.948250</td>
      <td>0.552577</td>
    </tr>
    <tr>
      <th>973</th>
      <td>Rage</td>
      <td>Heavy/Speed/Power Metal</td>
      <td>34314</td>
      <td>16389</td>
      <td>56064</td>
      <td>0.612051</td>
      <td>0.292327</td>
    </tr>
    <tr>
      <th>225</th>
      <td>Cannibal Corpse</td>
      <td>Death Metal</td>
      <td>30596</td>
      <td>16355</td>
      <td>32398</td>
      <td>0.944379</td>
      <td>0.504815</td>
    </tr>
    <tr>
      <th>181</th>
      <td>Blind Guardian</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>34836</td>
      <td>15901</td>
      <td>38090</td>
      <td>0.914571</td>
      <td>0.417459</td>
    </tr>
    <tr>
      <th>650</th>
      <td>Iron Maiden</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>34843</td>
      <td>15254</td>
      <td>52673</td>
      <td>0.661496</td>
      <td>0.289598</td>
    </tr>
    <tr>
      <th>664</th>
      <td>Judas Priest</td>
      <td>Heavy Metal</td>
      <td>30143</td>
      <td>15078</td>
      <td>51177</td>
      <td>0.588995</td>
      <td>0.294625</td>
    </tr>
    <tr>
      <th>915</th>
      <td>Overkill</td>
      <td>Thrash Metal; Thrash/Groove Metal</td>
      <td>32485</td>
      <td>14982</td>
      <td>47540</td>
      <td>0.683319</td>
      <td>0.315145</td>
    </tr>
    <tr>
      <th>1040</th>
      <td>Saxon</td>
      <td>NWOBHM, Heavy Metal</td>
      <td>36759</td>
      <td>14778</td>
      <td>53755</td>
      <td>0.683825</td>
      <td>0.274914</td>
    </tr>
    <tr>
      <th>578</th>
      <td>Helloween</td>
      <td>Power/Speed Metal</td>
      <td>32472</td>
      <td>14181</td>
      <td>48991</td>
      <td>0.662816</td>
      <td>0.289461</td>
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
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>band_genre</th>
      <th>band_word_count</th>
      <th>band_unique_word_count</th>
      <th>band_seconds</th>
      <th>band_words_per_second</th>
      <th>band_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>Archspire</td>
      <td>Technical Death Metal</td>
      <td>7454</td>
      <td>3948</td>
      <td>5189</td>
      <td>1.436500</td>
      <td>0.760840</td>
    </tr>
    <tr>
      <th>829</th>
      <td>Municipal Waste</td>
      <td>Thrash Metal/Crossover</td>
      <td>10587</td>
      <td>6370</td>
      <td>7479</td>
      <td>1.415564</td>
      <td>0.851718</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Blood Freak</td>
      <td>Death Metal/Grindcore</td>
      <td>6123</td>
      <td>3899</td>
      <td>4447</td>
      <td>1.376883</td>
      <td>0.876771</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Acrania</td>
      <td>Brutal Deathcore</td>
      <td>2282</td>
      <td>1309</td>
      <td>1674</td>
      <td>1.363202</td>
      <td>0.781959</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>21458</td>
      <td>10700</td>
      <td>16021</td>
      <td>1.339367</td>
      <td>0.667873</td>
    </tr>
    <tr>
      <th>1154</th>
      <td>The Berzerker</td>
      <td>Industrial Death Metal/Grindcore</td>
      <td>10562</td>
      <td>4456</td>
      <td>8290</td>
      <td>1.274065</td>
      <td>0.537515</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Animosity</td>
      <td>Death Metal/Metalcore/Grindcore</td>
      <td>5795</td>
      <td>3261</td>
      <td>4677</td>
      <td>1.239042</td>
      <td>0.697242</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Absurd</td>
      <td>Black Metal/RAC, Pagan Black Metal</td>
      <td>145</td>
      <td>76</td>
      <td>122</td>
      <td>1.188525</td>
      <td>0.622951</td>
    </tr>
    <tr>
      <th>1155</th>
      <td>The Black Dahlia Murder</td>
      <td>Melodic Death Metal</td>
      <td>20061</td>
      <td>11010</td>
      <td>17441</td>
      <td>1.150221</td>
      <td>0.631271</td>
    </tr>
  </tbody>
</table>
</div>
</details>


#### Bands with highest unique word density

<details>
<summary>Show table</summary>
<div class="table-wrapper" markdown="block">
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
      <th>band_name</th>
      <th>band_genre</th>
      <th>band_word_count</th>
      <th>band_unique_word_count</th>
      <th>band_seconds</th>
      <th>band_words_per_second</th>
      <th>band_unique_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>187</th>
      <td>Blood Freak</td>
      <td>Death Metal/Grindcore</td>
      <td>6123</td>
      <td>3899</td>
      <td>4447</td>
      <td>1.376883</td>
      <td>0.876771</td>
    </tr>
    <tr>
      <th>642</th>
      <td>Intestinal Disgorge</td>
      <td>Noise/Grindcore (early), Brutal Death Metal/No...</td>
      <td>1628</td>
      <td>1274</td>
      <td>1493</td>
      <td>1.090422</td>
      <td>0.853315</td>
    </tr>
    <tr>
      <th>829</th>
      <td>Municipal Waste</td>
      <td>Thrash Metal/Crossover</td>
      <td>10587</td>
      <td>6370</td>
      <td>7479</td>
      <td>1.415564</td>
      <td>0.851718</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Acrania</td>
      <td>Brutal Deathcore</td>
      <td>2282</td>
      <td>1309</td>
      <td>1674</td>
      <td>1.363202</td>
      <td>0.781959</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Archspire</td>
      <td>Technical Death Metal</td>
      <td>7454</td>
      <td>3948</td>
      <td>5189</td>
      <td>1.436500</td>
      <td>0.760840</td>
    </tr>
    <tr>
      <th>635</th>
      <td>Insect Warfare</td>
      <td>Grindcore</td>
      <td>1233</td>
      <td>950</td>
      <td>1253</td>
      <td>0.984038</td>
      <td>0.758180</td>
    </tr>
    <tr>
      <th>1338</th>
      <td>Wormrot</td>
      <td>Grindcore</td>
      <td>1638</td>
      <td>1292</td>
      <td>1709</td>
      <td>0.958455</td>
      <td>0.755998</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>Soilent Green</td>
      <td>Sludge/Death Metal/Grindcore</td>
      <td>12320</td>
      <td>7920</td>
      <td>11229</td>
      <td>1.097159</td>
      <td>0.705317</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Animosity</td>
      <td>Death Metal/Metalcore/Grindcore</td>
      <td>5795</td>
      <td>3261</td>
      <td>4677</td>
      <td>1.239042</td>
      <td>0.697242</td>
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
    
### Scatter plot

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

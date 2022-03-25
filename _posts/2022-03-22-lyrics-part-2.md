---
layout: post
date: 2022-03-22
title: "Analysis of Heavy Metal Lyrics - Part 2: Lexical Diversity"
categories: jekyll update
permalink: /projects/heavy-metal-analysis/lyrics-part-2
summary: |
  Comparison of lexical diversity measures and what they tell us about artists and genres.
---

This article is the second part of the lyrical analysis [heavy metal lyrics](/projects/heavy-metal-analysis.html).
Below is the same dashboard presented in [Part 1](./lyrics-part-1-overview.html)
([click here for full-size version](https://metal-lyrics-feature-plots.herokuapp.com/){:target="_blank"}).
Here we will look at the lexical diversity measures included among the plot options (e.g. TTR, MTLD, and vocd-D).
If you're interested in seeing the full code (a lot is omitted here), check out the
[original notebook](https://github.com/pdqnguyen/metallyrics/blob/main/analyses/lyrics/notebooks/lyrics-part-2-lexical-diversity.ipynb){:target="_blank"}.
In the [next article](./lyrics-part-3.html) we'll use word clouds to describe the different genres.

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

* We will use lexical diversity measures like [MTLD](https://doi.org/10.3758/BRM.42.2.381) and
[vocd-D](https://www.palgrave.com/gp/book/9781403902313) to quantify the complexity of heavy metal lyrics.
* We will compare lexical diversity across songs, bands, and genres.


## Table of Contents
2. [Loading and pre-processing data](#loading-and-pre-processing-data)
3. [Lexical diversity measures](#lexical-diversity-measures)
   1. [MTLD](#mtld)
   2. [vocd-D](#vocd-d)
   3. [HD-D](#hd-d)
   4. [vocd-D or HD-D](#vocd-d-or-hd-d)
4. [Histograms](#histograms)
5. [Ranking songs](#songs-ranked-by-lexical-diversity)
6. [Ranking bands](#bands-ranked-by-lexical-diversity)
7. [Ranking genres](#genres-ranked-by-lexical-diversity)


## Loading and pre-processing data


<details>
<summary>Show code</summary>
{% highlight python %}
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.style.use('seaborn')
import seaborn as sns
sns.set(font_scale=2)
from nltk.corpus import words as nltk_words
from scipy.optimize import curve_fit
from scipy.stats import hypergeom

import sys
sys.path.append('../scripts/')
from nlp import tokenize
{% endhighlight %}


{% highlight python %}
def get_genres(data):
    columns = [col for col in data.columns if 'genre_' in col]
    genres = [re.sub(r"^genre\_", "", col) for col in columns]
    return genres, columns


def get_bands(data):
    genres, genre_cols = get_genres(data)
    # Combine songs from same band
    band_genres = df.groupby('band_name')['band_genre'].first()
    band_labels = data.groupby('band_name')[genre_cols].max()
    band_lyrics = data.groupby('band_name')['song_darklyrics'].sum()
    bands = pd.concat((band_genres, band_labels, band_lyrics), axis=1)
    bands.index.name = 'band'
    bands.columns = ['genre'] + genres + ['lyrics']
    bands['words'] = bands['lyrics'].apply(tokenize)
    return bands


def get_songs(data):
    genres, genre_cols = get_genres(data)
    songs = data[['band_name', 'band_genre', 'song_name'] + genre_cols + ['song_darklyrics']].copy()
    songs.columns = ['band', 'genre', 'song'] + genres + ['lyrics']
    songs['words'] = songs['lyrics'].apply(tokenize)
    return songs
{% endhighlight %}


{% highlight python %}
df = pd.read_csv('../songs-10pct.csv')
genres, genre_cols = get_genres(df)
df_bands = get_bands(df)
df_songs = get_songs(df)
{% endhighlight %}
</details><br>


## Lexical diversity measures

One simple approach to quantifying lexical diversity is to divide the number of unique words (types, $$V$$) by the total word count (tokens, $$N$$). This type-token ratio or TTR is heavily biased toward short texts, since longer texts are more likely to repeat tokens without necessarily diminishing complexity. A few ways exist for rescaling the relationship to reduce this bias; for this notebook the root-TTR and log-TTR are used:

$$
\begin{split}
&LD_{TTR} &= \frac{V}{N} &\hspace{1cm} (\textrm{type-token ratio})\\
&LD_{rootTTR} &= \frac{V}{\sqrt{N}} &\hspace{1cm} (\textrm{root type-token ratio})\\
&LD_{logTTR} &= \frac{\log{V}}{\log{N}} &\hspace{1cm} (\textrm{logarithmic type-token ratio})\\
\end{split}
$$

#### MTLD

More sophisticated approaches look at how types are distributed in the text. The bluntly named Measure of Textual Lexical Diversity (MTLD), described by [McCarthy and Jarvis (2010)](https://doi.org/10.3758/BRM.42.2.381), is based on the mean length of token sequences in the text that exceed a certain TTR threshold. The algorithm begins with a sequence consisting of the first token in the text, and iteratively adds the following token, each time recomputing the TTR of the sequence so far. Once the sequence TTR drops below the pre-determined threshold, the sequence ends and a new sequence begins at the next token. This continues until the end of the text is reached, at which point the mean sequence length is computed. The process is repeated from the last token, going backwards, to produce another mean sequence length. The mean of these two results is the final MTLD figure.

Unlike the simpler methods, MTLD has a tunable parameter. The TTR threshold is chosen by the authors to be 0.720, which is approximately where the cumulative TTR curves for texts in the Project Gutenburg Text Archives reached a point of stabilization. The same can be done with the DarkLyrics data by plotting cumulative TTR values for a large number of bands and identifying the point of stabilization. This cannot be done with single-song lyrics since refrains in the lyrics heavily warp the cumulative TTR curves, such that most never stabilize. Unfortunately even when looking at band lyrics, the cumulative TTR does not stabilize very well, as the curves seem to continue decaying well into the thousands of tokens. However one can roughly identify a point of stabilization somewhere around a TTR of 0.5, occuring at about 200 tokens, so this is used as the threshold for MTLD.


<details>
<summary>Show code</summary>
{% highlight python %}
def TTR(x):
    return len(set(x)) / len(x)

def cumulative_TTR(words):
    out = [TTR(words[: i + 1]) for i in range(len(words))]
    return out

for i in range(0, 1000, 10):
    plt.plot(cumulative_TTR(df_bands.iloc[i].words[:2000]), alpha=0.3)
plt.xlabel('Tokens')
plt.ylabel('Cumulative TTR')
plt.show()
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics2/lyrics-part-2-lexical-diversity_9_0.png)

    



<details>
<summary>Show code</summary>
{% highlight python %}
def MTLD_forward(words, threshold):
    factor = 0
    segment = []
    i = 0
    while i < len(words):
        segment.append(words[i])
        segTTR = TTR(segment)
        if segTTR <= threshold:
            segment = []
            factor += 1
        i += 1
    if len(segment) > 0:
        factor += (1.0 - segTTR) / (1.0 - threshold)
    factor = max(1.0, factor)
    mtld = len(words) / factor
    return mtld


def MTLD(words, threshold=0.720):
    if len(words) == 0:
        return 0.0
    forward = MTLD_forward(words, threshold)
    reverse = MTLD_forward(words[::-1], threshold)
    return 0.5 * (forward + reverse)
{% endhighlight %}
</details><br>


#### vocd-D

The *vocd-D* method devised by [Malvern *et al.* (2004)](https://www.palgrave.com/gp/book/9781403902313) computes the mean TTR across 100 samples of lengths 35, 36, ... 49, 50. A function is fit to the resulting TTR vs. sample size data and the extracted best fit parameter $$D$$ is the lexical diversity index. Since this is a sampling-based method, the routine is done three times and the average of the $$D$$ values is output. I did not have access to Malvern *et al.* (2004) but did find the relevant function to fit from a [publicly viewable implementation](https://metacpan.org/release/Lingua-Diversity/source/lib/Lingua/Diversity/VOCD.pm) that is cited on a [Text Inspector page](https://textinspector.com/help/lexical-diversity).

$$f_{vocd}(N_s) = \frac{D}{N_s} \left( \sqrt{1 + 2 \frac{N_s}{D}} - 1 \right) \label{vocd-D}\tag{1}$$

where $$N_s$$ is the sample size. The higher the value of $$D$$, the more diverse the text.

#### HD-D

[McCarthy and Jarvis (2007)](https://doi.org/10.1177%2F0265532207080767) showed that *vocd-D* was merely approximating a result that could directly be computed from the [hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution). They developed an alternate implementation *HD-D* that computes the mean TTR for each sample size by summing the contribution of each type in the text to the overall mean TTR. The contribution of a type $$t$$ for a given sample size $$N_s$$ is equal to the product of that type's TTR contribution ($$1/N_s$$) and the probability of finding at least one instance of type $$t$$ in any sample. This probability is one minus the probability of finding exactly zero instances of $$t$$ in any sample, which can be computed by the hypergeometric distribution $$P_t(k_t=0, N, n_t, N_s)$$, where $$k_t$$ is the number of instances of type $$t$$, $$N$$ is still the number of tokens in the text, and $$n_t$$ is the number of occurences of $$t$$ in the full text (the order of arguments here is chosen to match the input of [`scipy.stats.hypergeom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html) rather than the example in McCarthy and Jarvis (2007)). Thus, in summary, the goal is to compute

$$ f_{HD}(N_s) = \frac{1}{N_s} \sum_{t=1}^{V} 1 - P_t(0, N, n_t, N_s) \label{HD-D}\tag{2}$$

and either equate this to the Eq. \ref{vocd-D} above and [solve](https://www.symbolab.com/solver/function-inverse-calculator/inverse%20f%5Cleft(x%5Cright)%3Dx%5Cleft(%5Csqrt%7B%5Cleft(1%2B%5Cfrac%7B2%7D%7Bx%7D%5Cright)%7D-1%5Cright)) for $$D$$:

$$ D(N_s) = -\frac{\left[f_{HD}(N_s)\right]^2}{2\left[f_{HD}(N_s)-1\right]} $$

where $$x$$ is the output of Eq. \ref{HD-D}. The average across all sample sizes gives the value of $$D$$ for the *HD-D* method. Alternatively one can instead fit Eq. \ref{vocd-D} to the output of Eq. \ref{HD-D} to determine $$D$$.


#### *vocd-D* or *HD-D*?

Although *vocd-D* merely approximates the result of *HD-D*, the latter is much slower, taking several seconds to produce *D* for a single artist's lyrics. The approximate method is still fairly accurate (within <1% away from the *HD-D* result in the example case) with just three trials, so it is the preferred method used in the rest of the notebook.


<details>
<summary>Show code</summary>
{% highlight python %}
# Example of one vocd-D trial for a single artist

words = df_bands.loc['A Canorous Quintet'].words
num_segs = 100
seglen_range = range(35, 51)
ttrs = np.zeros((len(seglen_range), num_segs))
for i, seglen in enumerate(seglen_range):
    for j in range(num_segs):
        sample = random.sample(words, seglen)
        ttrs[i, j] = TTR(sample)
avg_ttrs_vocdD = ttrs.mean(1)
curve = lambda x, D: (D / x) * (np.sqrt(1 + 2 * (x / D)) - 1)
(D_vocdD,), _ = curve_fit(curve, seglen_range, avg_ttrs_vocdD)
print(D_vocdD)

plt.plot(seglen_range, avg_ttrs_vocdD)
plt.plot(seglen_range, curve(seglen_range, D_vocdD))
plt.xlabel('Segment length (tokens)')
plt.ylabel('Mean TTR')
plt.show()
{% endhighlight %}

<pre class="code-output">
109.51382921462954
</pre>

</details><br>




![png](/assets/images/heavy-metal-lyrics/lyrics2/lyrics-part-2-lexical-diversity_12_1.png)


<details>
<summary>Show code</summary>
{% highlight python %}
# vocd-D implementation and example D values

def vocdD_curve(x, D):
    return (D / x) * (np.sqrt(1 + 2 * (x / D)) - 1)

def vocdD(words, num_trials=3, num_segs=100, min_seg=35, max_seg=50):
    if max_seg > len(words):
        return np.nan
    D_trials = []
    seglen_range = range(min_seg, max_seg + 1)
    for _ in range(num_trials):
        ttrs = np.zeros((len(seglen_range), num_segs))
        for i, seglen in enumerate(seglen_range):
            for j in range(num_segs):
                sample = random.sample(words, seglen)
                ttrs[i, j] = TTR(sample)
        avg_ttrs = ttrs.mean(1)
        (D_trial,), _ = curve_fit(vocdD_curve, seglen_range, avg_ttrs)
        D_trials.append(D_trial)
    return np.mean(D_trials)

print(vocdD(df_bands.loc['A Canorous Quintet'].words))
print(vocdD(df_bands.loc['Bal-Sagoth'].words))
print(vocdD(df_bands.loc['Dalriada'].words))
{% endhighlight %}

<pre class="code-output">
105.70245659158117
100.39975160360206
292.8665221852202
</pre>
</details><br>

<details>
<summary>Show code</summary>
{% highlight python %}
# Example of HD-D for a single artist and comparison to vocd-D

N = len(words)
types = {t: words.count(t) for t in set(words)}
seglen_range_HDD = range(35, 51)
avg_ttrs_HDD = np.zeros(len(seglen_range_HDD))
D_HDDs = np.zeros(len(seglen_range_HDD))
for s, N_s in enumerate(seglen_range_HDD):
    avg_ttr = 0
    for t, n_t in types.items():
        P_t = hypergeom(N, n_t, N_s).pmf(0)
        avg_ttr += (1 - P_t) / float(N_s)
    avg_ttrs_HDD[s] = avg_ttr
    D_HDDs[s] = -N_s * avg_ttr ** 2 / (2 * (avg_ttr - 1))
D_HDD1 = D_HDDs.mean()
print(D_HDD1)
(D_HDD2,), _ = curve_fit(vocdD_curve, seglen_range, avg_ttrs_HDD)
print(D_HDD2)

plt.plot(seglen_range, avg_ttrs_vocdD, 'o', label='vocd-D average TTR')
plt.plot(seglen_range, avg_ttrs_HDD, 'o', label='HD-D average TTR')
plt.plot(seglen_range, vocdD_curve(seglen_range, D_vocdD), label='$$D_{vocd}$$')
plt.plot(seglen_range, vocdD_curve(seglen_range, np.mean([D_HDD1, D_HDD2])), label='$$D_{HD}$$')
plt.xlabel('Segment length (tokens)')
plt.ylabel('Average TTR')
plt.legend(fontsize=12)
plt.show()
{% endhighlight %}

<pre class="code-output">
107.01527500420116
107.1183411481678
</pre>
</details><br>

![png](/assets/images/heavy-metal-lyrics/lyrics2/lyrics-part-2-lexical-diversity_14_1.png)

<details>
<summary>Show code</summary>
{% highlight python %}
def get_lexical_diversity(data):
    N = data.words.apply(len)
    V = data.words.apply(lambda x: len(set(x)))
    data['N'] = N
    data['V'] = V
    data['TTR'] = V / N
    data['rootTTR'] = V / np.sqrt(N)
    data['logTTR'] = np.log(V) / np.log(N)
    data['mtld'] = data.words.apply(MTLD, threshold=0.5)
    data['logmtld'] = np.log(data['mtld'])
    data['vocdd'] = data.words.apply(vocdD, num_segs=10)
    return data[data.N > 0]


df_bands = get_lexical_diversity(df_bands)
df_songs = get_lexical_diversity(df_songs)
{% endhighlight %}
</details><br>


## Histograms


<details>
<summary>Show code</summary>
{% highlight python %}
def plot_histograms(data):
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Lexical diversity of heavy metal artists", fontsize=30)
    axes = axes.flatten()

    ax = axes[0]
    logNmin, logNmax = np.log10(data.N.min()), np.log10(data.N.max())
    logbins = np.logspace(logNmin, logNmax, 20)
    data.N.hist(bins=logbins, edgecolor='k', ax=ax)
    ax.set_xscale('log')
    ax.set_title("Vocabulary sizes")
    ax.set_xlabel("N (tokens)")
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    ax = axes[1]
    data.TTR.hist(bins=20, edgecolor='k', ax=ax)
    ax.set_title("Type-token ratio")
    ax.set_xlabel("$$\mathrm{LD_{TTR} = V/N}$$")

    ax = axes[2]
    data.rootTTR.hist(bins=20, edgecolor='k', ax=ax)
    ax.set_title("Root type-token ratio")
    ax.set_xlabel("$$\mathrm{LD_{rootTTR}} = \sqrt{V/N}$$")

    ax = axes[3]
    data.logTTR.hist(bins=30, edgecolor='k', ax=ax)
    ax.set_title("Logarithmic type-token ratio")
    ax.set_xlabel("$$\mathrm{LD_{logTTR}} = \log V / \log N$$")

    ax = axes[4]
    data.logmtld[data.logmtld > -np.inf].hist(bins=30, edgecolor='k', ax=ax)
    ax.set_title("Measure of Textual Lexical Diversity")
    ax.set_xlabel("$$\log(\mathrm{MTLD})$$")

    ax = axes[5]
    data.vocdd[~data.vocdd.isnull()].hist(bins=30, edgecolor='k', ax=ax)
    ax.set_title("vocd-D")
    ax.set_xlabel("$$D_{\mathrm{fit}}$$")

    for ax in axes:
        ax.set_ylabel("Artists", rotation=0, labelpad=40)
        ax.grid(None)
        ax.grid(axis='y', color='k')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return
{% endhighlight %}


{% highlight python %}
plot_histograms(df_bands)
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics2/lyrics-part-2-lexical-diversity_19_0.png)

## Songs ranked by lexical diversity

Several of the highest MTLD scores come from Bal-Sagoth songs.
These are inflated as discussed in [Part 1](./lyrics0.ipnby):
Bal-Sagoth's lyrics consist of entire chapters of prose that are not actually sung in the songs themselves.
The top band, Ásmegin, also has their MTLD inflated,
in their case due to the fact that both Norwegian and English lyrics are posted to DarkLyrics for each song.
Bilingual lyrics and unsung commentary likewise inflate the lyrics of Cripple Bastards' songs,
and likewise that of Dalriada's Zách Klára.
These exceptions aside, the highest MTLD song is Divina Enema's avant-garde metal track
[Gargoyles Ye Rose Aloft](https://youtu.be/scJqkAokDus).

There's a pretty big difference between the MTLD and vocd-D charts,
with no bands shared in common among the top-ten of each metric.
MTLD seems to weigh longer lyrics more heavily,
while vocd-D almost seems to be reflecting TTR,
but at least not with the bias towards extremely short ($$N<10$$) songs.

The bottom of the chart is mostly populated by very short, usually one-word, songs.
Of songs with at least ten words, the honor of least lyrically diverse song goes to
none other than the magnificent "Thunderhorse" by Dethklok,
which consists of the words "ride", "thunder", "horse", "revenge", and of course "thunderhorse",
uttered a total of 33 times altogether.
At such low word counts, vocd-D often can't yield a score, hence the missing values.


#### Highest MTLD

<details>
<summary>Show code</summary>
{% highlight python %}
song_cols = list(df_songs.columns)
song_cols_show = list(df_songs.columns[:song_cols.index('song') + 1]) + list(df_songs.columns[song_cols.index('N'):])

pd.options.display.float_format = '{:,.2f}'.format

df_songs.sort_values('mtld', ascending=False).reset_index(drop=True).shift(1).loc[1:10, song_cols_show].convert_dtypes()
{% endhighlight %}
</details><br>





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
      <th>song</th>
      <th>N</th>
      <th>V</th>
      <th>TTR</th>
      <th>rootTTR</th>
      <th>logTTR</th>
      <th>mtld</th>
      <th>logmtld</th>
      <th>vocdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Ásmegin</td>
      <td>Viking/Folk Metal</td>
      <td>Generalen og Troldharen</td>
      <td>1029</td>
      <td>613</td>
      <td>0.60</td>
      <td>19.11</td>
      <td>0.93</td>
      <td>1,029.00</td>
      <td>6.94</td>
      <td>216.17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>The Obsidian Crown Unbound</td>
      <td>2391</td>
      <td>993</td>
      <td>0.42</td>
      <td>20.31</td>
      <td>0.89</td>
      <td>914.29</td>
      <td>6.82</td>
      <td>81.36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cripple Bastards</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>Splendore e tenebra</td>
      <td>865</td>
      <td>472</td>
      <td>0.55</td>
      <td>16.05</td>
      <td>0.91</td>
      <td>865.00</td>
      <td>6.76</td>
      <td>161.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>Six Score and Ten Oblations to a Malefic Avatar</td>
      <td>840</td>
      <td>453</td>
      <td>0.54</td>
      <td>15.63</td>
      <td>0.91</td>
      <td>840.00</td>
      <td>6.73</td>
      <td>130.15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cripple Bastards</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>La repulsione negli occhi</td>
      <td>835</td>
      <td>469</td>
      <td>0.56</td>
      <td>16.23</td>
      <td>0.91</td>
      <td>835.00</td>
      <td>6.73</td>
      <td>206.22</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cripple Bastards</td>
      <td>Noisecore (early); Grindcore (later)</td>
      <td>When Immunities Fall</td>
      <td>816</td>
      <td>492</td>
      <td>0.60</td>
      <td>17.22</td>
      <td>0.92</td>
      <td>816.00</td>
      <td>6.70</td>
      <td>210.61</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Dalriada</td>
      <td>Folk Metal</td>
      <td>Zách Klára</td>
      <td>799</td>
      <td>467</td>
      <td>0.58</td>
      <td>16.52</td>
      <td>0.92</td>
      <td>799.00</td>
      <td>6.68</td>
      <td>246.61</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Divina Enema</td>
      <td>Avant-garde Metal</td>
      <td>Gargoyles Ye Rose Aloft</td>
      <td>842</td>
      <td>419</td>
      <td>0.50</td>
      <td>14.44</td>
      <td>0.90</td>
      <td>789.38</td>
      <td>6.67</td>
      <td>143.85</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ulver</td>
      <td>Black/Folk Metal (early); Ambient/Avant-garde/...</td>
      <td>Stone Angels</td>
      <td>769</td>
      <td>418</td>
      <td>0.54</td>
      <td>15.07</td>
      <td>0.91</td>
      <td>769.00</td>
      <td>6.65</td>
      <td>162.67</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>Summoning the Guardians of the Astral Gate</td>
      <td>759</td>
      <td>404</td>
      <td>0.53</td>
      <td>14.66</td>
      <td>0.90</td>
      <td>759.00</td>
      <td>6.63</td>
      <td>87.29</td>
    </tr>
  </tbody>
</table>
</div>



#### Highest vocd-D


<details>
<summary>Show code</summary>
{% highlight python %}
df_songs.sort_values('vocdd', ascending=False).reset_index(drop=True).shift(1).loc[1:10, song_cols_show].convert_dtypes()
{% endhighlight %}
</details><br>





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
      <th>song</th>
      <th>N</th>
      <th>V</th>
      <th>TTR</th>
      <th>rootTTR</th>
      <th>logTTR</th>
      <th>mtld</th>
      <th>logmtld</th>
      <th>vocdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Hail of Bullets</td>
      <td>Death Metal</td>
      <td>DAK</td>
      <td>54</td>
      <td>54</td>
      <td>1.00</td>
      <td>7.35</td>
      <td>1.00</td>
      <td>54</td>
      <td>3.99</td>
      <td>787,177.13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cadaveria</td>
      <td>Black/Gothic Metal</td>
      <td>Exercise1</td>
      <td>119</td>
      <td>119</td>
      <td>1.00</td>
      <td>10.91</td>
      <td>1.00</td>
      <td>119</td>
      <td>4.78</td>
      <td>787,177.13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cenotaph</td>
      <td>Brutal Death Metal</td>
      <td>Voluptuously Minced</td>
      <td>67</td>
      <td>66</td>
      <td>0.99</td>
      <td>8.06</td>
      <td>1.00</td>
      <td>67</td>
      <td>4.20</td>
      <td>2,095.87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yearning</td>
      <td>Atmospheric Doom Metal</td>
      <td>Elegy of Blood</td>
      <td>56</td>
      <td>55</td>
      <td>0.98</td>
      <td>7.35</td>
      <td>1.00</td>
      <td>56</td>
      <td>4.03</td>
      <td>1,576.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sempiternal Deathreign</td>
      <td>Death/Doom Metal</td>
      <td>Devastating Empire Towards Humanity</td>
      <td>75</td>
      <td>73</td>
      <td>0.97</td>
      <td>8.43</td>
      <td>0.99</td>
      <td>75</td>
      <td>4.32</td>
      <td>1,455.21</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Incantation</td>
      <td>Death Metal</td>
      <td>Invoked Infinity</td>
      <td>53</td>
      <td>52</td>
      <td>0.98</td>
      <td>7.14</td>
      <td>1.00</td>
      <td>53</td>
      <td>3.97</td>
      <td>1,432.34</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Houwitser</td>
      <td>Brutal Death Metal</td>
      <td>Vile Amputation</td>
      <td>53</td>
      <td>52</td>
      <td>0.98</td>
      <td>7.14</td>
      <td>1.00</td>
      <td>53</td>
      <td>3.97</td>
      <td>1,379.03</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Kill the Client</td>
      <td>Grindcore</td>
      <td>As Roaches</td>
      <td>52</td>
      <td>51</td>
      <td>0.98</td>
      <td>7.07</td>
      <td>1.00</td>
      <td>52</td>
      <td>3.95</td>
      <td>1,378.93</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Incantation</td>
      <td>Death Metal</td>
      <td>Ancients Arise</td>
      <td>73</td>
      <td>71</td>
      <td>0.97</td>
      <td>8.31</td>
      <td>0.99</td>
      <td>73</td>
      <td>4.29</td>
      <td>1,131.15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Venom</td>
      <td>NWOBHM/Black/Speed Metal</td>
      <td>All There Is Fear</td>
      <td>78</td>
      <td>75</td>
      <td>0.96</td>
      <td>8.49</td>
      <td>0.99</td>
      <td>78</td>
      <td>4.36</td>
      <td>1,078.96</td>
    </tr>
  </tbody>
</table>
</div>




#### Lowest MTLD

<details>
<summary>Show code</summary>
{% highlight python %}
df_songs[df_songs.N > 10].sort_values('mtld', ascending=True).reset_index(drop=True).shift(1).loc[1:10, song_cols_show].convert_dtypes()
{% endhighlight %}
</details><br>





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
      <th>song</th>
      <th>N</th>
      <th>V</th>
      <th>TTR</th>
      <th>rootTTR</th>
      <th>logTTR</th>
      <th>mtld</th>
      <th>logmtld</th>
      <th>vocdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Dethklok</td>
      <td>Melodic Death Metal</td>
      <td>Thunderhorse</td>
      <td>33</td>
      <td>5</td>
      <td>0.15</td>
      <td>0.87</td>
      <td>0.46</td>
      <td>2.95</td>
      <td>1.08</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Putrid Pile</td>
      <td>Brutal Death Metal</td>
      <td>Toxic Shock Therapy</td>
      <td>18</td>
      <td>3</td>
      <td>0.17</td>
      <td>0.71</td>
      <td>0.38</td>
      <td>3.30</td>
      <td>1.19</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Axxis</td>
      <td>Melodic Heavy/Power Metal</td>
      <td>Journey to Utopia</td>
      <td>15</td>
      <td>2</td>
      <td>0.13</td>
      <td>0.52</td>
      <td>0.26</td>
      <td>3.75</td>
      <td>1.32</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Katatonia</td>
      <td>Doom/Death Metal (early); Gothic/Alternative/P...</td>
      <td>Dancing December</td>
      <td>12</td>
      <td>2</td>
      <td>0.17</td>
      <td>0.58</td>
      <td>0.28</td>
      <td>4.00</td>
      <td>1.39</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Floodgate</td>
      <td>Doom/Stoner Metal</td>
      <td>Feel You Burn</td>
      <td>56</td>
      <td>2</td>
      <td>0.04</td>
      <td>0.27</td>
      <td>0.17</td>
      <td>4.00</td>
      <td>1.39</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Lost Society</td>
      <td>Thrash Metal (early); Groove Metal/Metalcore (...</td>
      <td>Fatal Anoxia</td>
      <td>15</td>
      <td>2</td>
      <td>0.13</td>
      <td>0.52</td>
      <td>0.26</td>
      <td>4.09</td>
      <td>1.41</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M.O.D.</td>
      <td>Thrash Metal/Hardcore/Crossover</td>
      <td>Bubble Butt</td>
      <td>25</td>
      <td>5</td>
      <td>0.20</td>
      <td>1.00</td>
      <td>0.50</td>
      <td>4.17</td>
      <td>1.43</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dawnbringer</td>
      <td>Heavy Metal with Black Metal influences</td>
      <td>Scream and Run</td>
      <td>51</td>
      <td>3</td>
      <td>0.06</td>
      <td>0.42</td>
      <td>0.28</td>
      <td>5.44</td>
      <td>1.69</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Fleshless</td>
      <td>Death Metal/Grindcore (early); Melodic/Brutal ...</td>
      <td>Evil Odium</td>
      <td>11</td>
      <td>3</td>
      <td>0.27</td>
      <td>0.90</td>
      <td>0.46</td>
      <td>5.81</td>
      <td>1.76</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bloodsoaked</td>
      <td>Brutal Technical Death Metal</td>
      <td>Devastation Death</td>
      <td>12</td>
      <td>4</td>
      <td>0.33</td>
      <td>1.15</td>
      <td>0.56</td>
      <td>6.00</td>
      <td>1.79</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
</div>


#### Lowest vocd-D


<details>
<summary>Show code</summary>
{% highlight python %}
df_songs.sort_values('vocdd', ascending=True).reset_index(drop=True).shift(1).loc[1:10, song_cols_show].convert_dtypes()
{% endhighlight %}
</details><br>





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
      <th>song</th>
      <th>N</th>
      <th>V</th>
      <th>TTR</th>
      <th>rootTTR</th>
      <th>logTTR</th>
      <th>mtld</th>
      <th>logmtld</th>
      <th>vocdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Floodgate</td>
      <td>Doom/Stoner Metal</td>
      <td>Feel You Burn</td>
      <td>56</td>
      <td>2</td>
      <td>0.04</td>
      <td>0.27</td>
      <td>0.17</td>
      <td>4.00</td>
      <td>1.39</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dawnbringer</td>
      <td>Heavy Metal with Black Metal influences</td>
      <td>Scream and Run</td>
      <td>51</td>
      <td>3</td>
      <td>0.06</td>
      <td>0.42</td>
      <td>0.28</td>
      <td>5.44</td>
      <td>1.69</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Type O Negative</td>
      <td>Gothic/Doom Metal</td>
      <td>Set Me on Fire</td>
      <td>72</td>
      <td>7</td>
      <td>0.10</td>
      <td>0.82</td>
      <td>0.46</td>
      <td>9.14</td>
      <td>2.21</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hanzel und Gretyl</td>
      <td>Industrial Rock (early); Industrial/Groove Met...</td>
      <td>Watch TV Do Nothing</td>
      <td>54</td>
      <td>10</td>
      <td>0.19</td>
      <td>1.36</td>
      <td>0.58</td>
      <td>9.90</td>
      <td>2.29</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nuit Noire</td>
      <td>Black Metal (early); Punk/Black Metal (later)</td>
      <td>I Am a Fairy</td>
      <td>150</td>
      <td>16</td>
      <td>0.11</td>
      <td>1.31</td>
      <td>0.55</td>
      <td>10.14</td>
      <td>2.32</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mortiis</td>
      <td>Darkwave/Dungeon Synth, Electronic/Industrial ...</td>
      <td>Thieving Bastards</td>
      <td>56</td>
      <td>9</td>
      <td>0.16</td>
      <td>1.20</td>
      <td>0.55</td>
      <td>9.33</td>
      <td>2.23</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Vendetta</td>
      <td>Thrash Metal</td>
      <td>Love Song</td>
      <td>55</td>
      <td>10</td>
      <td>0.18</td>
      <td>1.35</td>
      <td>0.57</td>
      <td>18.52</td>
      <td>2.92</td>
      <td>1.47</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Devin Townsend</td>
      <td>Progressive Metal, Ambient</td>
      <td>Unity</td>
      <td>77</td>
      <td>12</td>
      <td>0.16</td>
      <td>1.37</td>
      <td>0.57</td>
      <td>8.70</td>
      <td>2.16</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hanzel und Gretyl</td>
      <td>Industrial Rock (early); Industrial/Groove Met...</td>
      <td>Hallo Berlin</td>
      <td>76</td>
      <td>10</td>
      <td>0.13</td>
      <td>1.15</td>
      <td>0.53</td>
      <td>13.24</td>
      <td>2.58</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th>10</th>
      <td>The Great Kat</td>
      <td>Speed/Thrash Metal, Shred</td>
      <td>Kill the Mothers</td>
      <td>199</td>
      <td>22</td>
      <td>0.11</td>
      <td>1.56</td>
      <td>0.58</td>
      <td>7.12</td>
      <td>1.96</td>
      <td>1.55</td>
    </tr>
  </tbody>
</table>
</div>


## Bands ranked by lexical diversity




#### Highest MTLD


<details>
<summary>Show code</summary>
{% highlight python %}
band_cols = list(df_bands.columns)
band_cols_show = ['band', 'genre'] + band_cols[band_cols.index('N'):]
{% endhighlight %}

{% highlight python %}
df_bands.sort_values('mtld', ascending=False).reset_index(drop=False).shift(1).loc[1:10, band_cols_show].convert_dtypes()
{% endhighlight %}
</details><br>





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
      <th>N</th>
      <th>V</th>
      <th>TTR</th>
      <th>rootTTR</th>
      <th>logTTR</th>
      <th>mtld</th>
      <th>logmtld</th>
      <th>vocdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Hail of Bullets</td>
      <td>Death Metal</td>
      <td>3383</td>
      <td>1804</td>
      <td>0.53</td>
      <td>31.02</td>
      <td>0.92</td>
      <td>3,383.00</td>
      <td>8.13</td>
      <td>206.33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Demolition Hammer</td>
      <td>Thrash Metal (early); Groove Metal (later)</td>
      <td>2845</td>
      <td>1665</td>
      <td>0.59</td>
      <td>31.22</td>
      <td>0.93</td>
      <td>2,845.00</td>
      <td>7.95</td>
      <td>363.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Malignancy</td>
      <td>Brutal Technical Death Metal</td>
      <td>6857</td>
      <td>2094</td>
      <td>0.31</td>
      <td>25.29</td>
      <td>0.87</td>
      <td>2,504.79</td>
      <td>7.83</td>
      <td>191.29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Abnormality</td>
      <td>Technical/Brutal Death Metal</td>
      <td>2443</td>
      <td>1235</td>
      <td>0.51</td>
      <td>24.99</td>
      <td>0.91</td>
      <td>2,443.00</td>
      <td>7.80</td>
      <td>155.92</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Botanist</td>
      <td>Experimental Post-Black Metal</td>
      <td>2367</td>
      <td>1342</td>
      <td>0.57</td>
      <td>27.58</td>
      <td>0.93</td>
      <td>2,367.00</td>
      <td>7.77</td>
      <td>266.92</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Moonsorrow</td>
      <td>Folk/Pagan/Black Metal</td>
      <td>7973</td>
      <td>3177</td>
      <td>0.40</td>
      <td>35.58</td>
      <td>0.90</td>
      <td>2,344.84</td>
      <td>7.76</td>
      <td>249.43</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thought Industry</td>
      <td>Progressive Thrash Metal (early); Alternative ...</td>
      <td>2235</td>
      <td>1245</td>
      <td>0.56</td>
      <td>26.33</td>
      <td>0.92</td>
      <td>2,235.00</td>
      <td>7.71</td>
      <td>305.54</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Scrambled Defuncts</td>
      <td>Brutal Technical Death Metal</td>
      <td>2121</td>
      <td>1097</td>
      <td>0.52</td>
      <td>23.82</td>
      <td>0.91</td>
      <td>2,121.00</td>
      <td>7.66</td>
      <td>166.62</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Brutality</td>
      <td>Death Metal</td>
      <td>3535</td>
      <td>1663</td>
      <td>0.47</td>
      <td>27.97</td>
      <td>0.91</td>
      <td>2,106.26</td>
      <td>7.65</td>
      <td>192.63</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Metsatöll</td>
      <td>Folk Metal</td>
      <td>3340</td>
      <td>1652</td>
      <td>0.49</td>
      <td>28.58</td>
      <td>0.91</td>
      <td>2,044.72</td>
      <td>7.62</td>
      <td>254.57</td>
    </tr>
  </tbody>
</table>
</div>



#### Highest vocd-D


<details>
<summary>Show code</summary>
{% highlight python %}
df_bands.sort_values('vocdd', ascending=False).reset_index(drop=False).shift(1).loc[1:10, band_cols_show].convert_dtypes()
{% endhighlight %}
</details><br>





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
      <th>N</th>
      <th>V</th>
      <th>TTR</th>
      <th>rootTTR</th>
      <th>logTTR</th>
      <th>mtld</th>
      <th>logmtld</th>
      <th>vocdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Krieg</td>
      <td>Black Metal</td>
      <td>54</td>
      <td>51</td>
      <td>0.94</td>
      <td>6.94</td>
      <td>0.99</td>
      <td>54.00</td>
      <td>3.99</td>
      <td>414.42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nortt</td>
      <td>Black/Funeral Doom Metal</td>
      <td>105</td>
      <td>94</td>
      <td>0.90</td>
      <td>9.17</td>
      <td>0.98</td>
      <td>105.00</td>
      <td>4.65</td>
      <td>378.53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Demolition Hammer</td>
      <td>Thrash Metal (early); Groove Metal (later)</td>
      <td>2845</td>
      <td>1665</td>
      <td>0.59</td>
      <td>31.22</td>
      <td>0.93</td>
      <td>2,845.00</td>
      <td>7.95</td>
      <td>363.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Four Question Marks</td>
      <td>Death Metal, Groove Metal</td>
      <td>901</td>
      <td>555</td>
      <td>0.62</td>
      <td>18.49</td>
      <td>0.93</td>
      <td>901.00</td>
      <td>6.80</td>
      <td>358.75</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Virulence</td>
      <td>Death Metal/Grindcore with Jazz influences</td>
      <td>719</td>
      <td>518</td>
      <td>0.72</td>
      <td>19.32</td>
      <td>0.95</td>
      <td>719.00</td>
      <td>6.58</td>
      <td>353.99</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Terravore</td>
      <td>Thrash Metal</td>
      <td>1364</td>
      <td>869</td>
      <td>0.64</td>
      <td>23.53</td>
      <td>0.94</td>
      <td>1,364.00</td>
      <td>7.22</td>
      <td>308.57</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thought Industry</td>
      <td>Progressive Thrash Metal (early); Alternative ...</td>
      <td>2235</td>
      <td>1245</td>
      <td>0.56</td>
      <td>26.33</td>
      <td>0.92</td>
      <td>2,235.00</td>
      <td>7.71</td>
      <td>305.54</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dalriada</td>
      <td>Folk Metal</td>
      <td>5458</td>
      <td>2215</td>
      <td>0.41</td>
      <td>29.98</td>
      <td>0.90</td>
      <td>1,034.99</td>
      <td>6.94</td>
      <td>297.45</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ásmegin</td>
      <td>Viking/Folk Metal</td>
      <td>5626</td>
      <td>2276</td>
      <td>0.40</td>
      <td>30.34</td>
      <td>0.90</td>
      <td>1,337.58</td>
      <td>7.20</td>
      <td>271.11</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Botanist</td>
      <td>Experimental Post-Black Metal</td>
      <td>2367</td>
      <td>1342</td>
      <td>0.57</td>
      <td>27.58</td>
      <td>0.93</td>
      <td>2,367.00</td>
      <td>7.77</td>
      <td>266.92</td>
    </tr>
  </tbody>
</table>
</div>




#### Lowest MTLD

<details>
<summary>Show code</summary>
{% highlight python %}
df_bands.sort_values('mtld', ascending=True).reset_index(drop=False).shift(1).loc[1:10, band_cols_show].convert_dtypes()
{% endhighlight %}
</details><br>





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
      <th>N</th>
      <th>V</th>
      <th>TTR</th>
      <th>rootTTR</th>
      <th>logTTR</th>
      <th>mtld</th>
      <th>logmtld</th>
      <th>vocdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>God Seed</td>
      <td>Black Metal</td>
      <td>51</td>
      <td>22</td>
      <td>0.43</td>
      <td>3.08</td>
      <td>0.79</td>
      <td>16.26</td>
      <td>2.79</td>
      <td>8.88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Great Kat</td>
      <td>Speed/Thrash Metal, Shred</td>
      <td>2233</td>
      <td>509</td>
      <td>0.23</td>
      <td>10.77</td>
      <td>0.81</td>
      <td>26.85</td>
      <td>3.29</td>
      <td>75.45</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Obús</td>
      <td>Heavy Metal/Hard Rock</td>
      <td>395</td>
      <td>115</td>
      <td>0.29</td>
      <td>5.79</td>
      <td>0.79</td>
      <td>31.73</td>
      <td>3.46</td>
      <td>17.15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dolorian</td>
      <td>Black/Doom Metal, Ritual Ambient</td>
      <td>1583</td>
      <td>397</td>
      <td>0.25</td>
      <td>9.98</td>
      <td>0.81</td>
      <td>35.49</td>
      <td>3.57</td>
      <td>87.97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Beyond Black Void</td>
      <td>Drone/Funeral Doom Metal</td>
      <td>43</td>
      <td>34</td>
      <td>0.79</td>
      <td>5.18</td>
      <td>0.94</td>
      <td>43.00</td>
      <td>3.76</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Steak Number Eight</td>
      <td>Atmospheric Sludge/Post-Metal</td>
      <td>712</td>
      <td>181</td>
      <td>0.25</td>
      <td>6.78</td>
      <td>0.79</td>
      <td>46.31</td>
      <td>3.84</td>
      <td>56.18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Svart</td>
      <td>Black Metal</td>
      <td>47</td>
      <td>31</td>
      <td>0.66</td>
      <td>4.52</td>
      <td>0.89</td>
      <td>47.00</td>
      <td>3.85</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Unida</td>
      <td>Stoner Metal</td>
      <td>1118</td>
      <td>247</td>
      <td>0.22</td>
      <td>7.39</td>
      <td>0.78</td>
      <td>49.85</td>
      <td>3.91</td>
      <td>57.99</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Krieg</td>
      <td>Black Metal</td>
      <td>54</td>
      <td>51</td>
      <td>0.94</td>
      <td>6.94</td>
      <td>0.99</td>
      <td>54.00</td>
      <td>3.99</td>
      <td>414.42</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Nuit Noire</td>
      <td>Black Metal (early); Punk/Black Metal (later)</td>
      <td>963</td>
      <td>310</td>
      <td>0.32</td>
      <td>9.99</td>
      <td>0.84</td>
      <td>55.39</td>
      <td>4.01</td>
      <td>52.26</td>
    </tr>
  </tbody>
</table>
</div>



#### Lowest vocd-D


<details>
<summary>Show code</summary>
{% highlight python %}
df_bands.sort_values('vocdd', ascending=True).reset_index(drop=False).shift(1).loc[1:10, band_cols_show].convert_dtypes()
{% endhighlight %}
</details><br>





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
      <th>N</th>
      <th>V</th>
      <th>TTR</th>
      <th>rootTTR</th>
      <th>logTTR</th>
      <th>mtld</th>
      <th>logmtld</th>
      <th>vocdd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>God Seed</td>
      <td>Black Metal</td>
      <td>51</td>
      <td>22</td>
      <td>0.43</td>
      <td>3.08</td>
      <td>0.79</td>
      <td>16.26</td>
      <td>2.79</td>
      <td>8.88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Obús</td>
      <td>Heavy Metal/Hard Rock</td>
      <td>395</td>
      <td>115</td>
      <td>0.29</td>
      <td>5.79</td>
      <td>0.79</td>
      <td>31.73</td>
      <td>3.46</td>
      <td>17.15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hämatom</td>
      <td>Groove Metal</td>
      <td>189</td>
      <td>64</td>
      <td>0.34</td>
      <td>4.66</td>
      <td>0.79</td>
      <td>60.73</td>
      <td>4.11</td>
      <td>18.19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Saratoga</td>
      <td>Power/Heavy Metal</td>
      <td>113</td>
      <td>56</td>
      <td>0.50</td>
      <td>5.27</td>
      <td>0.85</td>
      <td>88.43</td>
      <td>4.48</td>
      <td>23.72</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mystic Forest</td>
      <td>Melodic Black Metal</td>
      <td>207</td>
      <td>88</td>
      <td>0.43</td>
      <td>6.12</td>
      <td>0.84</td>
      <td>100.23</td>
      <td>4.61</td>
      <td>34.62</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gris</td>
      <td>Black Metal</td>
      <td>122</td>
      <td>59</td>
      <td>0.48</td>
      <td>5.34</td>
      <td>0.85</td>
      <td>122.00</td>
      <td>4.80</td>
      <td>36.22</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bucovina</td>
      <td>Black/Folk Metal</td>
      <td>238</td>
      <td>103</td>
      <td>0.43</td>
      <td>6.68</td>
      <td>0.85</td>
      <td>110.66</td>
      <td>4.71</td>
      <td>37.48</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EgoNoir</td>
      <td>Black Metal</td>
      <td>58</td>
      <td>40</td>
      <td>0.69</td>
      <td>5.25</td>
      <td>0.91</td>
      <td>58.00</td>
      <td>4.06</td>
      <td>41.01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Shadowbreed</td>
      <td>Death Metal</td>
      <td>306</td>
      <td>100</td>
      <td>0.33</td>
      <td>5.72</td>
      <td>0.80</td>
      <td>100.38</td>
      <td>4.61</td>
      <td>42.39</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Krigere Wolf</td>
      <td>Black/Death Metal</td>
      <td>830</td>
      <td>345</td>
      <td>0.42</td>
      <td>11.98</td>
      <td>0.87</td>
      <td>309.29</td>
      <td>5.73</td>
      <td>43.43</td>
    </tr>
  </tbody>
</table>
</div>






## Genres ranked by lexical diversity


<details>
<summary>Show code</summary>
{% highlight python %}
def plot_violinplots(data, figsize=(16, 18)):

    def violinplot(col, ax):
        violindata = []
        labels = data.columns[1:list(data.columns).index('lyrics')]
        for label in labels:
            values = data[data[label] == 1][col]
            values = values[(values > -np.inf) & (values < np.inf)]
            violindata.append((label, values))
        violindata.sort(key=lambda x: -x[1].median())
        plot_labels, plot_data = zip(*violindata)
        sns.violinplot(data=plot_data, cut=0, orient='h', ax=ax, color='c')
        ax.set_yticklabels(plot_labels)
        return

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle("Lexical diversity of artists by genre", fontsize=30)
    axes = axes.flatten()

    ax = axes[0]
    violinplot('N', ax)
    ax.set_title("Word counts")
    ax.set_xlabel("N (tokens)")

    ax = axes[1]
    violinplot('V', ax)
    ax.set_title("Unique word counts")
    ax.set_xlabel("V (types)")

    ax = axes[2]
    violinplot('TTR', ax)
    ax.set_title("Type-token ratio")
    ax.set_xlabel(r"$$\mathrm{LD_{TTR}}$$")

    ax = axes[3]
    violinplot('logTTR', ax)
    ax.set_title("Logarithmic type-token ratio")
    ax.set_xlabel(r"$$\mathrm{LD_{logTTR}}$$")

    ax = axes[4]
    violinplot('logmtld', ax)
    ax.set_title("Measure of Textual Lexical Diversity")
    ax.set_xlabel("$$\log(\mathrm{MTLD})$$")

    ax = axes[5]
    violinplot('vocdd', ax)
    ax.set_title("vocd-D")
    ax.set_xlabel("$$D_{\mathrm{fit}}$$")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return
{% endhighlight %}


{% highlight python %}
plot_violinplots(df_bands, figsize=(16, 18))
{% endhighlight %}
</details><br>



    
![png](/assets/images/heavy-metal-lyrics/lyrics2/lyrics-part-2-lexical-diversity_22_0.png)

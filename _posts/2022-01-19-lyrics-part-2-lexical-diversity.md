---
layout: post
title: "Analysis of heavy metal lyrics - Part 2: Lexical diversity"
categories: jekyll update
hidden: true
---

<pre style="margin-left: 50px; margin-right: 50px; font-size: 13px">
Explicit/NSFW content warning: this project features examples of heavy metal lyrics and song/album/band names.
These often contain words and themes that some may find offensive/inappropriate.
</pre>

This article is the second part of the lyrical analysis [heavy metal lyrics](./heavy-metal-lyrics.html).
Below is the same dashboard presented in [Part 1](./lyrics-part-1-overview.html)
([click here for full-size version](https://metal-lyrics-feature-plots.herokuapp.com/){:target="_blank"}).
Here we will look at the lexical diversity measures included among the plot options (e.g. TTR, MTLD, and vocd-D).
If you're interested in seeing the full code (a lot is omitted here), check out the
[original notebook](https://github.com/pdqnguyen/metallyrics/blob/main/analyses/lyrics/notebooks/lyrics-part-2-lexical-diversity.ipynb).
In the [next article](./lyrics-part-3-word-clouds.html) we'll use word clouds to describe the different genres.

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



## Module imports

<details>
<summary>Show code</summary>
{% highlight python %}
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.style.use('seaborn')
import seaborn as sns
sns.set(font_scale=2)
from nltk.corpus import words as nltk_words

from nlp import tokenize
{% endhighlight %}
</details>

## Preprocessing data

<details>
<summary>Show code</summary>
{% highlight python %}
def get_genres(data):
    columns = [col for col in data.columns if 'genre_' in col]
    genres = [re.sub(r"^genre\_", "", col) for col in columns]
    return genres, columns


def get_bands(data):
    genres, genre_cols = get_genres(data)
    # Combine songs from same band
    band_genres = data.groupby('band_name')[genre_cols].max()
    band_lyrics = data.groupby('band_name').song_darklyrics.sum()
    bands = pd.concat((band_genres, band_lyrics), axis=1)
    bands.columns = genres + ['lyrics']
    bands['words'] = bands.lyrics.apply(tokenize)
    return bands


def get_songs(data):
    genres, genre_cols = get_genres(data)
    songs = data[['band_name', 'song_name'] + genre_cols + ['song_darklyrics']].copy()
    songs.columns = ['band_name', 'song_name'] + genres + ['lyrics']
    songs['words'] = songs.lyrics.apply(tokenize)
    return songs
{% endhighlight %}

{% highlight python %}
df = pd.read_csv('songs-10pct.csv')
df_bands = get_bands(df)
df_songs = get_songs(df)
{% endhighlight %}
</details>

## Lexical diversity measures

One simple approach to quantifying lexical diversity is to divide the number of unique words (types, $V$)
by the total word count (tokens, $N$). This type-token ratio or TTR is heavily biased toward short texts,
since longer texts are more likely to repeat tokens without necessarily diminishing complexity.
A few ways exist for rescaling the relationship to reduce this bias;
for this notebook the root-TTR and log-TTR are used:

$$
\begin{split}
&LD_{TTR}       &= \frac{V}{N}              &\hspace{1cm} (\textrm{type-token ratio})               \\
&LD_{rootTTR}   &= \frac{V}{\sqrt{N}}       &\hspace{1cm} (\textrm{root type-token ratio})          \\
&LD_{logTTR}    &= \frac{\log{V}}{\log{N}}  &\hspace{1cm} (\textrm{logarithmic type-token ratio})   \\
\end{split}
$$

#### MTLD

More sophisticated approaches look at how types are distributed in the text.
The bluntly named Measure of Textual Lexical Diversity (MTLD),
described by [McCarthy and Jarvis (2010)](https://doi.org/10.3758/BRM.42.2.381),
is based on the mean length of token sequences in the text that exceed a certain TTR threshold.
The algorithm begins with a sequence consisting of the first token in the text,
and iteratively adds the following token, each time recomputing the TTR of the sequence so far.
Once the sequence TTR drops below the pre-determined threshold,
the sequence ends and a new sequence begins at the next token.
This continues until the end of the text is reached, at which point the mean sequence length is computed.
The process is repeated from the last token, going backwards, to produce another mean sequence length.
The mean of these two results is the final MTLD figure.

Unlike the simpler methods, MTLD has a tunable parameter.
The TTR threshold is chosen by the authors to be 0.720,
which is approximately where the cumulative TTR curves for texts in the Project Gutenburg Text Archives
reached a point of stabilization. The same can be done with the DarkLyrics data by plotting cumulative TTR
values for a large number of bands and identifying the point of stabilization.
This cannot be done with single-song lyrics since refrains in the lyrics heavily warp the cumulative TTR curves,
such that most never stabilize. Unfortunately even when looking at band lyrics,
the cumulative TTR does not stabilize very well,
as the curves seem to continue decaying well into the thousands of tokens.
However one can roughly identify a point of stabilization somewhere around a TTR of 0.5,
occuring at about 200 tokens, so this is used as the threshold for MTLD.


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

Here is the implementation of MTLD:

![mtld](/assets/images/heavy-metal-lyrics/mtld.png)

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

#### vocd-D

The *vocd-D* method devised by [Malvern *et al.* (2004)](https://www.palgrave.com/gp/book/9781403902313)
computes the mean TTR across 100 samples of lengths 35, 36, ... 49, 50.
A function is fit to the resulting TTR vs. sample size data and the extracted best fit parameter $D$
is the lexical diversity index.
Since this is a sampling-based method, the routine is done three times and the average of the $D$ values is output.
I did not have access to Malvern *et al.* (2004) but did find the relevant function to fit from a
[publicly viewable implementation](https://metacpan.org/release/Lingua-Diversity/source/lib/Lingua/Diversity/VOCD.pm)
that is cited on a [Text Inspector page](https://textinspector.com/help/lexical-diversity).

$$f_{vocd}(N_s) = \frac{D}{N_s} \left( \sqrt{1 + 2 \frac{N_s}{D}} - 1 \right) \label{vocd-D}\tag{1}$$

where $N_s$ is the sample size. The higher the value of $D$, the more diverse the text.

#### HD-D

[McCarthy and Jarvis (2007)](https://doi.org/10.1177%2F0265532207080767) showed that *vocd-D* was merely
approximating a result that could directly be computed from the
[hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution).
They developed an alternate implementation *HD-D* that computes the mean TTR for each sample size by summing the
contribution of each type in the text to the overall mean TTR.
The contribution of a type $t$ for a given sample size $N_s$ is equal to the product of that type's TTR contribution
($1/N_s$) and the probability of finding at least one instance of type $t$ in any sample.
This probability is one minus the probability of finding exactly zero instances of $t$ in any sample,
which can be computed by the hypergeometric distribution $P_t(k_t=0, N, n_t, N_s)$,
where $k_t$ is the number of instances of type $t$,
$N$ is still the number of tokens in the text, and $n_t$ is the number of occurences of $t$ in the full text
(the order of arguments here is chosen to match the input of
[`scipy.stats.hypergeom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html)
rather than the example in McCarthy and Jarvis (2007)).
Thus, in summary, the goal is to compute

$$ f_{HD}(N_s) = \frac{1}{N_s} \sum_{t=1}^{V} 1 - P_t(0, N, n_t, N_s) \label{HD-D}\tag{2}$$

and either equate this to the Eq. \ref{vocd-D} above and [solve](https://www.symbolab.com/solver/function-inverse-calculator/inverse%20f%5Cleft(x%5Cright)%3Dx%5Cleft(%5Csqrt%7B%5Cleft(1%2B%5Cfrac%7B2%7D%7Bx%7D%5Cright)%7D-1%5Cright)) for $D$:

$$ D(N_s) = -\frac{\left[f_{HD}(N_s)\right]^2}{2\left[f_{HD}(N_s)-1\right]} $$

where $x$ is the output of Eq. \ref{HD-D}. The average across all sample sizes gives the value of $D$ for the
*HD-D* method. Alternatively one can instead fit Eq. \ref{vocd-D} to the output of Eq. \ref{HD-D} to determine $D$.


#### *vocd-D* or *HD-D*?

Although *vocd-D* merely approximates the result of *HD-D*, the latter is much slower,
taking several seconds to produce *D* for a single artist's lyrics.
The approximate method is still fairly accurate (within <1% away from the *HD-D* result in the example case below)
with just three trials, so it is the preferred method used in the rest of the notebook.

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

![vocd-D](/assets/images/heavy-metal-lyrics/vocdd.png)

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
107.24246453948517
100.87182576210539
201.867083419284
</pre>

Here is an example of an *HD-D* trial. I print the value taken from hypergeometric method as well as
from the curve fit to the *vocd-D* data, but they're essentially equivalent.

{% highlight python %}
# Example of HD-D for a single artist and comparison to vocd-D

N = len(words)
types = {t: words.count(t) for t in set(words)}
seglen_range_HDD = range(35, 51)
avg_ttrs_HDD = np.zeros(len(seglen_range_HDD))
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
plt.plot(seglen_range, vocdD_curve(seglen_range, D_vocdD), label='$D_{vocd}$')
plt.plot(seglen_range, vocdD_curve(seglen_range, np.mean([D_HDD1, D_HDD2])), label='$D_{HD}$')
plt.xlabel('Segment length (tokens)')
plt.ylabel('Average TTR')
plt.legend(fontsize=12)
plt.show()
{% endhighlight %}

<pre class="code-output">
107.01527500420107
107.11834112312057
</pre>

![vocdd-hdd](/assets/images/heavy-metal-lyrics/vocdd-hdd.png)

#### Applying the lexical diversity measures

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
{% endhighlight %}

## Histograms

![lexical diversity histograms](/assets/images/heavy-metal-lyrics/ld_hist.png)

## Genre comparison (top 5 genres)

![ld_violin](/assets/images/heavy-metal-lyrics/ld_violin.png)

## Genre comparison (top 13 genres)

![ld_violin_2](/assets/images/heavy-metal-lyrics/ld_violin_2.png)

## Ranking songs

#### Highest MTLD songs

Several of the highest MTLD scores come from Bal-Sagoth songs.
These are inflated as discussed in [Part 1](./lyrics-part-1-overview.html):
[Bal-Sagoth's lyrics](http://www.darklyrics.com/lyrics/balsagoth/thechthonicchronicles.html#4)
consist of entire chapters of prose that are not actually sung in the songs themselves.
Apart from this, The song with the highest MTLD is a 6-minute track by the Hungarian folk metal band Dalriada,
[Zách Klára](http://www.darklyrics.com/lyrics/dalriada/aranyalbum.html#1).

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
      <td>83.49</td>
    </tr>
    <tr>
      <th>2</th>
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
      <td>122.56</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>253.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ulver</td>
      <td>Black/Folk Metal (early), Ambient/Avant-garde/...</td>
      <td>Stone Angels</td>
      <td>769</td>
      <td>418</td>
      <td>0.54</td>
      <td>15.07</td>
      <td>0.91</td>
      <td>769.00</td>
      <td>6.65</td>
      <td>163.42</td>
    </tr>
    <tr>
      <th>5</th>
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
      <td>89.30</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Dalriada</td>
      <td>Folk Metal</td>
      <td>Ágnes asszony (1. rész)</td>
      <td>744</td>
      <td>393</td>
      <td>0.53</td>
      <td>14.41</td>
      <td>0.90</td>
      <td>744.00</td>
      <td>6.61</td>
      <td>139.09</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>To Dethrone the Witch-Queen of Mytos K'unn (Th...</td>
      <td>1413</td>
      <td>621</td>
      <td>0.44</td>
      <td>16.52</td>
      <td>0.89</td>
      <td>741.83</td>
      <td>6.61</td>
      <td>91.81</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>Unfettering the Hoary Sentinels of Karnak</td>
      <td>1344</td>
      <td>627</td>
      <td>0.47</td>
      <td>17.10</td>
      <td>0.89</td>
      <td>730.72</td>
      <td>6.59</td>
      <td>87.07</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cradle of Filth</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>Beneath the Howling Stars</td>
      <td>706</td>
      <td>426</td>
      <td>0.60</td>
      <td>16.03</td>
      <td>0.92</td>
      <td>706.00</td>
      <td>6.56</td>
      <td>142.68</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>Blood Slakes the Sand at the Circus Maximus</td>
      <td>1280</td>
      <td>586</td>
      <td>0.46</td>
      <td>16.38</td>
      <td>0.89</td>
      <td>701.02</td>
      <td>6.55</td>
      <td>110.94</td>
    </tr>
  </tbody>
</table>
</div>

#### Highest *vocd-D* songs

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
      <td>2,447.52</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>1,460.79</td>
    </tr>
    <tr>
      <th>4</th>
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
      <td>1,100.15</td>
    </tr>
    <tr>
      <th>5</th>
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
      <td>1,015.47</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cianide</td>
      <td>Death/Doom Metal</td>
      <td>One-Thousand Ways to Die</td>
      <td>97</td>
      <td>92</td>
      <td>0.95</td>
      <td>9.34</td>
      <td>0.99</td>
      <td>97</td>
      <td>4.57</td>
      <td>835.13</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Pessimist</td>
      <td>Brutal Death/Black Metal</td>
      <td>Psychological Autopsy</td>
      <td>88</td>
      <td>84</td>
      <td>0.95</td>
      <td>8.95</td>
      <td>0.99</td>
      <td>88</td>
      <td>4.48</td>
      <td>806.54</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Schizo</td>
      <td>Thrash/Black Metal</td>
      <td>Mind K</td>
      <td>55</td>
      <td>53</td>
      <td>0.96</td>
      <td>7.15</td>
      <td>0.99</td>
      <td>55</td>
      <td>4.01</td>
      <td>719.35</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Autopsy</td>
      <td>Death Metal</td>
      <td>Ugliness and Secretions</td>
      <td>55</td>
      <td>53</td>
      <td>0.96</td>
      <td>7.15</td>
      <td>0.99</td>
      <td>55</td>
      <td>4.01</td>
      <td>717.19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Judas Priest</td>
      <td>Heavy Metal</td>
      <td>Solar Angels</td>
      <td>52</td>
      <td>50</td>
      <td>0.96</td>
      <td>6.93</td>
      <td>0.99</td>
      <td>52</td>
      <td>3.95</td>
      <td>668.13</td>
    </tr>
  </tbody>
</table>
</div>

#### Lowest MTLD songs (with more than ten words)

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
      <td>Doom/Death Metal (early), Depressive Rock/Meta...</td>
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
      <td>Lost Society</td>
      <td>Thrash Metal (early), Groove Metal (later)</td>
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
      <th>6</th>
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
      <th>7</th>
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
      <th>8</th>
      <td>The Great Kat</td>
      <td>Speed/Thrash Metal, Shred</td>
      <td>Metal Messiah</td>
      <td>83</td>
      <td>20</td>
      <td>0.24</td>
      <td>2.20</td>
      <td>0.68</td>
      <td>6.55</td>
      <td>1.88</td>
      <td>2.97</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Yob</td>
      <td>Stoner/Sludge/Doom Metal</td>
      <td>Lungs Reach</td>
      <td>22</td>
      <td>4</td>
      <td>0.18</td>
      <td>0.85</td>
      <td>0.45</td>
      <td>6.67</td>
      <td>1.90</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dornenreich</td>
      <td>Melodic/Symphonic Black Metal, Neofolk</td>
      <td>Im Fluss die Flammen</td>
      <td>72</td>
      <td>20</td>
      <td>0.28</td>
      <td>2.36</td>
      <td>0.70</td>
      <td>6.87</td>
      <td>1.93</td>
      <td>6.18</td>
    </tr>
  </tbody>
</table>
</div>

#### Lowest *vocd-D* songs

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
      <th>2</th>
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
      <th>3</th>
      <td>Nuit Noire</td>
      <td>Black Metal (early), Punk/Black Metal (later)</td>
      <td>I Am a Fairy</td>
      <td>150</td>
      <td>16</td>
      <td>0.11</td>
      <td>1.31</td>
      <td>0.55</td>
      <td>10.14</td>
      <td>2.32</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mortiis</td>
      <td>Ambient/Darkwave; Industrial Rock</td>
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
      <th>5</th>
      <td>Anthrax</td>
      <td>Speed/Thrash Metal; Groove Metal</td>
      <td>604</td>
      <td>52</td>
      <td>10</td>
      <td>0.19</td>
      <td>1.39</td>
      <td>0.58</td>
      <td>13.00</td>
      <td>2.56</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>6</th>
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
      <td>1.49</td>
    </tr>
    <tr>
      <th>7</th>
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
      <td>1.54</td>
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
      <td>1.56</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jesu</td>
      <td>Drone/Doom Metal/Shoegaze/Post-Rock</td>
      <td>Walk on Water</td>
      <td>90</td>
      <td>12</td>
      <td>0.13</td>
      <td>1.26</td>
      <td>0.55</td>
      <td>17.44</td>
      <td>2.86</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Suicidal Tendencies</td>
      <td>Thrash Metal/Crossover, Hardcore Punk</td>
      <td>Public Dissension</td>
      <td>217</td>
      <td>16</td>
      <td>0.07</td>
      <td>1.09</td>
      <td>0.52</td>
      <td>9.11</td>
      <td>2.21</td>
      <td>2.04</td>
    </tr>
  </tbody>
</table>
</div>

#### Highest MTLD bands

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
      <td>3544</td>
      <td>1867</td>
      <td>0.53</td>
      <td>31.36</td>
      <td>0.92</td>
      <td>3,544.00</td>
      <td>8.17</td>
      <td>182.72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Malignancy</td>
      <td>Brutal Technical Death Metal</td>
      <td>5530</td>
      <td>2094</td>
      <td>0.38</td>
      <td>28.16</td>
      <td>0.89</td>
      <td>3,110.13</td>
      <td>8.04</td>
      <td>184.68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Demolition Hammer</td>
      <td>Thrash Metal (early), Groove Metal (later)</td>
      <td>2845</td>
      <td>1665</td>
      <td>0.59</td>
      <td>31.22</td>
      <td>0.93</td>
      <td>2,845.00</td>
      <td>7.95</td>
      <td>352.84</td>
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
      <td>147.21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Moonsorrow</td>
      <td>Folk/Pagan/Black Metal</td>
      <td>7231</td>
      <td>2954</td>
      <td>0.41</td>
      <td>34.74</td>
      <td>0.90</td>
      <td>2,183.94</td>
      <td>7.69</td>
      <td>229.12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Brutality</td>
      <td>Death Metal</td>
      <td>3535</td>
      <td>1663</td>
      <td>0.47</td>
      <td>27.97</td>
      <td>0.91</td>
      <td>2,106.26</td>
      <td>7.65</td>
      <td>182.02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wormed</td>
      <td>Brutal/Technical Death Metal</td>
      <td>2007</td>
      <td>1103</td>
      <td>0.55</td>
      <td>24.62</td>
      <td>0.92</td>
      <td>2,007.00</td>
      <td>7.60</td>
      <td>199.26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mortal Decay</td>
      <td>Death Metal</td>
      <td>2257</td>
      <td>1121</td>
      <td>0.50</td>
      <td>23.60</td>
      <td>0.91</td>
      <td>1,997.74</td>
      <td>7.60</td>
      <td>170.85</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cormorant</td>
      <td>Melodic Death Metal (early); Progressive Folk/...</td>
      <td>5692</td>
      <td>2295</td>
      <td>0.40</td>
      <td>30.42</td>
      <td>0.89</td>
      <td>1,922.69</td>
      <td>7.56</td>
      <td>124.73</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Impaled</td>
      <td>Death Metal</td>
      <td>6706</td>
      <td>2697</td>
      <td>0.40</td>
      <td>32.93</td>
      <td>0.90</td>
      <td>1,813.13</td>
      <td>7.50</td>
      <td>161.05</td>
    </tr>
  </tbody>
</table>
</div>

#### Highest *vocd-D* bands

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
      <td>Absurd</td>
      <td>Black Metal/RAC, Pagan Black Metal</td>
      <td>154</td>
      <td>81</td>
      <td>0.53</td>
      <td>6.53</td>
      <td>0.87</td>
      <td>103.93</td>
      <td>4.64</td>
      <td>43.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Vinterland</td>
      <td>Melodic Black Metal</td>
      <td>706</td>
      <td>194</td>
      <td>0.27</td>
      <td>7.30</td>
      <td>0.80</td>
      <td>77.90</td>
      <td>4.36</td>
      <td>49.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Crypt of Kerberos</td>
      <td>Death/Doom Metal (early), Progressive Death Me...</td>
      <td>166</td>
      <td>87</td>
      <td>0.52</td>
      <td>6.75</td>
      <td>0.87</td>
      <td>166.00</td>
      <td>5.11</td>
      <td>50.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>October Falls</td>
      <td>Folk/Ambient, Dark Metal</td>
      <td>1896</td>
      <td>581</td>
      <td>0.31</td>
      <td>13.34</td>
      <td>0.84</td>
      <td>277.63</td>
      <td>5.63</td>
      <td>52.62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nuit Noire</td>
      <td>Black Metal (early), Punk/Black Metal (later)</td>
      <td>963</td>
      <td>310</td>
      <td>0.32</td>
      <td>9.99</td>
      <td>0.84</td>
      <td>55.39</td>
      <td>4.01</td>
      <td>54.28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Earth</td>
      <td>Drone/Doom Metal (early); Psychedelic/Post-Roc...</td>
      <td>355</td>
      <td>136</td>
      <td>0.38</td>
      <td>7.22</td>
      <td>0.84</td>
      <td>161.29</td>
      <td>5.08</td>
      <td>56.41</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Interment</td>
      <td>Death Metal</td>
      <td>217</td>
      <td>99</td>
      <td>0.46</td>
      <td>6.72</td>
      <td>0.85</td>
      <td>167.10</td>
      <td>5.12</td>
      <td>57.72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Negative Plane</td>
      <td>Black Metal</td>
      <td>2231</td>
      <td>728</td>
      <td>0.33</td>
      <td>15.41</td>
      <td>0.85</td>
      <td>284.91</td>
      <td>5.65</td>
      <td>60.08</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Om</td>
      <td>Stoner/Drone/Doom Metal</td>
      <td>2122</td>
      <td>663</td>
      <td>0.31</td>
      <td>14.39</td>
      <td>0.85</td>
      <td>223.50</td>
      <td>5.41</td>
      <td>62.01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ghost Bath</td>
      <td>Depressive/Post-Black Metal</td>
      <td>316</td>
      <td>157</td>
      <td>0.50</td>
      <td>8.83</td>
      <td>0.88</td>
      <td>159.56</td>
      <td>5.07</td>
      <td>62.26</td>
    </tr>
  </tbody>
</table>
</div>


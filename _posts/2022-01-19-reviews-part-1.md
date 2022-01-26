---
layout: post
title: "Reviews of Heavy Metal Albums - Part 1: Exploratory Analysis"
categories: jekyll update
permalink: /projects/heavy-metal-analysis/reviews-part-1
summary: |
  A data-driven discussion of the history and global demographics of the heavy metal music industry
  and its many genres, leveraging statistical information extracted from album reviews on Metal-Archives.
---

This article is a part of my [heavy metal lyrics project](/projects/heavy-metal-analysis.html).
If you're interested in seeing the code, check out the
[original notebook](https://github.com/pdqnguyen/metallyrics/blob/main/analyses/reviews/reviews1.ipynb).
In the [next article](./reviews-part-2.html)
we'll use machine learning to perform review score prediction from album review text.

## Dataset

The dataset consists of nearly 50,000 album reviews extracted from Metal-Archives (MA from here on).
Each review is comprised of a block of text and a score ranging from 0 to 100.
The reviews cover 10,100 albums produced by 1,787 bands.
I collected the data awhile ago, so keep in mind it's a little out-of-date.
The distribution of scores, shown below, is very top-heavy, with an average 79% and median of 85%.
There are peaks in the distribution at multiples of ten and five due to the propensity of reviewers to round out their scores.
Over a fifth of the reviews gave scores of at least 95%, and nearly a tenth of reviews gave a full 100%.

![png](/assets/images/heavy-metal-lyrics/reviews/reviews_hist.png)

## Weighted-average album score

Compare albums by simply looking at average review ratings fails to consider each album's popularity (or infamy).
This is important when the number of reviews per album vary dramatically.
Just like looking at product reviews, we naturally assign more weight to album review scores
that are averaged from the experiences of many people.

As you can see below, there are countless albums on MA with only a single, 100% review.
It doesn't make much sense to say these albums are all better than the most popular albums.
I could apply a minimum number of reviews required to consider an album's review score legitimate,
but this shrinks down the dataset considerably and still weighs albums near the cutoff number and
near the maximum equally.

Instead, I will use a weighted-average score that treats individual reviews for an album as "evidence"
that the album ought to deviate from the population mean (of 79%).
Ideally, this method would distinguish good albums based on them having many positive reviews, not just a handful.
Likewise, it should help us reveal which albums draw a consensus of disdain from the MA community.

<details>
<summary>Show top average reviews</summary>
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
      <th>band_name</th>
      <th>name</th>
      <th>review_avg</th>
      <th>review_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Nightmare</td>
      <td>Cosmovision</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spiritual Beggars</td>
      <td>Demons</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Plasmatics</td>
      <td>Coup d'Ã‰tat</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Moloch</td>
      <td>Человечье слишком овечье</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Moloch</td>
      <td>Meine alte Melancholie</td>
      <td>100</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</details>
<br>

<details>
<summary>Show lowest average reviews</summary>
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
      <th>band_name</th>
      <th>name</th>
      <th>review_avg</th>
      <th>review_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Boris</td>
      <td>Warpath</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kult ov Azazel</td>
      <td>Destroying the Sacred</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Car Door Dick Smash</td>
      <td>Garbage</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Opera IX</td>
      <td>Anphisbena</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Intronaut</td>
      <td>The Direction of Last Things</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</details>

#### IMDB weighted rating

Supposedly IMDb rates content using the following weighted averaging scheme:

$$ W_i = \frac{R_iv_i + Cm}{v_i + m} $$

where $$R_i$$ and $$v_i$$ are the average review score and number of reviews for album $$i$$,
$$C$$ represents the a priori rating which I assume to be the population mean ($$C = \sum_{i=1}^n R_i$$),
and $$m$$ is a tunable parameter representing the number of reviews for an album to be considered in a Top-X list.
In this case I choose $$m$$ to be the $$\alpha$$-th percentile album review count, and instead tune $$\alpha$$.

If $$\alpha = 0$$, then $$m = 0$$, so and we recover the raw sample average: $$W_i = R_i$$.
The larger $$\alpha$$ is, the more heavily we weigh the number of reviews.
If $$\alpha = 1$$, then $$m = \max(v_i)$$. This means that an album with no reviews starts at $$W_i = C$$
and is incrementally pulled away from $$C$$ in the direction of new reviews,
each review having a weaker pull that the last.

This method is a more Bayesian-like approach to quantifying album scores (as opposed to the
purely frequentist method of just using $$R_i$$). A more sophisticated approach could further weigh samples
by the standard deviation of their review scores, such that a sample with a low standard deviation is
given a stronger pull from $$C$$.

The choice of $$\alpha$$ is up to us. Since I'm really interested in seeing the most popular/infamous albums,
I'll set it to max value of 1. For bands, since the number of reviews per band is much higher,
I'll set it to a lower value, say 0.75.


#### Weighted score distribution

Using a high $$\alpha$$ results in a very sharp distribution relative to the original distribution,
since it weighs the population mean maximally.

![png](/assets/images/heavy-metal-lyrics/reviews/weighted_hist.png)



#### Best and worst albums

Metallica rides the lightning to the top of the weighted score ranking with their 1984 sophomore record.
Most of the top-20 albums hail from the 80s and 90s, a testament to their reputations as heavy metal classics.

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
      <th>band_name</th>
      <th>name</th>
      <th>year</th>
      <th>band_genre</th>
      <th>review_num</th>
      <th>review_avg</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Metallica</td>
      <td>Ride the Lightning</td>
      <td>1984</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>30</td>
      <td>93.93</td>
      <td>85.16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Death</td>
      <td>Symbolic</td>
      <td>1995</td>
      <td>Death Metal (early), Progressive Death Metal (...</td>
      <td>30</td>
      <td>93.00</td>
      <td>84.76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Black Sabbath</td>
      <td>Paranoid</td>
      <td>1970</td>
      <td>Heavy/Doom Metal</td>
      <td>30</td>
      <td>92.87</td>
      <td>84.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sepultura</td>
      <td>Beneath the Remains</td>
      <td>1989</td>
      <td>Death/Thrash Metal (early), Nu-Metal, Groove/T...</td>
      <td>28</td>
      <td>93.32</td>
      <td>84.65</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Iron Maiden</td>
      <td>Powerslave</td>
      <td>1984</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>28</td>
      <td>92.36</td>
      <td>84.25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Entombed</td>
      <td>Left Hand Path</td>
      <td>1990</td>
      <td>Death Metal/Death 'n' Roll</td>
      <td>16</td>
      <td>98.06</td>
      <td>84.14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bolt Thrower</td>
      <td>Realm of Chaos: Slaves to Darkness</td>
      <td>1989</td>
      <td>Death Metal</td>
      <td>22</td>
      <td>94.23</td>
      <td>84.13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Symphony X</td>
      <td>The Divine Wings of Tragedy</td>
      <td>1996</td>
      <td>Progressive Power Metal</td>
      <td>18</td>
      <td>96.33</td>
      <td>84.09</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Burzum</td>
      <td>Hvis lyset tar oss</td>
      <td>1994</td>
      <td>Black Metal, Ambient</td>
      <td>30</td>
      <td>91.40</td>
      <td>84.07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ahab</td>
      <td>The Call of the Wretched Sea</td>
      <td>2006</td>
      <td>Funeral Doom Metal</td>
      <td>20</td>
      <td>94.95</td>
      <td>84.04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Death</td>
      <td>Human</td>
      <td>1991</td>
      <td>Death Metal (early), Progressive Death Metal (...</td>
      <td>26</td>
      <td>92.35</td>
      <td>84.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Morbid Angel</td>
      <td>Altars of Madness</td>
      <td>1989</td>
      <td>Death Metal</td>
      <td>30</td>
      <td>91.17</td>
      <td>83.97</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Katatonia</td>
      <td>Dance of December Souls</td>
      <td>1993</td>
      <td>Doom/Death Metal (early), Depressive Rock/Meta...</td>
      <td>19</td>
      <td>95.21</td>
      <td>83.93</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Morbid Saint</td>
      <td>Spectrum of Death</td>
      <td>1990</td>
      <td>Thrash Metal</td>
      <td>20</td>
      <td>94.50</td>
      <td>83.89</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Black Sabbath</td>
      <td>Master of Reality</td>
      <td>1971</td>
      <td>Heavy/Doom Metal</td>
      <td>23</td>
      <td>93.04</td>
      <td>83.86</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Artillery</td>
      <td>By Inheritance</td>
      <td>1990</td>
      <td>Thrash Metal</td>
      <td>20</td>
      <td>94.30</td>
      <td>83.82</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Judas Priest</td>
      <td>Stained Class</td>
      <td>1978</td>
      <td>Heavy Metal</td>
      <td>20</td>
      <td>94.25</td>
      <td>83.80</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Dissection</td>
      <td>Storm of the Light's Bane</td>
      <td>1995</td>
      <td>Melodic Black Metal (early), Melodic Death Met...</td>
      <td>30</td>
      <td>90.73</td>
      <td>83.79</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Blind Guardian</td>
      <td>Somewhere Far Beyond</td>
      <td>1992</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>20</td>
      <td>94.10</td>
      <td>83.75</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Judas Priest</td>
      <td>Painkiller</td>
      <td>1990</td>
      <td>Heavy Metal</td>
      <td>27</td>
      <td>91.26</td>
      <td>83.69</td>
    </tr>
  </tbody>
</table>
</div>

At the other end of the ranking, Waking the Cadaver's "Perverse Recollections of a Necromangler" is a clear outlier,
scoring significantly below the next worst album.
Metallica's infamous St. Anger does indeed get thrown to the gutters by the weighted scoring method.
Most of the worst albums were made by highly talented artists whose fans were expecting so much more.
It almost seems like a rite of passage to disappoint your fans after a decade or two of consistency.
Next we'll look at which bands have done well at _not_ pissing of the MA community.

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
      <th>band_name</th>
      <th>name</th>
      <th>year</th>
      <th>band_genre</th>
      <th>review_num</th>
      <th>review_avg</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Waking the Cadaver</td>
      <td>Perverse Recollections of a Necromangler</td>
      <td>2007</td>
      <td>Slam/Brutal Death Metal/Deathcore (early), Dea...</td>
      <td>35</td>
      <td>24.23</td>
      <td>53.21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Morbid Angel</td>
      <td>Illud Divinum Insanus</td>
      <td>2011</td>
      <td>Death Metal</td>
      <td>28</td>
      <td>35.82</td>
      <td>60.97</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hellyeah</td>
      <td>Hellyeah</td>
      <td>2007</td>
      <td>Groove Metal</td>
      <td>13</td>
      <td>10.69</td>
      <td>61.93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thrash or Die</td>
      <td>Poser Holocaust</td>
      <td>2011</td>
      <td>Thrash Metal</td>
      <td>14</td>
      <td>17.86</td>
      <td>62.84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cryptopsy</td>
      <td>The Unspoken King</td>
      <td>2008</td>
      <td>Brutal/Technical Death Metal, Deathcore (2008)</td>
      <td>18</td>
      <td>31.00</td>
      <td>63.81</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Metallica</td>
      <td>St. Anger</td>
      <td>2003</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>31</td>
      <td>45.00</td>
      <td>63.92</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Six Feet Under</td>
      <td>Graveyard Classics 2</td>
      <td>2004</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>10</td>
      <td>9.30</td>
      <td>64.72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>In Flames</td>
      <td>Soundtrack to Your Escape</td>
      <td>2004</td>
      <td>Melodic Death Metal (early), Melodic Groove Me...</td>
      <td>20</td>
      <td>39.10</td>
      <td>65.42</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Winds of Plague</td>
      <td>Decimate the Weak</td>
      <td>2008</td>
      <td>Symphonic Deathcore</td>
      <td>19</td>
      <td>37.84</td>
      <td>65.46</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Celtic Frost</td>
      <td>Cold Lake</td>
      <td>1988</td>
      <td>Thrash/Death/Black Metal (early), Gothic/Doom ...</td>
      <td>14</td>
      <td>31.79</td>
      <td>66.45</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Queensrÿche</td>
      <td>American Soldier</td>
      <td>2009</td>
      <td>Heavy/Power/Progressive Metal</td>
      <td>12</td>
      <td>27.17</td>
      <td>66.71</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Slayer</td>
      <td>God Hates Us All</td>
      <td>2001</td>
      <td>Thrash Metal</td>
      <td>25</td>
      <td>48.36</td>
      <td>66.96</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Dimmu Borgir</td>
      <td>Abrahadabra</td>
      <td>2010</td>
      <td>Symphonic Black Metal</td>
      <td>27</td>
      <td>51.26</td>
      <td>67.57</td>
    </tr>
    <tr>
      <th>14</th>
      <td>In Flames</td>
      <td>Siren Charms</td>
      <td>2014</td>
      <td>Melodic Death Metal (early), Melodic Groove Me...</td>
      <td>16</td>
      <td>40.69</td>
      <td>67.75</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Burzum</td>
      <td>Dauði Baldrs</td>
      <td>1997</td>
      <td>Black Metal, Ambient</td>
      <td>20</td>
      <td>46.45</td>
      <td>67.87</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Machine Head</td>
      <td>Supercharger</td>
      <td>2001</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>11</td>
      <td>29.09</td>
      <td>67.90</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Megadeth</td>
      <td>Super Collider</td>
      <td>2013</td>
      <td>Speed/Thrash Metal (early/later); Heavy Metal/...</td>
      <td>19</td>
      <td>45.58</td>
      <td>67.95</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Massacre</td>
      <td>Promise</td>
      <td>1996</td>
      <td>Death Metal</td>
      <td>7</td>
      <td>7.86</td>
      <td>68.05</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Bathory</td>
      <td>Octagon</td>
      <td>1995</td>
      <td>Black/Viking Metal, Thrash Metal</td>
      <td>11</td>
      <td>29.91</td>
      <td>68.08</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Sepultura</td>
      <td>Roots</td>
      <td>1996</td>
      <td>Death/Thrash Metal (early), Nu-Metal, Groove/T...</td>
      <td>20</td>
      <td>47.20</td>
      <td>68.12</td>
    </tr>
  </tbody>
</table>
</div>

#### Best and worst bands

To accumulate a high weighted-average score, a band must put out many albums, each garnering many positive reviews.
Any below-average albums will drag a band's weighted score down.
Based on this metric, Death is the most successful metal band.
Given their discography this is no surprise.
These guys really didn't know how to put out a bad album.
Their lowest reviewed full-length record, "The Sound of Perseverance" averaged just 1% below the 79% global average,
while the rest sit above 81%.
There's quite a lot of variety the top few bands: Type O Negative, Agra, Candlemass, and Rush!

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
      <th>name</th>
      <th>genre</th>
      <th>review_num</th>
      <th>review_avg</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Death</td>
      <td>Death Metal (early), Progressive Death Metal (...</td>
      <td>170</td>
      <td>87.22</td>
      <td>81.87</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Type O Negative</td>
      <td>Gothic/Doom Metal</td>
      <td>75</td>
      <td>91.79</td>
      <td>81.51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Angra</td>
      <td>Power/Progressive Metal</td>
      <td>77</td>
      <td>90.83</td>
      <td>81.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Candlemass</td>
      <td>Epic Doom Metal</td>
      <td>109</td>
      <td>87.96</td>
      <td>81.38</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rush</td>
      <td>Hard Rock/Heavy Metal (early); Progressive Roc...</td>
      <td>182</td>
      <td>85.21</td>
      <td>81.34</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Moonsorrow</td>
      <td>Folk/Pagan/Black Metal</td>
      <td>72</td>
      <td>91.06</td>
      <td>81.33</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Immolation</td>
      <td>Death Metal</td>
      <td>96</td>
      <td>88.27</td>
      <td>81.27</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Blind Guardian</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>193</td>
      <td>84.63</td>
      <td>81.23</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bolt Thrower</td>
      <td>Death Metal</td>
      <td>120</td>
      <td>86.62</td>
      <td>81.21</td>
    </tr>
    <tr>
      <th>10</th>
      <td>W.A.S.P.</td>
      <td>Heavy Metal/Hard Rock</td>
      <td>100</td>
      <td>87.35</td>
      <td>81.14</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Primordial</td>
      <td>Celtic Folk/Black Metal</td>
      <td>69</td>
      <td>90.01</td>
      <td>81.12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bathory</td>
      <td>Black/Viking Metal, Thrash Metal</td>
      <td>171</td>
      <td>84.40</td>
      <td>81.02</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Alice in Chains</td>
      <td>Heavy Metal/Grunge</td>
      <td>76</td>
      <td>88.49</td>
      <td>81.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>King Diamond</td>
      <td>Heavy Metal</td>
      <td>123</td>
      <td>85.56</td>
      <td>80.99</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Manilla Road</td>
      <td>Epic Heavy/Power Metal</td>
      <td>88</td>
      <td>87.35</td>
      <td>80.98</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Symphony X</td>
      <td>Progressive Power Metal</td>
      <td>133</td>
      <td>85.17</td>
      <td>80.98</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Insomnium</td>
      <td>Melodic Death Metal</td>
      <td>72</td>
      <td>88.56</td>
      <td>80.95</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Rotting Christ</td>
      <td>Grindcore, Black Metal (early); Gothic Metal (...</td>
      <td>110</td>
      <td>85.81</td>
      <td>80.92</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Suffocation</td>
      <td>Brutal/Technical Death Metal</td>
      <td>104</td>
      <td>86.02</td>
      <td>80.91</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Katatonia</td>
      <td>Doom/Death Metal (early), Depressive Rock/Meta...</td>
      <td>141</td>
      <td>84.47</td>
      <td>80.85</td>
    </tr>
  </tbody>
</table>
</div>

Among the lowest-rated bands, with Six Feet Under earning the worst weighted score.
At least Waking the Cadaver manage to dodge last place this time.

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
      <th>name</th>
      <th>genre</th>
      <th>review_num</th>
      <th>review_avg</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Six Feet Under</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>121</td>
      <td>50.52</td>
      <td>72.74</td>
    </tr>
    <tr>
      <th>2</th>
      <td>In Flames</td>
      <td>Melodic Death Metal (early), Melodic Groove Me...</td>
      <td>184</td>
      <td>60.72</td>
      <td>73.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Machine Head</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>123</td>
      <td>57.51</td>
      <td>74.31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waking the Cadaver</td>
      <td>Slam/Brutal Death Metal/Deathcore (early), Dea...</td>
      <td>48</td>
      <td>32.35</td>
      <td>74.43</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Dimmu Borgir</td>
      <td>Symphonic Black Metal</td>
      <td>218</td>
      <td>66.33</td>
      <td>74.85</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Queensrÿche</td>
      <td>Heavy/Power/Progressive Metal</td>
      <td>158</td>
      <td>65.08</td>
      <td>75.41</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Whitechapel</td>
      <td>Deathcore</td>
      <td>75</td>
      <td>55.71</td>
      <td>75.74</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Lacuna Coil</td>
      <td>Gothic Metal/Rock (early); Alternative Rock (l...</td>
      <td>81</td>
      <td>60.35</td>
      <td>76.28</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Winds of Plague</td>
      <td>Symphonic Deathcore</td>
      <td>34</td>
      <td>38.79</td>
      <td>76.32</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Annihilator</td>
      <td>Technical Speed/Thrash Metal (early); Groove/T...</td>
      <td>172</td>
      <td>69.13</td>
      <td>76.39</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Metallica</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>313</td>
      <td>72.42</td>
      <td>76.40</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hellyeah</td>
      <td>Groove Metal</td>
      <td>22</td>
      <td>23.73</td>
      <td>76.61</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Killswitch Engage</td>
      <td>Metalcore</td>
      <td>83</td>
      <td>64.66</td>
      <td>76.97</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Pantera</td>
      <td>Glam/Heavy Metal (early), Groove Metal (later)</td>
      <td>155</td>
      <td>70.56</td>
      <td>77.02</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Trivium</td>
      <td>Metalcore, Thrash/Heavy Metal</td>
      <td>69</td>
      <td>62.64</td>
      <td>77.04</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Lamb of God</td>
      <td>Groove Metal/Metalcore</td>
      <td>102</td>
      <td>67.60</td>
      <td>77.10</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Megadeth</td>
      <td>Speed/Thrash Metal (early/later); Heavy Metal/...</td>
      <td>344</td>
      <td>74.42</td>
      <td>77.16</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Iced Earth</td>
      <td>Power/Thrash Metal</td>
      <td>175</td>
      <td>71.91</td>
      <td>77.21</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Manowar</td>
      <td>Heavy/Power Metal</td>
      <td>160</td>
      <td>71.87</td>
      <td>77.34</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Arch Enemy</td>
      <td>Melodic Death Metal</td>
      <td>129</td>
      <td>70.58</td>
      <td>77.34</td>
    </tr>
  </tbody>
</table>
</div>

#### Best and worst genres

Doom metal tops the chart of total album score, followed closely by black and progressive metal.
There seems to be a preference for genres that emphasize long, instrumentally-focused song structures and fewer lyrics.
I can see how those genres are hard to hate, and therefore garner higher average reviews.
Reviewers might also find it easier to assess the quality of the albums on the basis of instrumentation.
These genres are also probably easier to listen to frequently enough to inspire writing reviews.

I'm pleasantly surprised to see progressive metal so high up.
Even in its unweighted average score is higher than I would have expected given some of the reviews I've seen
on my favorite prog metal bands.
I believe this is the result of "progressive" being more often used as a adjective prepended to other genres,
rather than as a genre itself.
The same could probably be said about avant-garde, which I imagine is being aided by its prevalence
as a sub-genre of black metal.

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
      <th>genre</th>
      <th>review_avg</th>
      <th>review_num</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>doom</td>
      <td>83.26</td>
      <td>4710</td>
      <td>79.78</td>
    </tr>
    <tr>
      <th>2</th>
      <td>progressive</td>
      <td>81.71</td>
      <td>5637</td>
      <td>79.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>black</td>
      <td>80.32</td>
      <td>11619</td>
      <td>79.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>folk</td>
      <td>81.98</td>
      <td>2157</td>
      <td>79.11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>atmospheric</td>
      <td>83.05</td>
      <td>1305</td>
      <td>79.05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>avant-garde</td>
      <td>83.90</td>
      <td>974</td>
      <td>79.02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>epic</td>
      <td>83.36</td>
      <td>895</td>
      <td>78.97</td>
    </tr>
    <tr>
      <th>8</th>
      <td>viking</td>
      <td>83.40</td>
      <td>843</td>
      <td>78.95</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sludge</td>
      <td>83.11</td>
      <td>818</td>
      <td>78.93</td>
    </tr>
    <tr>
      <th>10</th>
      <td>speed</td>
      <td>79.77</td>
      <td>3563</td>
      <td>78.91</td>
    </tr>
  </tbody>
</table>
</div>

At the bottom, groove metal, metalcore, and deathcore are perhaps to no one's surprise the lowest-scoring genres on MA.
Whether you like them or not, it seems they simply don't match the preferences of the typical MA reviewer.
A surprises here is thrash metal, with an average under 78% with nearly 9,000 reviews.
I'm guessing that, despite having plenty of highly praised artists,
the genre is stylistically saturated due to its consistent popularity over the decades.

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
      <th>genre</th>
      <th>review_avg</th>
      <th>review_num</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>groove</td>
      <td>70.37</td>
      <td>2880</td>
      <td>77.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>metalcore</td>
      <td>68.40</td>
      <td>1137</td>
      <td>77.99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>deathcore</td>
      <td>65.29</td>
      <td>782</td>
      <td>78.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alternative</td>
      <td>68.12</td>
      <td>616</td>
      <td>78.30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nu-metal</td>
      <td>65.68</td>
      <td>453</td>
      <td>78.33</td>
    </tr>
    <tr>
      <th>6</th>
      <td>melodic</td>
      <td>77.42</td>
      <td>5052</td>
      <td>78.39</td>
    </tr>
    <tr>
      <th>7</th>
      <td>symphonic</td>
      <td>76.61</td>
      <td>2678</td>
      <td>78.39</td>
    </tr>
    <tr>
      <th>8</th>
      <td>thrash</td>
      <td>77.95</td>
      <td>8754</td>
      <td>78.43</td>
    </tr>
    <tr>
      <th>9</th>
      <td>rock</td>
      <td>77.69</td>
      <td>5012</td>
      <td>78.46</td>
    </tr>
    <tr>
      <th>10</th>
      <td>slam</td>
      <td>63.74</td>
      <td>209</td>
      <td>78.51</td>
    </tr>
  </tbody>
</table>
</div>

#### Best and worse countries

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
      <th>country_of_origin</th>
      <th>review_num</th>
      <th>review_avg</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>United Kingdom</td>
      <td>4804</td>
      <td>80.30</td>
      <td>79.52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>630</td>
      <td>84.76</td>
      <td>79.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Japan</td>
      <td>588</td>
      <td>82.78</td>
      <td>79.41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>4445</td>
      <td>79.82</td>
      <td>79.41</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Greece</td>
      <td>585</td>
      <td>82.00</td>
      <td>79.39</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ireland</td>
      <td>184</td>
      <td>85.67</td>
      <td>79.36</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Poland</td>
      <td>846</td>
      <td>80.70</td>
      <td>79.36</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Australia</td>
      <td>600</td>
      <td>80.51</td>
      <td>79.33</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Netherlands</td>
      <td>928</td>
      <td>80.06</td>
      <td>79.33</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ukraine</td>
      <td>313</td>
      <td>81.27</td>
      <td>79.33</td>
    </tr>
  </tbody>
</table>
</div>

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
      <th>country_of_origin</th>
      <th>review_num</th>
      <th>review_avg</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>16069</td>
      <td>77.51</td>
      <td>78.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Italy</td>
      <td>988</td>
      <td>76.50</td>
      <td>79.13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Finland</td>
      <td>2847</td>
      <td>78.77</td>
      <td>79.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>1884</td>
      <td>78.85</td>
      <td>79.24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Brazil</td>
      <td>567</td>
      <td>78.47</td>
      <td>79.26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Belgium</td>
      <td>218</td>
      <td>77.71</td>
      <td>79.27</td>
    </tr>
    <tr>
      <th>7</th>
      <td>International</td>
      <td>96</td>
      <td>76.32</td>
      <td>79.27</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Switzerland</td>
      <td>406</td>
      <td>78.61</td>
      <td>79.27</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Philippines</td>
      <td>19</td>
      <td>67.79</td>
      <td>79.28</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Spain</td>
      <td>195</td>
      <td>78.34</td>
      <td>79.28</td>
    </tr>
  </tbody>
</table>
</div>

## Geographic distribution of albums and genres

#### Countries of origin


<details>
<summary>Show code</summary>
{% highlight python %}
countries_num_albums = df_albums.groupby('band_country_of_origin').apply(len).sort_values(ascending=False)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.pie(
    countries_num_albums.values,
    labels=[('{}: {}'.format(country, num_albums) if num_albums > 0.01 * countries_num_albums.sum() else '')
            for country, num_albums in countries_num_albums.items()],
    wedgeprops=dict(width=0.1),
    startangle=90,
    counterclock=False,
    textprops={'fontsize': 16},
)
ax.set_aspect('equal')
ax.set_title('Number of albums from each country')
plt.show()
{% endhighlight %}
</details>
<br>

    
![png](/assets/images/heavy-metal-lyrics/reviews/donut_albums.png)
    


#### U.S. states of origin

<details>
<summary>Show code</summary>
{% highlight python %}
df_states = pd.read_csv('../../data/state_population/2019_Census_US_Population_Data_By_State_Lat_Long.csv')
state_pops = pd.Series(df_states['POPESTIMATE2019'].values, index=df_states['STATE'].rename('state'))
df_usa = df_albums[df_albums['band_country_of_origin'] == 'United States'].copy()
pattern = '(' + '|'.join(state_pops.index) + ')'
df_usa['state'] = df_usa['band_location'].str.extract(pattern)
states_num_albums = df_usa.groupby('state').apply(len).sort_values(ascending=False)
states_num_albums_density = states_num_albums / state_pops * 1e6
states_num_albums_density = states_num_albums_density.fillna(0).round(2).sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.pie(
    states_num_albums.values,
    labels=[('{}: {}'.format(state, num_albums) if num_albums > 0.013 * states_num_albums.sum() else '')
            for state, num_albums in states_num_albums.items()],
    wedgeprops=dict(width=0.1),
    startangle=90,
    counterclock=False,
    textprops={'fontsize': 14},
)
ax1.set_aspect('equal')
ax1.set_title("Number of albums\nfrom each U.S. state")
ax2.pie(
    states_num_albums_density.values,
    labels=[('{}: {}'.format(state, num_albums) if num_albums > 0.015 * states_num_albums_density.sum() else '')
            for state, num_albums in states_num_albums_density.items()],
    wedgeprops=dict(width=0.1),
    startangle=90,
    counterclock=False,
    textprops={'fontsize': 14},
)
ax2.set_aspect('equal')
ax2.set_title("Number of albums per million\npeople in each U.S. state")
plt.show()
{% endhighlight %}
</details>
<br>


    
![png](/assets/images/heavy-metal-lyrics/reviews/donut_states.png)
    


#### Top countries in each genre


<details>
<summary>Show code</summary>
{% highlight python %}
genres_num_albums = df_albums[[col for col in df_albums.columns if 'genre_' in col]].sum(0).sort_values(ascending=False)[:10]
fig, ax = plt.subplots(5, 2, figsize=(16, 35))
fig.suptitle("Number of albums from each country, by genre", y=0.92)
ax = ax.flatten()
for i, col in enumerate(genres_num_albums.index):
    countries_genre_num_albums = df_albums[df_albums[col] > 0].groupby('band_country_of_origin').apply(len).sort_values(ascending=False)
    ax[i].pie(
        countries_genre_num_albums.values,
        labels=[('{}: {}'.format(country, num_albums) if num_albums > 0.02 * countries_genre_num_albums.sum() else '')
                for country, num_albums in countries_genre_num_albums.items()],
        wedgeprops=dict(width=0.1),
        startangle=90,
        counterclock=False,
        textprops={'fontsize': 14},
    )
    ax[i].set_aspect('equal')
    ax[i].set_title(f'Genre: {col[6:]}')
plt.show()
{% endhighlight %}
</details>
<br>


    
![png](/assets/images/heavy-metal-lyrics/reviews/donut_genres.png)
    


#### Top genres in each country


<details>
<summary>Show code</summary>
{% highlight python %}
min_albums = 100
ncols = 3
country_albums = df_albums.groupby('band_country_of_origin').apply(len)
countries = country_albums[country_albums > min_albums].index
nrows = int(np.ceil(len(countries) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.suptitle("Album counts of most popular genres by country", y=0.91)
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i >= len(countries):
        ax.set_axis_off()
        continue
    country = countries[i]
    df_country = df_albums[df_albums['band_country_of_origin'] == country]
    popular_genres = []
    albums_in_genre = []
    for genre in genres:
        num_albums = df_country['genre_' + genre].sum()
        popular_genres.append(genre)
        albums_in_genre.append(num_albums)
    x, y = [
        list(k[::-1]) for k in 
        zip(
            *sorted(
                zip(popular_genres, albums_in_genre),
                key=lambda pair: pair[1]
            )
        )
    ]
    ax.pie(
        y,
        labels=[('{}: {}'.format(country, num_albums) if num_albums > 0.02 * sum(y) else '')
                for country, num_albums in zip(x, y)],
        wedgeprops=dict(width=0.2),
        startangle=90,
        counterclock=False,
        textprops={'fontsize': 12},
    )
    ax.set_aspect('equal')
    ax.set_title(country)
plt.show()
{% endhighlight %}
</details>
<br>


    
![png](/assets/images/heavy-metal-lyrics/reviews/donut_countries.png)
    


## Decline of top-rated bands

If we grab a bunch of artists who've produced very high-scoring albums,
we can see the decline of these bands since their debut albums.
Most had at least one album better than their debut,
but it looks as if only Bolt Thrower has averaged above where they started.


<details>
<summary>Show code</summary>
{% highlight python %}
min_albums = 5
num_bands = 10
bands_album_counts = df_albums.groupby('band_name').apply(len)
bands_max_scores = df_albums.groupby('band_name')['review_weighted'].max()[bands_album_counts >= min_albums]
bands_top = bands_max_scores.sort_values(ascending=False).iloc[:num_bands].index
band_scores = {}
for band in bands_top:
    df_band = df_albums[df_albums['band_name'] == band].sort_values('year')
    band_scores[band] = (df_band['year'].values, df_band['review_weighted'].values)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
cmap = plt.cm.hsv
ax.set_prop_cycle(color=[cmap(i) for i in np.linspace(0, 0.9, len(band_scores))])
for band, (years, scores) in band_scores.items():
    ax.plot(years - years.min(), scores - scores[0], label=band)
ax.set_title(f"Album reviews of bands with a top-{num_bands} album")
ax.set_xlabel("Years since debut album")
ax.set_ylabel("Weighted\nscore\nrelative\nto debut\nalbum", rotation=0, labelpad=60, y=0.25)
ax.legend(bbox_to_anchor=(1, 1))
plt.show()
{% endhighlight %}
</details>
<br>


    
![png](/assets/images/heavy-metal-lyrics/reviews/decline.png)
    


## Global album trends

Now for a more comprehensive look.
We can see that the number of metal albums grew up until the late 2000's,
and has been in a sharp decline since.
Popularity, as determined by the annual average of weighted-average scores (fourth plot),
peaks towards the end of the 80s, about the same time that many of those top bands
from above were releasing their most iconic albums.
As the metal scene saturated, the quality dropped,
with scores hitting a low point that coincides with the peak in number of albums.

An alternative explanation could be that nostalgia inflates reviewers' opinions of earlier albums,
especially in those bands above whose quality floundered over the years.
There's little that can be done to test this hypothesis since all of the reviews of early albums here are written with hindsight,
but maybe in a few more years we can take another look and see if review sentiment shifts upward
as 2000s bands become the new nostalgia.

Although album production has nose-dived in the last few years,
and review rates for newer albums still trail behind those that have
been around for a while, recent records are performing much better
than those from the 2000s and early 2010s.
The annual weighted average is almost back to matching that of early 80s albums.


<details>
<summary>Show code</summary>
{% highlight python %}
df_years = df[df.columns[df.columns.str.match('^(band_|album_|genre_)')]].groupby('album_year').first()
df_years['album_num'] = df.groupby(['album_year', 'album_name']).first().reset_index().groupby('album_year').apply(len)
df_years['review_avg'] = df.groupby('album_year')['review_score'].mean()
df_years['review_num'] = df.groupby('album_year').apply(len)
df_years.reset_index(inplace=True)
df_years.rename(columns={'album_year': 'year'}, inplace=True)
df_years['review_weighted'] = weighted_scores(df_years['review_avg'], df_years['review_num'], alpha=1)
{% endhighlight %}

{% highlight python %}
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
fig.subplots_adjust(hspace=0.3)
fig.suptitle("Global trends in weighted scores")
ax1.plot(df_years['year'], df_years['album_num'], lw=3)
ax2.plot(df_years['year'], df_years['review_num'] / df_years['album_num'], lw=3)
ax3.plot(df_years['year'], df_years['review_avg'], lw=3)
ax4.plot(df_years['year'], df_years['review_weighted'], lw=3)
ax1.set_title("Albums recorded")
ax2.set_title("Reviews per album")
ax3.set_title("Average scores")
ax4.set_title("Weighted average scores")
fig.tight_layout()
plt.show()
{% endhighlight %}
</details>
<br>


    
![png](/assets/images/heavy-metal-lyrics/reviews/global.png)
    



<details>
<summary>Show code</summary>
{% highlight python %}
def padded_array(x, pad):
    return np.concatenate((np.ones(pad) * x[0], x, np.ones(pad) * x[-1]))

def smooth(x, kernel, w, pad):
    kernel /= kernel.sum()
    x_smooth = np.convolve(x, kernel, mode='same')
    std = np.sqrt(np.convolve((x - x_smooth)**2, kernel, mode='same'))
    if pad > 0:
        x_smooth = x_smooth[pad:-pad]
    if pad > 0:
        std = std[pad:-pad]
    return x_smooth, std

def gsmooth(x, w=1, pad='auto', kernel_threshold=1e-5):
    if w == 0:
        return x, np.zeros_like(x)
    if pad == 'auto':
        pad = w
    x_padded = padded_array(x, pad)
    kernel_x = np.linspace(-x.size, x.size, x_padded.size)
    sigma = w / (2 * np.sqrt(2 * np.log(2)))
    kernel = np.exp(-kernel_x**2 / (2 * sigma**2))
    kernel[kernel < kernel_threshold] = 0
    return smooth(x_padded, kernel, w, pad)
{% endhighlight %}
</details>
<br>

#### Yearly album output by genre

Although most genres contributed to the rise of metal in the 2000s,
black and death metal dominated the trend.
Before then, it was heavy metal and thrash metal that ruled supreme,
spearheading the 80s upswing that brought heavy metal into the public spotlight.


<details>
<summary>Show code</summary>
{% highlight python %}
min_albums = 500
smoothing = 5
genre_years = df_albums.groupby('year')[df_albums.columns[df_albums.columns.str.contains('genre_')]].sum()
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
for i, col in genre_years.iteritems():
    if col.sum() > min_albums:
        x = col.index
        y, y_std = gsmooth(col.values, smoothing)
        plt.plot(x, y, lw=3, label=i.replace('genre_', ''))
        plt.fill_between(x, y - 2 * y_std, y + 2 * y_std, alpha=0.1)
ax.set_xlabel("Year")
ax.set_ylabel("Annual\naverage of\nweighted-\naverage\nscores", rotation=0, labelpad=80, y=0.25)
if smoothing > 0:
    title = "Album production of genres with over {} albums ({}-year smoothing applied)".format(min_albums, smoothing)
else:
    title = "Album production of genres with over {} albums".format(min_albums)
ax.set_title(title)
ax.legend(bbox_to_anchor=(1, 1))
ax.grid(alpha=0.5)
{% endhighlight %}
</details>
<br>


    
![png](/assets/images/heavy-metal-lyrics/reviews/genre_albums.png)
    


#### Yearly average album score

Here we see both genre-specific and global trends in the annual average of weighted-average scores.
Early doom metal (led by Black Sabbath) is very highly rated but has converged towards the average since then.
Early black metal albums likewise had a high-scoring start followed by gradual decline.


<details>
<summary>Show code</summary>
{% highlight python %}
min_albums = 500
smoothing = 5
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
for genre in genres:
    df_genre = df_albums[df_albums['genre_' + genre] > 0]#[['year', 'review_weighted']].sort_values('year').reset_index(drop=True)
    if len(df_genre) > min_albums:
        series = df_genre.groupby('year')['review_weighted'].mean()
        x = series.index
        y, y_std = gsmooth(series.values, smoothing)
        plt.plot(x, y, lw=3, label=genre.replace('genre_', ''))
# avg = df_albums.groupby('year')['review_weighted'].mean()
# x = avg.index
# y, y_std = gsmooth(avg.values, smoothing)
# ax.plot(x, y, 'k--', lw=3, label='all genres')
# ax.plot(df_years['year'], df_years['review_weighted'], 'k--', lw=3, label='all genres')
ax.set_xlabel("Year")
ax.set_ylabel("Annual\naverage of\nweighted-\naverage\nscores", rotation=0, labelpad=80, y=0.25)
if smoothing > 0:
    title = "Weighted-average scores of genres with over {} albums ({}-year smoothing applied)".format(min_albums, smoothing)
else:
    title = "Weighted-average scores of genres with over {} albums".format(min_albums)
ax.set_title(title)
ax.legend(bbox_to_anchor=(1, 1))
ax.grid(alpha=0.5)
{% endhighlight %}
</details>
<br>


    
![png](/assets/images/heavy-metal-lyrics/reviews/genre_scores.png)
    


## Geographic trends

These plots show just how consistently American bands have dominated the metal scence over the decades.
The U.K. was the first nation to pull ahead of others in producing metal albums,
but with the rise of the thrash era in the late 80s,
the U.S.A. took the lead and never lost it,
despite the Scandinavian black/death metal scene pulling some attention back across the Atlantic.


<details>
<summary>Show code</summary>
{% highlight python %}
min_albums = 300
smoothing = 5
countries_albums = df_albums.groupby('band_country_of_origin').size()
countries = countries_albums[countries_albums > min_albums]
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
for country in countries.index:
    df_country = df_albums[df_albums['band_country_of_origin'] == country]
    series = df_country.groupby('year').size()
    x = series.index
    y, y_std = gsmooth(series.values, smoothing)
    ax.plot(x, y, lw=3, label=country)
    ax.fill_between(x, y - 2 * y_std, y + 2 * y_std, alpha=0.1)
ax.set_xlabel("Year")
ax.set_ylabel("Albums", rotation=0, labelpad=50)
if smoothing > 0:
    title = "Album production of countries with over {} albums ({}-year smoothing applied)".format(min_albums, smoothing)
else:
    title = "Album production of countries with over {} albums".format(min_albums)
ax.set_title(title)
ax.legend(bbox_to_anchor=(1, 1))
ax.grid(alpha=0.5)
{% endhighlight %}
</details>
<br>


    
![png](/assets/images/heavy-metal-lyrics/reviews/country_albums.png)
    


The passing of trends is even more clear when looking at the weighted-average scores.
With each spike in national averages comes a new shift between genres;
the U.K.-to-U.S.A. transition marks the rise of thrash,
the U.S.A. gives way in the mid 80s to Swedish death metal,
and the Swedes hand the baton to their black metal neighbors in the 90s.
Sadly no nation was up to the task of ushering forth a new frontier in metal when the Norwegian scene entered its decline.


<details>
<summary>Show code</summary>
{% highlight python %}
min_albums = 300
smoothing = 5
countries_albums = df_albums.groupby('band_country_of_origin').size()
countries = countries_albums[countries_albums > min_albums]
fig, ax = plt.subplots(1, 1, figsize=(13, 6))
for country in countries.index:
    df = df_albums[df_albums['band_country_of_origin'] == country]
    series = df.groupby('year')['review_weighted'].mean()
    x = series.index
    y, y_std = gsmooth(series.values, smoothing)
    ax.plot(x, y, lw=3, label=country)
ax.set_xlabel("Year")
ax.set_ylabel("National\naverage of\nweighted-\naverage\nalbum\nscores", rotation=0, labelpad=80, y=0.25)
if smoothing > 0:
    title = "Weighted-average scores of countries with over {} albums ({}-year smoothing applied)".format(min_albums, smoothing)
else:
    title ="Weighted-average scores of countries with over {} albums".format(min_albums)
ax.set_title(title)
ax.legend(bbox_to_anchor=(1, 1))
ax.grid(alpha=0.5)
plt.show()
{% endhighlight %}
</details>
<br>
    
![png](/assets/images/heavy-metal-lyrics/reviews/country_scores.png)
    


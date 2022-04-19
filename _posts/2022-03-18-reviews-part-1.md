---
layout: post
date: 2022-03-18
title: "Reviews of Heavy Metal Albums - Part 1: Exploratory Analysis"
categories: jekyll update
permalink: /projects/heavy-metal-analysis/reviews-part-1
summary: |
  A data-driven discussion of the history and global demographics of the heavy metal music industry and its many genres, leveraging statistical information extracted from album reviews on Metal-Archives.
---

This article is a part of my [heavy metal lyrics project](/projects/heavy-metal-analysis.html).
If you're interested in seeing the code, check out the
[original notebook](https://github.com/pdqnguyen/metallyrics/blob/main/analyses/reviews/reviews1.ipynb).
In the [next article](./reviews-part-2.html)
we'll use machine learning to perform review score prediction from album review text.
Below is an interactive historical map of metal artists and albums
reflecting the analyses shown here
([click here for full-size version](https://metal-lyrics-maps.herokuapp.com/){:target="_blank"}).

<script>
  function resizeIframe(obj) {
    obj.style.height = obj.contentWindow.document.documentElement.scrollHeight + 'px';
  }
</script>

<div style="overflow: scroll; width:100%; height:800px">
<iframe src="https://metal-lyrics-maps.herokuapp.com" title="Interactive maps" scrolling="no" 
style="width: 1600px; height: 1200px; border: 0px"></iframe>
</div>

## Summary

**Things we'll do:**

* <span class="strong-text">Implement a Bayesian-weighted-average scoring metric</span>
  for comparing albums with differing numbers of reviews.
* Look at what the [Metal-Archives](https://www.metal-archives.com/) community considers the
  <span class="strong-text">best and worst albums, bands, genres, and countries</span>
  based on weighted-average album scores.
* Compare the popularity of different metal genres around the world.
* <span class="strong-text">Identify historical trends in heavy metal</span>
  as told through album statistics and review scores.

## Table of Contents
1. [Dataset](#dataset)
1. [Weighted-average album score](#weighted-average-album-score)
1. [Geographic distribution of albums and genres](#geographic-distribution-of-albums-and-genres)
1. [Global album trends](#global-album-trends)
1. [Geographic trends](#geographic-trends)





## Dataset





The dataset consists of nearly <span class="strong-text">86,000 album reviews</span> extracted from Metal-Archives (MA).
Each review is comprised of a block of text and a score ranging from 0 to 100.
The reviews cover about <span class="strong-text">35,000 albums</span> produced by over
<span class="strong-text">18,000 bands</span>.









    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_7_0.png)

    


<span class="strong-text">Review traffic peaked around 2010 after a healthy rise throughout MA's early years.
Although the monthly rate has dropped since then, it's held rather steadily since.</span>















    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_11_0.png)

    


A few things to note about the distribution of scores:
* <span class="strong-text">The distribution is very top-heavy, with an average of 78.5% and median of 81%.</span>
* There are peaks at multiples of ten and five because reviewers often round out their scores.
* <span class="strong-text">Over a fifth of the reviews gave scores of at least 95%,</span>
  and <span class="strong-text">nearly a tenth gave a full 100%.</span>

## Split dataset by album, band, genre, and country

For the upcoming analyses, I split the data in various ways.



#### By album




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
      <th>band_genre</th>
      <th>year</th>
      <th>review_avg</th>
      <th>review_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Slayer</td>
      <td>Reign in Blood</td>
      <td>Thrash Metal</td>
      <td>1986</td>
      <td>84.75</td>
      <td>44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iron Maiden</td>
      <td>Iron Maiden</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>1980</td>
      <td>85.58</td>
      <td>40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Metallica</td>
      <td>Master of Puppets</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>1986</td>
      <td>80.64</td>
      <td>39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mayhem</td>
      <td>De Mysteriis Dom Sathanas</td>
      <td>Black Metal</td>
      <td>1994</td>
      <td>88.26</td>
      <td>38</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Iron Maiden</td>
      <td>The Number of the Beast</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>1982</td>
      <td>86.55</td>
      <td>38</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Megadeth</td>
      <td>Rust in Peace</td>
      <td>Speed/Thrash Metal (early/later); Heavy Metal/...</td>
      <td>1990</td>
      <td>92.31</td>
      <td>36</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Black Sabbath</td>
      <td>13</td>
      <td>Heavy/Doom Metal</td>
      <td>2013</td>
      <td>67.81</td>
      <td>36</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Wintersun</td>
      <td>Time I</td>
      <td>Symphonic Melodic Death Metal</td>
      <td>2012</td>
      <td>68.39</td>
      <td>36</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Metallica</td>
      <td>Ride the Lightning</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>1984</td>
      <td>93.37</td>
      <td>35</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Megadeth</td>
      <td>Countdown to Extinction</td>
      <td>Speed/Thrash Metal (early/later); Heavy Metal/...</td>
      <td>1992</td>
      <td>77.06</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>







<span class="strong-text">Slayer's Reign in Blood is the most-reviewed album</span>, having been reviewed 44 times,
with plenty of other famous classics making the top-ten.









#### By band










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
      <th>review_avg</th>
      <th>review_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Iron Maiden</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>80.94</td>
      <td>480</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Black Sabbath</td>
      <td>Heavy/Doom Metal</td>
      <td>81.99</td>
      <td>388</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Megadeth</td>
      <td>Speed/Thrash Metal (early/later); Heavy Metal/...</td>
      <td>74.94</td>
      <td>357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Judas Priest</td>
      <td>Heavy Metal</td>
      <td>78.94</td>
      <td>328</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Darkthrone</td>
      <td>Death Metal (early); Black Metal (mid); Black/...</td>
      <td>79.47</td>
      <td>289</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Metallica</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>72.11</td>
      <td>281</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Slayer</td>
      <td>Thrash Metal</td>
      <td>74.72</td>
      <td>273</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Overkill</td>
      <td>Thrash Metal, Thrash/Groove Metal</td>
      <td>79.04</td>
      <td>248</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Opeth</td>
      <td>Progressive Death Metal, Progressive Rock</td>
      <td>78.55</td>
      <td>244</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cannibal Corpse</td>
      <td>Death Metal</td>
      <td>80.31</td>
      <td>232</td>
    </tr>
  </tbody>
</table>
</div>


Aggregating over discographies, <span class="strong-text">Iron Maiden is the most-reviewed artist</span>.
With 480 album reviews submitted, they stand well above the others.


#### By genre

This split in genres is a little different from the rest because there can be multiple genres per row.



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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>death</td>
      <td>78.53</td>
      <td>27520</td>
    </tr>
    <tr>
      <th>2</th>
      <td>black</td>
      <td>79.20</td>
      <td>22283</td>
    </tr>
    <tr>
      <th>3</th>
      <td>thrash</td>
      <td>77.42</td>
      <td>14533</td>
    </tr>
    <tr>
      <th>4</th>
      <td>heavy</td>
      <td>78.56</td>
      <td>12576</td>
    </tr>
    <tr>
      <th>5</th>
      <td>power</td>
      <td>78.62</td>
      <td>11539</td>
    </tr>
    <tr>
      <th>6</th>
      <td>progressive</td>
      <td>81.11</td>
      <td>9355</td>
    </tr>
    <tr>
      <th>7</th>
      <td>doom</td>
      <td>81.45</td>
      <td>8923</td>
    </tr>
    <tr>
      <th>8</th>
      <td>melodic</td>
      <td>76.99</td>
      <td>8519</td>
    </tr>
    <tr>
      <th>9</th>
      <td>rock</td>
      <td>77.07</td>
      <td>7325</td>
    </tr>
    <tr>
      <th>10</th>
      <td>speed</td>
      <td>79.41</td>
      <td>4675</td>
    </tr>
  </tbody>
</table>
</div>

<span class="strong-text">Death metal and black metal albums receive the most reviews,
not surprisingly since they are the most popular genres in the dataset.</span>


#### By country of origin


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
      <th>review_avg</th>
      <th>review_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>United States</td>
      <td>77.49</td>
      <td>25433</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sweden</td>
      <td>78.82</td>
      <td>8095</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Germany</td>
      <td>78.58</td>
      <td>7745</td>
    </tr>
    <tr>
      <th>4</th>
      <td>United Kingdom</td>
      <td>79.56</td>
      <td>6760</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Finland</td>
      <td>78.32</td>
      <td>4617</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Norway</td>
      <td>79.56</td>
      <td>4398</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Canada</td>
      <td>78.67</td>
      <td>3417</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Italy</td>
      <td>76.47</td>
      <td>2571</td>
    </tr>
    <tr>
      <th>9</th>
      <td>France</td>
      <td>79.24</td>
      <td>2292</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Netherlands</td>
      <td>79.43</td>
      <td>1775</td>
    </tr>
  </tbody>
</table>
</div>

<span class="strong-text">The U.S. clearly leads the world in album production and reviews.</span>
There may be a slight U.S. bias in the data because it's an English-speaking website,
but the coverage of countries around the world is actually very comprehensive on Metal-Archives.


## Weighted-average album score

Comparing albums by simply looking at average review ratings may not be the best way to gauge an album's popularity (or infamy).
This is important when the number of reviews per album vary dramatically.
Just like looking at product reviews, when judging albums we naturally assign more weight to album review scores
that are averaged from the experiences of many people.

As you can see below, <span class="strong-text">there are plenty of albums on MA with only a single, 100% review.</span>
It doesn't make much sense to say these albums are all better than the most popular albums.
Likewise, <span class="strong-text">there are plenty of albums with only a single 0% review.</span>
The same can be seen when splitting the data by band.
I could apply a minimum number of reviews required to consider an album's review score legitimate,
but this shrinks down the dataset considerably and still weighs albums near the cutoff number and
near the maximum equally.

<span class="strong-text">Instead, I will use a weighted-average score that treats individual reviews for an album as "evidence"
that the album ought to deviate from the population mean (of 79%).</span>
Ideally, this method would distinguish good albums based on them having many positive reviews, not just a handful.
Likewise, it should help us reveal which albums draw a consensus of disdain from the MA community.








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
      <td>Beldam</td>
      <td>Still the Wretched Linger</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Extinct Dreams</td>
      <td>Фрагменты вечности</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pestilent</td>
      <td>Purgatory of Punishment</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ceremonial Castings</td>
      <td>Cthulhu</td>
      <td>100</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sorhin</td>
      <td>Apokalypsens ängel</td>
      <td>100</td>
      <td>1</td>
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
      <th>band_name</th>
      <th>name</th>
      <th>review_avg</th>
      <th>review_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Patrons of the Rotting Gate</td>
      <td>The Rose Coil</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lotus Circle</td>
      <td>Bottomless Vales and Boundless Floods</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lost Souls</td>
      <td>Fracture</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waverly Hills</td>
      <td>The Nurse</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cloak of Displacement</td>
      <td>This Is the Only Way</td>
      <td>0</td>
      <td>1</td>
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
      <th>name</th>
      <th>review_avg</th>
      <th>review_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Pimentola</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sunken</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nebula</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dying Humanity</td>
      <td>100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Dead Asylum</td>
      <td>100</td>
      <td>1</td>
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
      <th>name</th>
      <th>review_avg</th>
      <th>review_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>The Ungrateful</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Seventh Army</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AIAA 7</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Killer Fox</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Dismembered Engorgement</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




#### Rating albums using Bayesian-weighted averaging

Supposedly IMDb rates content using the following
[weighted averaging scheme](https://www.quora.com/How-does-IMDbs-rating-system-work):

$$ W_i = \frac{R_in_i + Cm}{n_i + m} $$

where $$R_i$$ and $$n_i$$ are average score and number of scores for a sample $$i$$
(a movie in the IMDb case, an album in our case),
$$C = \sum_{i=1}^n R_i$$ represents the average of the full collection of scores,
which I'll call the population mean,
and $$m$$ is a tunable threshold for the number of ratings required to be included in the Top-250 list (25,000).
(The page I linked uses $$v$$ for "votes" instead of $$n$$; I prefer this notation.)
The issue with this is that it does depend on our choice of $$m$$,
which would have to be tailored to the dataset.
One way to pick it is to choose some percentile of the album review count distribution.
Choosing a high percentile would give a higher number for $$m$$,
weighing the second term in the numerator more heavily.
This effectively gives more weight to the number of reviews,
since a larger $$n_i$$ is required to pull $$W_i$$ away from the population mean $$C$$.
The choice of $$m$$ therefore matters a lot, but it's chosen subjectively.
Ideally we should be weighing the population term in a way that reflects
how confident we are that the population mean describes individual samples.

This weighted averaging is inspired by Bayesian statistics:
the population parameter $$C$$ represents a prior belief about an album's true score,
and the weighted average updates that prior based on the observations $$R_i n_i$$.
We can define a weighted average more rigourously if we fully adopt a Bayesian framework.
Note that the album scores are distributed within the range 0-100,
much like probabilities of a binary process (e.g. flipping a coin).
Let's say an album has some "true" score that the reviewers are estimating,
like people guessing how likely a flipped coin will land on heads.
We might have some _prior_ belief about that how that coin might behave,
based on our understanding of how coins typically land.
We could then update that belief using our observations,
arriving at a _posterior_ estimate for the probability of landing heads.
In the case of the album reviews, the scores can be anywhere in the range 0-100,
rather than purely binary, but we'll see that it doesn't affect the math.
We can follow the classic example of determining the posterior distribution for a Beta-Bernoulli model,
in which we assume that the review scores for an album follow a Bernoulli distribution.

Let's describe the population distribution of scores using the
[Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution),
using the population mean and variance to determine the appropriate $$\alpha_0$$ and $$\beta_0$$ parameters
(see the "mean and variance" section of the wiki page).
We can use this to represent our prior belief for the parameters of the model.
Importantly, the Beta distribution is the conjugate prior for the Bernoulli distribution,
meaning the posterior probability distribution is itself a Beta distribution.
The posterior for an album $$i$$ with $$n_i$$ reviews and an average rating of $$R_i$$
is a Beta distibution with the parameters

$$\alpha = s_i + \alpha_0$$
$$\beta = n_i - s_i + \beta_0$$

where $$s_i$$ is the number of sucessful trials (coins landing on heads),
which in our case is the average review rating times the number of reviews: $$s_i = R_i n_i$$.

The mean of a Beta distribution with parameters $$\alpha$$ and $$\beta$$ is given by

$$\mu = \frac{\alpha}{\alpha + \beta}$$

Thus we can define as our weighted-average score in terms of the sample and prior parameters:

$$\mu = \frac{R_i n_i + \alpha_0}{s_i + \alpha_0 + n_i - s_i + \beta_0}$$


$$\mu = \frac{R_i n_i + \alpha_0}{n_i + \alpha_0 + \beta_0}$$

Doing the same thing for the prior mean $$\mu_0$$ in terms of $$\alpha_0$$ and $$\beta_0$$, we get

$$\mu = \frac{R_i n_i + \mu_0 (\alpha_0 + \beta_0)}{n_i + (\alpha_0 + \beta_0)}$$

Look familiar? <span class="strong-text">The prior mean is the same as the $$C$$ parameter in the IMDb average</span>,
so the only difference between this formula and the other is that
<span class="strong-text">we now have a clear definition for the $$m$$ parameter: it's the sum of the Beta priors,
which are directly derived from the population mean and variance!</span>

$$\alpha_0 + \beta_0 = \frac{1}{\sigma_0^2} \left(\mu_0 - \mu_0^2 - \sigma_0^2\right)$$

To be fair it's not obvious how this parameter behaves,
but generally speaking a more narrow distribution of reviews in the broad population would
yield a smaller variance $$\sigma_0^2$$, which has a similar effect to using a larger $$m$$ in the IMDb method.
This is the expected behavior; a small prior variance represents a more confident prior belief about the mean.
<span class="strong-text">The key is that now we are allowing the prior variance to be based on the actual variance of the data,
rather than simply picking a number.</span>

Implementing this is straightforward: we just have to compute the prior parameters from the full dataset,
then for each album use its average score and number of reviews to compute the parameters of the posterior distribution.
From the parameters we then compute the posterior mean and call that our weighted average.
Below are some examples of what the posteriors look like for some extreme examples
where clearly the weighted average favors/punishes large sample sizes.









#### Examples

In these albums we can see the posterior probability distribution of each album compared to the prior,
which is simply the population un-weighted average score distribution fit to a Beta distribution.
The weighted average score is the mean of this posterior,
computed independently for each album.
Metallica's Ride the Lightning has a raw average (sample mean) of 94%,
which is less than the 100% of Nightmare's Cosmovision,
but the posterior mean of Ride the Lightning (91%) outranks that of Cosmovision (81%)
due to its much larger sample size.
Likewise, on the other side of the population mean,
St. Anger's 44% sample mean is better than the one 0% review of Boris' Warpath,
but the weighted average puts St. Anger at 49%, worse than Warpath's 67%.








#### Apply weighted score to dataset

















#### Weighted score distribution






    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_55_0.png)

    


Looking at the histograms of the weighted average scores and raw average scores,
we can see that <span class="strong-text">the weighted scores push most samples towards the population mean</span>.
This is stronger when looking at albums since the sample sizes are generally smaller.
In the case of genres and countries, the effect is quite weak and probably
won't affect rankings too heavily.

#### Best and worst albums
















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
      <td>Entombed</td>
      <td>Left Hand Path</td>
      <td>1990</td>
      <td>Death Metal/Death 'n' Roll</td>
      <td>25</td>
      <td>97.76</td>
      <td>93.81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>King Diamond</td>
      <td>Abigail</td>
      <td>1987</td>
      <td>Heavy Metal</td>
      <td>18</td>
      <td>97.22</td>
      <td>92.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Katatonia</td>
      <td>Dance of December Souls</td>
      <td>1993</td>
      <td>Doom/Death Metal (early); Gothic/Alternative/P...</td>
      <td>22</td>
      <td>95.45</td>
      <td>91.59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bolt Thrower</td>
      <td>Realm of Chaos: Slaves to Darkness</td>
      <td>1989</td>
      <td>Death Metal</td>
      <td>24</td>
      <td>95.08</td>
      <td>91.55</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Symphony X</td>
      <td>The Divine Wings of Tragedy</td>
      <td>1996</td>
      <td>Progressive Power Metal</td>
      <td>17</td>
      <td>96.53</td>
      <td>91.54</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Morbid Saint</td>
      <td>Spectrum of Death</td>
      <td>1990</td>
      <td>Thrash Metal</td>
      <td>23</td>
      <td>95.00</td>
      <td>91.36</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Atheist</td>
      <td>Unquestionable Presence</td>
      <td>1991</td>
      <td>Progressive Death/Thrash Metal with Jazz influ...</td>
      <td>21</td>
      <td>95.33</td>
      <td>91.35</td>
    </tr>
    <tr>
      <th>8</th>
      <td>W.A.S.P.</td>
      <td>The Crimson Idol</td>
      <td>1992</td>
      <td>Heavy Metal/Hard Rock</td>
      <td>16</td>
      <td>96.38</td>
      <td>91.20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Suffocation</td>
      <td>Effigy of the Forgotten</td>
      <td>1991</td>
      <td>Brutal/Technical Death Metal</td>
      <td>23</td>
      <td>94.78</td>
      <td>91.19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Primordial</td>
      <td>To the Nameless Dead</td>
      <td>2007</td>
      <td>Celtic Folk/Black Metal</td>
      <td>13</td>
      <td>97.54</td>
      <td>91.18</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Bathory</td>
      <td>Under the Sign of the Black Mark</td>
      <td>1987</td>
      <td>Black/Viking Metal, Thrash Metal</td>
      <td>18</td>
      <td>95.72</td>
      <td>91.14</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Metallica</td>
      <td>Ride the Lightning</td>
      <td>1984</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>35</td>
      <td>93.37</td>
      <td>91.04</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Ahab</td>
      <td>The Call of the Wretched Sea</td>
      <td>2006</td>
      <td>Funeral Doom Metal</td>
      <td>20</td>
      <td>94.95</td>
      <td>90.90</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Rush</td>
      <td>Moving Pictures</td>
      <td>1981</td>
      <td>Progressive Rock</td>
      <td>15</td>
      <td>96.20</td>
      <td>90.83</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Kreator</td>
      <td>Coma of Souls</td>
      <td>1990</td>
      <td>Thrash Metal</td>
      <td>21</td>
      <td>94.62</td>
      <td>90.80</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Riot V</td>
      <td>Thundersteel</td>
      <td>1988</td>
      <td>Heavy/Power/Speed Metal</td>
      <td>15</td>
      <td>95.87</td>
      <td>90.60</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Death</td>
      <td>Symbolic</td>
      <td>1995</td>
      <td>Death Metal (early); Progressive Death Metal (...</td>
      <td>31</td>
      <td>93.13</td>
      <td>90.58</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Acid Bath</td>
      <td>When the Kite String Pops</td>
      <td>1994</td>
      <td>Sludge/Doom Metal</td>
      <td>17</td>
      <td>95.18</td>
      <td>90.55</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Type O Negative</td>
      <td>October Rust</td>
      <td>1996</td>
      <td>Gothic/Doom Metal</td>
      <td>16</td>
      <td>95.44</td>
      <td>90.52</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Terrorizer</td>
      <td>World Downfall</td>
      <td>1989</td>
      <td>Death Metal/Grindcore</td>
      <td>14</td>
      <td>96.00</td>
      <td>90.43</td>
    </tr>
  </tbody>
</table>
</div>




<span class="strong-text">The best album by weighted-average score is the 1990 death metal record
[Left Hand Path](https://en.wikipedia.org/wiki/Left_Hand_Path_(album)) by Entombed</span>.
Most of the top-20 albums hail from the 80s and 90s, a testament to their reputations as heavy metal classics.
Primordial's [To the Nameless Dead](https://en.wikipedia.org/wiki/To_the_Nameless_Dead) and
Ahab's [The Call of the Wretched Sea](https://en.wikipedia.org/wiki/The_Call_of_the_Wretched_Sea)
are the only post-2000 albums that make the top twenty.








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
      <td>Thrash or Die</td>
      <td>Poser Holocaust</td>
      <td>2011</td>
      <td>Thrash Metal</td>
      <td>14</td>
      <td>10.71</td>
      <td>31.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hellyeah</td>
      <td>Hellyeah</td>
      <td>2007</td>
      <td>Groove Metal</td>
      <td>13</td>
      <td>10.69</td>
      <td>32.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Waking the Cadaver</td>
      <td>Perverse Recollections of a Necromangler</td>
      <td>2007</td>
      <td>Slam/Brutal Death Metal/Deathcore (early); Dea...</td>
      <td>33</td>
      <td>25.45</td>
      <td>33.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Six Feet Under</td>
      <td>Graveyard Classics 2</td>
      <td>2004</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>10</td>
      <td>7.50</td>
      <td>34.12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Massacre</td>
      <td>Promise</td>
      <td>1996</td>
      <td>Death Metal</td>
      <td>9</td>
      <td>8.11</td>
      <td>36.24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Grieving Age</td>
      <td>Merely the Fleshless We and the Awed Obsequy</td>
      <td>2013</td>
      <td>Doom/Death Metal</td>
      <td>6</td>
      <td>0.83</td>
      <td>39.60</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Six Feet Under</td>
      <td>Nightmares of the Decomposed</td>
      <td>2020</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>11</td>
      <td>19.27</td>
      <td>40.13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Morbid Angel</td>
      <td>Illud Divinum Insanus</td>
      <td>2011</td>
      <td>Death Metal</td>
      <td>30</td>
      <td>35.43</td>
      <td>42.58</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cryptopsy</td>
      <td>The Unspoken King</td>
      <td>2008</td>
      <td>Brutal/Technical Death Metal</td>
      <td>18</td>
      <td>31.00</td>
      <td>42.83</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Skinlab</td>
      <td>Revolting Room</td>
      <td>2002</td>
      <td>Groove Metal (early); Nu-Metal (later)</td>
      <td>6</td>
      <td>8.00</td>
      <td>43.15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Machine Head</td>
      <td>Catharsis</td>
      <td>2018</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>8</td>
      <td>17.75</td>
      <td>43.70</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Dark Moor</td>
      <td>Project X</td>
      <td>2015</td>
      <td>Power Metal</td>
      <td>6</td>
      <td>9.50</td>
      <td>43.90</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Queensrÿche</td>
      <td>American Soldier</td>
      <td>2009</td>
      <td>Heavy/Power/Progressive Metal (early/later); H...</td>
      <td>12</td>
      <td>27.17</td>
      <td>44.21</td>
    </tr>
    <tr>
      <th>14</th>
      <td>In Flames</td>
      <td>Battles</td>
      <td>2016</td>
      <td>Melodic Death Metal (early); Melodic Groove Me...</td>
      <td>10</td>
      <td>24.90</td>
      <td>44.92</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Celtic Frost</td>
      <td>Cold Lake</td>
      <td>1988</td>
      <td>Thrash/Death/Black Metal (early); Gothic/Doom ...</td>
      <td>16</td>
      <td>33.12</td>
      <td>45.44</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Aryan Terrorism</td>
      <td>War</td>
      <td>2000</td>
      <td>Black Metal</td>
      <td>6</td>
      <td>13.33</td>
      <td>45.80</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Machine Head</td>
      <td>Supercharger</td>
      <td>2001</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>12</td>
      <td>29.92</td>
      <td>46.03</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Six Feet Under</td>
      <td>Graveyard Classics IV: 666 - The Number of the...</td>
      <td>2016</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>5</td>
      <td>8.60</td>
      <td>46.59</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Bathory</td>
      <td>Octagon</td>
      <td>1995</td>
      <td>Black/Viking Metal, Thrash Metal</td>
      <td>11</td>
      <td>29.91</td>
      <td>46.97</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Huntress</td>
      <td>Spell Eater</td>
      <td>2012</td>
      <td>Heavy Metal</td>
      <td>11</td>
      <td>30.27</td>
      <td>47.20</td>
    </tr>
  </tbody>
</table>
</div>




<span class="strong-text">At the other end of the rankings,
[Poser Holocaust](https://www.metal-archives.com/albums/Thrash_or_Die/Poser_Holocaust/307982)
by Thrash or Die earns the worst weighted score</span>,
but Waking the Cadaver's debut album
[Perverse Recollections of a Necromangler](https://en.wikipedia.org/wiki/Perverse_Recollections_of_a_Necromangler)
deserves a mention for having a whopping 33 reviews:
it's amusing that more people have reviewed this record than have reviewed Ride the Lightning.
Maybe this just reflects on the sort of opinions album reviewers enjoy sharing...

<span class="strong-text">Many of the bottom twenty albums actually come from fairly well-respected artists
whose fans simply expected better.</span>
It almost seems like a rite of passage to disappoint your fans after a decade or two of consistent performance.
Next we'll look at which bands have done the best or worst over the full course of their careers.

#### Best and worst bands












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
      <td>Evoken</td>
      <td>Funeral Doom/Death Metal</td>
      <td>32</td>
      <td>94.94</td>
      <td>91.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Acid Bath</td>
      <td>Sludge/Doom Metal</td>
      <td>27</td>
      <td>95.07</td>
      <td>91.44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Morbid Saint</td>
      <td>Thrash Metal</td>
      <td>27</td>
      <td>93.70</td>
      <td>90.35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Moonsorrow</td>
      <td>Folk/Pagan/Black Metal</td>
      <td>74</td>
      <td>91.24</td>
      <td>90.05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Intestine Baalism</td>
      <td>Melodic Death Metal</td>
      <td>19</td>
      <td>94.63</td>
      <td>89.99</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Type O Negative</td>
      <td>Gothic/Doom Metal</td>
      <td>89</td>
      <td>90.93</td>
      <td>89.95</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Demilich</td>
      <td>Technical/Avant-garde Death Metal</td>
      <td>26</td>
      <td>93.04</td>
      <td>89.72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Satan</td>
      <td>NWOBHM, Heavy Metal</td>
      <td>28</td>
      <td>92.39</td>
      <td>89.39</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Forefather</td>
      <td>Black/Viking Metal</td>
      <td>17</td>
      <td>94.29</td>
      <td>89.36</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Angra</td>
      <td>Power/Progressive Metal</td>
      <td>83</td>
      <td>90.25</td>
      <td>89.25</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Solitude Aeturnus</td>
      <td>Epic Doom Metal</td>
      <td>28</td>
      <td>92.11</td>
      <td>89.16</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Gorement</td>
      <td>Death Metal</td>
      <td>10</td>
      <td>97.40</td>
      <td>89.15</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Repulsion</td>
      <td>Grindcore/Death Metal</td>
      <td>17</td>
      <td>94.00</td>
      <td>89.15</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Esoteric</td>
      <td>Funeral Doom/Death Metal</td>
      <td>34</td>
      <td>91.50</td>
      <td>89.09</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Primordial</td>
      <td>Celtic Folk/Black Metal</td>
      <td>71</td>
      <td>90.21</td>
      <td>89.06</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kyuss</td>
      <td>Stoner Rock/Metal</td>
      <td>38</td>
      <td>90.74</td>
      <td>88.66</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sabbat</td>
      <td>Black/Thrash Metal</td>
      <td>21</td>
      <td>92.29</td>
      <td>88.55</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Man Must Die</td>
      <td>Technical Death Metal</td>
      <td>14</td>
      <td>94.00</td>
      <td>88.45</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Wardruna</td>
      <td>Folk/Ambient</td>
      <td>12</td>
      <td>94.83</td>
      <td>88.39</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Impaled</td>
      <td>Death Metal</td>
      <td>16</td>
      <td>93.00</td>
      <td>88.24</td>
    </tr>
  </tbody>
</table>
</div>




To accumulate a high weighted-average score, a band must put out many albums,
each garnering many positive reviews.
Many of the most popular bands in metal do poorly in this sense,
as they set very high standards for themselves and almost always have one or two albums
that far underperform compared to the rest of their discography.
This leaves us with some surprise appearances at the top of the weighted-average rankings:
<span class="strong-text">[Evoken](https://en.wikipedia.org/wiki/Evoken)
takes the crown for highest weighted score.
Over a six-album discography, they boast a minimum album rating of 91%.</span>
There's quite a lot of variety among the top few bands,
with many different genres and nationalities being represented.








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
      <td>Thrash or Die</td>
      <td>Thrash Metal</td>
      <td>16</td>
      <td>14.94</td>
      <td>33.52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Skinlab</td>
      <td>Groove Metal (early); Nu-Metal (later)</td>
      <td>17</td>
      <td>17.35</td>
      <td>34.47</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hellyeah</td>
      <td>Groove Metal</td>
      <td>22</td>
      <td>23.73</td>
      <td>36.36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waking the Cadaver</td>
      <td>Slam/Brutal Death Metal/Deathcore (early); Dea...</td>
      <td>46</td>
      <td>34.46</td>
      <td>39.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Legion of Thor</td>
      <td>Hardcore Punk/Metalcore (early); Death Metal/D...</td>
      <td>7</td>
      <td>3.86</td>
      <td>40.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Winds of Plague</td>
      <td>Symphonic Deathcore</td>
      <td>33</td>
      <td>37.24</td>
      <td>44.07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Animae Capronii</td>
      <td>Black Metal</td>
      <td>10</td>
      <td>22.60</td>
      <td>44.70</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Grieving Age</td>
      <td>Doom/Death Metal</td>
      <td>8</td>
      <td>18.75</td>
      <td>45.61</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Car Door Dick Smash</td>
      <td>Thrash Metal/Grindcore</td>
      <td>5</td>
      <td>4.40</td>
      <td>46.35</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Aryan Terrorism</td>
      <td>Black Metal</td>
      <td>6</td>
      <td>13.33</td>
      <td>47.26</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Huntress</td>
      <td>Heavy Metal</td>
      <td>14</td>
      <td>33.43</td>
      <td>47.74</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Six Feet Under</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>134</td>
      <td>48.49</td>
      <td>49.88</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Damageplan</td>
      <td>Groove Metal</td>
      <td>11</td>
      <td>35.73</td>
      <td>51.56</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Mulletcorpse</td>
      <td>Brutal Death Metal/Grindcore</td>
      <td>9</td>
      <td>32.78</td>
      <td>51.89</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Enemy of the Sun</td>
      <td>Groove/Alternative Metal/Metalcore</td>
      <td>6</td>
      <td>24.17</td>
      <td>52.33</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Groza</td>
      <td>Black Metal</td>
      <td>5</td>
      <td>18.60</td>
      <td>52.35</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Propagating the Abomination</td>
      <td>Brutal Death Metal</td>
      <td>5</td>
      <td>21.20</td>
      <td>53.45</td>
    </tr>
    <tr>
      <th>18</th>
      <td>The Great Kat</td>
      <td>Speed/Thrash Metal, Shred</td>
      <td>14</td>
      <td>42.29</td>
      <td>53.69</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Procer Veneficus</td>
      <td>Black Metal/Dark Ambient/Experimental Acoustic</td>
      <td>12</td>
      <td>42.00</td>
      <td>54.72</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Jewicide</td>
      <td>Raw Black Metal</td>
      <td>4</td>
      <td>17.25</td>
      <td>54.97</td>
    </tr>
  </tbody>
</table>
</div>




<span class="strong-text">Yet again [Thrash or Die](https://en.wikipedia.org/wiki/Thrash_or_Die)
take up their position at the bottom of this ranking.</span>
It's mostly due to that horrid debut album of theirs;
it makes up 14 of their 16 overall reviews.
Their 2015 follow-up album at least managed to get a couple of reviews over 40%...

#### Best and worst genres








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
      <th>review_num</th>
      <th>review_avg</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>eastern</td>
      <td>142</td>
      <td>85.31</td>
      <td>84.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>middle</td>
      <td>125</td>
      <td>85.28</td>
      <td>83.87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>grunge</td>
      <td>147</td>
      <td>84.42</td>
      <td>83.32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>post-metal</td>
      <td>564</td>
      <td>82.59</td>
      <td>82.34</td>
    </tr>
    <tr>
      <th>5</th>
      <td>epic</td>
      <td>1358</td>
      <td>82.42</td>
      <td>82.32</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dungeon</td>
      <td>55</td>
      <td>84.51</td>
      <td>82.09</td>
    </tr>
    <tr>
      <th>7</th>
      <td>synth</td>
      <td>55</td>
      <td>84.51</td>
      <td>82.09</td>
    </tr>
    <tr>
      <th>8</th>
      <td>funeral</td>
      <td>734</td>
      <td>82.23</td>
      <td>82.05</td>
    </tr>
    <tr>
      <th>9</th>
      <td>fusion</td>
      <td>133</td>
      <td>82.99</td>
      <td>82.01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>jazz</td>
      <td>139</td>
      <td>82.88</td>
      <td>81.95</td>
    </tr>
  </tbody>
</table>
</div>




Interestingly, <span class="strong-text">Middle-Eastern metal is rated very highly</span>
(it gets split up into "middle" and "eastern" because of the genre tag parsing).
This chart has a lot of unexpected appearances,
and I'm not sure there's much to read into here.
When grouping by genres there are far more reviews per sample,
<span class="strong-text">so the weighted averaging doesn't stray far from the raw average.</span>








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
      <th>review_num</th>
      <th>review_avg</th>
      <th>review_weighted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>pop</td>
      <td>83</td>
      <td>62.80</td>
      <td>65.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nu-metal</td>
      <td>534</td>
      <td>65.94</td>
      <td>66.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>deathcore</td>
      <td>1007</td>
      <td>66.88</td>
      <td>67.11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>acoustic</td>
      <td>83</td>
      <td>65.34</td>
      <td>67.86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tribal</td>
      <td>30</td>
      <td>61.93</td>
      <td>68.45</td>
    </tr>
    <tr>
      <th>6</th>
      <td>metalcore</td>
      <td>1892</td>
      <td>68.82</td>
      <td>68.91</td>
    </tr>
    <tr>
      <th>7</th>
      <td>indie</td>
      <td>18</td>
      <td>61.44</td>
      <td>70.09</td>
    </tr>
    <tr>
      <th>8</th>
      <td>groove</td>
      <td>3703</td>
      <td>70.12</td>
      <td>70.16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cappella</td>
      <td>16</td>
      <td>60.94</td>
      <td>70.32</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cybergrind</td>
      <td>12</td>
      <td>58.25</td>
      <td>70.47</td>
    </tr>
  </tbody>
</table>
</div>




At the bottom, no-one can be surprised to see <span class="strong-text">pop music having lowest weighted reviews</span>.
Metal purists are known for the dislike of most deathcore, metalcore, and nu-metal, and it shows here as well.

#### Best and worse countries

Similar to genres, the large sample sizes of most countries seems to
weaken the differences between the raw and weighted averages.
Nevertheless the weighting brings up some interesting contenders
for the best and worst countries when it comes to producing metal records.












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
      <td>Ireland</td>
      <td>280</td>
      <td>83.01</td>
      <td>82.49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>140</td>
      <td>83.47</td>
      <td>82.44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Denmark</td>
      <td>962</td>
      <td>82.49</td>
      <td>82.35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>28</td>
      <td>86.64</td>
      <td>81.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tunisia</td>
      <td>29</td>
      <td>85.76</td>
      <td>81.60</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Singapore</td>
      <td>100</td>
      <td>82.37</td>
      <td>81.26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Japan</td>
      <td>1102</td>
      <td>81.28</td>
      <td>81.18</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jordan</td>
      <td>33</td>
      <td>84.12</td>
      <td>80.99</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Faroe Islands</td>
      <td>71</td>
      <td>82.24</td>
      <td>80.84</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Georgia</td>
      <td>16</td>
      <td>87.00</td>
      <td>80.82</td>
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
      <td>Saudi Arabia</td>
      <td>21</td>
      <td>58.29</td>
      <td>69.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ecuador</td>
      <td>15</td>
      <td>66.67</td>
      <td>73.64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Vietnam</td>
      <td>9</td>
      <td>63.33</td>
      <td>73.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Algeria</td>
      <td>10</td>
      <td>65.60</td>
      <td>74.27</td>
    </tr>
    <tr>
      <th>5</th>
      <td>North Macedonia</td>
      <td>8</td>
      <td>65.00</td>
      <td>74.61</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Iraq</td>
      <td>5</td>
      <td>60.40</td>
      <td>74.79</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Unknown</td>
      <td>50</td>
      <td>73.70</td>
      <td>75.01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Syria</td>
      <td>12</td>
      <td>70.00</td>
      <td>75.15</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Botswana</td>
      <td>14</td>
      <td>71.43</td>
      <td>75.38</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Nepal</td>
      <td>17</td>
      <td>72.41</td>
      <td>75.48</td>
    </tr>
  </tbody>
</table>
</div>




## Geographic distribution of albums and genres

Now we'll look at some geographic numbers describing the popularity of metal around the world.

#### Countries of origin






    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_83_0.png)

    


* <span class="strong-text">The U.S. clearly dominates in album production
(or at least the production of albums that are cataloged on MA;
although I'm quite confident the selection bias is not too strong considering how active the MA community is).</span>

* <span class="strong-text">Most of the top countries in the scene are European.</span>

#### U.S. states of origin






    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_86_0.png)

    


* Within the U.S. itself, <span class="strong-text">California produces the most metal albums</span>,
followed by a few other high-population states.
* <span class="strong-text">Normalized by population, Oregon and Washington produce the most;</span>
Pacific Northwest weather is perfect for heavy metal after all.

#### Top countries in each genre






    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_89_0.png)

    


* <span class="strong-text">The U.S. is still the top producer of albums in most genres.</span>
* <span class="strong-text">Italy is the only country to overtake the U.S. in any genre, outproducing the Americans in
symphonic metal albums.</span>
* When it comes to the "melodic" genre tag,
which most often refers to melodic death metal in particular,
the Swedes and Finns contribute a large portion.
* Germany produces a notably large portion of power metal albums.

#### Top genres in each country






    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_92_0.png)

    


For most countries, death and black metal are the most common genres.
The Ukranians seem to hold the strongest preference for a single genre,
with nearly a third of Ukranian albums coming from bands identified as black metal.
The largest stake power metal takes is in Germany.





## Decline of top-rated bands






    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_100_0.png)

As mentioned before, popular artists who have been around for a while often struggle to meet fan expectations as years pass.
<span class="strong-text">Here we see that eight of the ten bands that produced a top-ten album (based on weighted score)
have seen overall declines in reviewer perception after a decade or two from their debuts.</span>


## Global album trends

Now for a more comprehensive look at the history of metal, as told by the MA community.










    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_104_0.png)

    


* <span class="strong-text">The number of metal albums released per year grew up until the late 2000's,
but has since plateaued.</span>
* The annual average of weighted-average scores (fourth plot)
stays <span class="strong-text">quite high throughout the 70s and 80s</span>,
about the same time that many of the all-time top bands
were releasing their most iconic albums.
* <span class="strong-text">As the metal scene saturated, quality dropped</span>,
with scores hitting a low point coinciding with the peak in number of albums.
* Albums from the past decade, however, have performed better,
with <span class="strong-text">the average just about returning to pre-90s levels last year</span>,
giving fans reason to be optimistic about the direction that heavy metal is going.
* Even the slump in album production in 2020-21 has not manifested in a drop in scores,
indicating that <span class="strong-text">the COVID-19 pandemic has not adversely affected the quality of metal albums!</span>
* The number of reviews per album is still in decline,
but at least for albums of the past few years this is likely just due to recency bias.
We saw at the beginning that the rate of reviews submitted to MA hasn't actually dropped in recent years.





#### Yearly album output and review scores by genre






    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_108_0.png)

    


Although most genres contributed to the rise of metal in the 2000s,
<span class="strong-text">black and death metal dominated the upward trend</span>.
Before then, it was heavy metal and thrash metal that ruled supreme,
spearheading the 80s upswing that brought heavy metal into the public spotlight.

<span class="strong-text">Early doom metal (led by Black Sabbath) is very highly rated</span>,
and even more recent doom metal albums continue to outperform material from other genres rather consistently.
Early black metal albums likewise had a high-scoring start, but have mostly converged towards
the overal average since then.
Death metal did well in the early nineties, followed by a long brutal decline throughout the 2000s.
Progressive rock/metal peaked twice, once in the late-80s wave (perhaps thanks to Rush),
and again the early 2000s as progressive metal started to take shape.
<span class="strong-text">Recently progressive rock/metal been the most highly-rated genre after doom metal.<span>

## Geographic trends






    
![png](/assets/images/heavy-metal-lyrics/reviews/reviews1_111_0.png)

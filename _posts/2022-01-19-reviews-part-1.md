---
layout: post
date: 2022-01-19
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


#### Most-reviewed albums

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
      <td>Metallica</td>
      <td>Master of Puppets</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>1986</td>
      <td>77.20</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Slayer</td>
      <td>Reign in Blood</td>
      <td>Thrash Metal</td>
      <td>1986</td>
      <td>86.42</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wintersun</td>
      <td>Time I</td>
      <td>Symphonic Melodic Death Metal</td>
      <td>2012</td>
      <td>69.24</td>
      <td>37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Metallica</td>
      <td>Hardwired... to Self-Destruct</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>2016</td>
      <td>62.64</td>
      <td>36</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Black Sabbath</td>
      <td>13</td>
      <td>Heavy/Doom Metal</td>
      <td>2013</td>
      <td>66.58</td>
      <td>36</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Waking the Cadaver</td>
      <td>Perverse Recollections of a Necromangler</td>
      <td>Slam/Brutal Death Metal/Deathcore (early), Dea...</td>
      <td>2007</td>
      <td>24.23</td>
      <td>35</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Metallica</td>
      <td>Kill 'Em All</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>1983</td>
      <td>87.63</td>
      <td>35</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Megadeth</td>
      <td>Countdown to Extinction</td>
      <td>Speed/Thrash Metal (early/later); Heavy Metal/...</td>
      <td>1992</td>
      <td>77.53</td>
      <td>34</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Pantera</td>
      <td>Vulgar Display of Power</td>
      <td>Glam/Heavy Metal (early), Groove Metal (later)</td>
      <td>1992</td>
      <td>61.39</td>
      <td>33</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Iron Maiden</td>
      <td>The Number of the Beast</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>1982</td>
      <td>85.06</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>

#### Bands with most reviews

<div style="width:80%">
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
      <td>79.80</td>
      <td>394</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Black Sabbath</td>
      <td>Heavy/Doom Metal</td>
      <td>82.17</td>
      <td>351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Megadeth</td>
      <td>Speed/Thrash Metal (early/later); Heavy Metal/...</td>
      <td>74.42</td>
      <td>344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Metallica</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>72.42</td>
      <td>313</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Judas Priest</td>
      <td>Heavy Metal</td>
      <td>79.80</td>
      <td>307</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Darkthrone</td>
      <td>Death Metal (early); Black Metal (mid); Black/...</td>
      <td>78.95</td>
      <td>274</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Slayer</td>
      <td>Thrash Metal</td>
      <td>73.99</td>
      <td>255</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Opeth</td>
      <td>Progressive Death Metal, Progressive Rock</td>
      <td>78.23</td>
      <td>240</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dimmu Borgir</td>
      <td>Symphonic Black Metal</td>
      <td>66.33</td>
      <td>218</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Overkill</td>
      <td>Thrash Metal; Thrash/Groove Metal</td>
      <td>79.77</td>
      <td>216</td>
    </tr>
  </tbody>
</table>
</div>

#### Genres with most reviews

<div style="width:40%">
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
      <td>79.06</td>
      <td>15231</td>
    </tr>
    <tr>
      <th>2</th>
      <td>black</td>
      <td>80.32</td>
      <td>11619</td>
    </tr>
    <tr>
      <th>3</th>
      <td>thrash</td>
      <td>77.95</td>
      <td>8754</td>
    </tr>
    <tr>
      <th>4</th>
      <td>heavy</td>
      <td>79.07</td>
      <td>8334</td>
    </tr>
    <tr>
      <th>5</th>
      <td>power</td>
      <td>78.76</td>
      <td>7645</td>
    </tr>
    <tr>
      <th>6</th>
      <td>progressive</td>
      <td>81.71</td>
      <td>5637</td>
    </tr>
    <tr>
      <th>7</th>
      <td>melodic</td>
      <td>77.42</td>
      <td>5052</td>
    </tr>
    <tr>
      <th>8</th>
      <td>rock</td>
      <td>77.69</td>
      <td>5012</td>
    </tr>
    <tr>
      <th>9</th>
      <td>doom</td>
      <td>83.26</td>
      <td>4710</td>
    </tr>
    <tr>
      <th>10</th>
      <td>speed</td>
      <td>79.77</td>
      <td>3563</td>
    </tr>
  </tbody>
</table>
</div>

#### Countries with most reviews

<div style="width:40%">
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
      <td>77.51</td>
      <td>16069</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sweden</td>
      <td>79.33</td>
      <td>5505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United Kingdom</td>
      <td>80.30</td>
      <td>4804</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>79.82</td>
      <td>4445</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Norway</td>
      <td>79.45</td>
      <td>3167</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Finland</td>
      <td>78.77</td>
      <td>2847</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Canada</td>
      <td>78.85</td>
      <td>1884</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Italy</td>
      <td>76.50</td>
      <td>988</td>
    </tr>
    <tr>
      <th>9</th>
      <td>France</td>
      <td>79.52</td>
      <td>985</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Netherlands</td>
      <td>80.06</td>
      <td>928</td>
    </tr>
  </tbody>
</table>
</div>

## Weighted-average album score

Comparing albums by simply looking at average review ratings fails to consider each album's popularity (or infamy).
This is important when the number of reviews per album vary dramatically.
Just like looking at product reviews, we naturally assign more weight to album review scores
that are averaged from the experiences of many people.

As you can see below, there are plenty of albums on MA with only a single, 100% review.
It doesn't make much sense to say these albums are all better than the most popular albums.
Likewise, there are plenty of albums with only a single 0% review.
The same can be seen when splitting the data by band.
I could apply a minimum number of reviews required to consider an album's review score legitimate,
but this shrinks down the dataset considerably and still weighs albums near the cutoff number and
near the maximum equally.

Instead, I will use a weighted-average score that treats individual reviews for an album as "evidence"
that the album ought to deviate from the population mean (of 79%).
Ideally, this method would distinguish good albums based on them having many positive reviews, not just a handful.
Likewise, it should help us reveal which albums draw a consensus of disdain from the MA community.

<details>
<summary>Show albums with highest average reviews</summary>
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
<summary>Show albums with lowest average reviews</summary>
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
<br>

<details>
<summary>Show bands with highest average reviews</summary>
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
      <td>Gorement</td>
      <td>100.00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ocean Machine</td>
      <td>96.10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Deathwish</td>
      <td>95.67</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Acid Bath</td>
      <td>95.29</td>
      <td>24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tyrant of Death</td>
      <td>95.14</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>
</details>
<br>

<details>
<summary>Show bands with lowest average reviews</summary>
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
      <td>Car Door Dick Smash</td>
      <td>6.67</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chainsaw Penis</td>
      <td>15.00</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Skinlab</td>
      <td>19.33</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thrash or Die</td>
      <td>19.87</td>
      <td>15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Animae Capronii</td>
      <td>22.60</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>
</details>
<br>

#### Rating albums using Bayesian inference: the IMDB weighted average rating

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

Look familiar? The prior mean is the same as the $$C$$ parameter in the IMDb average,
so the only difference between this formula and the other is that we now have a clear definition
for the $$m$$ parameter now: it's the sum of the Beta priors,
which are directly derived from the population mean and variance!

$$\alpha_0 + \beta_0 = \frac{1}{\sigma_0^2} \left(\mu_0 - \mu_0^2 - \sigma_0^2\right)$$

To be fair it's not obvious how this parameter behaves,
but generally speaking a more narrow distribution of reviews in the broad population would
yield a smaller variance $$\sigma_0^2$$, which has a similar effect to using a larger $$m$$ in the IMDb method.
This is the expected behavior; a small prior variance represents a more confident prior belief about the mean.
The key is that now we are allowing the prior variance to be based on the actual variance of the data,
rather than simply picking a number.

Implementing this is straightforward: we just have to compute the prior parameters from the full dataset,
then for each album use its average score and number of reviews to compute the parameters of the posterior distribution.
From the parameters we then compute the posterior mean and call that our weighted average.
Below are some examples of what the posteriors look like for some extreme examples
where clearly the weighted average favors/punishes large sample sizes.


<details>
<summary>Show code</summary>
{% highlight python %}
def beta_prior(x):
    mu = x.mean()
    var = x.std() ** 2
    a = mu * (mu * (1 - mu) / var - 1)
    b = (1 - mu) * (mu * (1 - mu) / var - 1)
    prior = beta(a, b)
    return prior


def beta_posteriors(data, prior):
    n = data['review_num']
    s = data['review_avg'] / 100 * n
    a_prior, b_prior = prior.args
    a_post = s + a_prior
    b_post = n - s + b_prior
    posts = [beta(a, b) for a, b in zip(a_post, b_post)]
    return posts


def weighted_scores(data, verbose=False):
    prior = beta_prior(data['review_avg'] / 100.)
    if verbose:
        print(f"Prior (alpha, beta) = ({prior.args[0]:.2f}, {prior.args[0]:.2f})")
        print(f"Prior weight = {sum(prior.args):.2f}")
    posteriors = beta_posteriors(data, prior)
    post_means = 100 * pd.Series([post.mean() for post in posteriors], index=data.index)
    return post_means
{% endhighlight %}
</details>


#### Examples

In these albums we can see the posterior probability distribution of each album compared to the prior,
which is simply the population un-weighted average score distribution fit to a Beta distribution.
The weighted average score is the mean of this posterior,
computed independently for each album.
Metallica's Ride the Lightning has a raw average (sample mean) of 94%,
which is less than the 100% of Nightmare's Cosmovision,
but the posterior mean of Ride the Lightning (91%) outranks that of Cosmovision (81%)
due to its much larger sample size.
Likewise, on the other side of the population mean, St. Anger's 45% sample mean
is better than the one 0% review of Boris' Warpath,
but the weighted average puts St. Anger at 51% worse than Warpath's 69%.

![png](/assets/images/heavy-metal-lyrics/reviews/weighted_album.png)

The same effect of sample size is clearly seen when comparing bands
with high/low raw/weighted-average scores.

![png](/assets/images/heavy-metal-lyrics/reviews/weighted_band.png)


#### Weighted score distribution

Looking at the histograms of the weighted average scores and raw average scores,
we can see that the weighted scores push most samples towards the population mean.
This is stronger when looking at albums since the sample sizes are generally smaller.
In the case of genres and countries, the effect is quite weak and probably
won't affect rankings too heavily.

![png](/assets/images/heavy-metal-lyrics/reviews/weighted_hist.png)



#### Best and worst albums

The best album by weighted-average score is the 1990 death metal record
[Left Hand Path](https://en.wikipedia.org/wiki/Left_Hand_Path_(album)) by Entombed.
Most of the top-20 albums hail from the 80s and 90s, a testament to their reputations as heavy metal classics.
The effect of weighing high review counts is seen in the placement of Metallica's
[Ride the Lightning](https://en.wikipedia.org/wiki/Ride_the_Lightning),
which rides its way to fourth place thanks to its 30 reviews,
outperforming albums with much higher averages but lower review counts,
such as Primordial's [To the Nameless Dead](https://en.wikipedia.org/wiki/To_the_Nameless_Dead).

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
      <td>16</td>
      <td>98.06</td>
      <td>92.08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Symphony X</td>
      <td>The Divine Wings of Tragedy</td>
      <td>1996</td>
      <td>Progressive Power Metal</td>
      <td>18</td>
      <td>96.33</td>
      <td>91.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Riot V</td>
      <td>Thundersteel</td>
      <td>1988</td>
      <td>Heavy/Power/Speed Metal</td>
      <td>14</td>
      <td>97.36</td>
      <td>91.04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Metallica</td>
      <td>Ride the Lightning</td>
      <td>1984</td>
      <td>Thrash Metal (early); Hard Rock (mid); Heavy/T...</td>
      <td>30</td>
      <td>93.93</td>
      <td>91.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Katatonia</td>
      <td>Dance of December Souls</td>
      <td>1993</td>
      <td>Doom/Death Metal (early), Depressive Rock/Meta...</td>
      <td>19</td>
      <td>95.21</td>
      <td>90.69</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ahab</td>
      <td>The Call of the Wretched Sea</td>
      <td>2006</td>
      <td>Funeral Doom Metal</td>
      <td>20</td>
      <td>94.95</td>
      <td>90.66</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Primordial</td>
      <td>To the Nameless Dead</td>
      <td>2007</td>
      <td>Celtic Folk/Black Metal</td>
      <td>12</td>
      <td>97.75</td>
      <td>90.63</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rush</td>
      <td>Moving Pictures</td>
      <td>1981</td>
      <td>Hard Rock/Heavy Metal (early); Progressive Roc...</td>
      <td>15</td>
      <td>96.20</td>
      <td>90.54</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bathory</td>
      <td>Under the Sign of the Black Mark</td>
      <td>1987</td>
      <td>Black/Viking Metal, Thrash Metal</td>
      <td>17</td>
      <td>95.47</td>
      <td>90.50</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bolt Thrower</td>
      <td>Realm of Chaos: Slaves to Darkness</td>
      <td>1989</td>
      <td>Death Metal</td>
      <td>22</td>
      <td>94.23</td>
      <td>90.41</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Terrorizer</td>
      <td>World Downfall</td>
      <td>1989</td>
      <td>Death Metal/Grindcore</td>
      <td>14</td>
      <td>96.36</td>
      <td>90.38</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Atheist</td>
      <td>Unquestionable Presence</td>
      <td>1991</td>
      <td>Death/Thrash Metal with Jazz and Progressive i...</td>
      <td>18</td>
      <td>95.00</td>
      <td>90.36</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sepultura</td>
      <td>Beneath the Remains</td>
      <td>1989</td>
      <td>Death/Thrash Metal (early), Nu-Metal, Groove/T...</td>
      <td>28</td>
      <td>93.32</td>
      <td>90.34</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Morbid Saint</td>
      <td>Spectrum of Death</td>
      <td>1990</td>
      <td>Thrash Metal</td>
      <td>20</td>
      <td>94.50</td>
      <td>90.33</td>
    </tr>
    <tr>
      <th>15</th>
      <td>W.A.S.P.</td>
      <td>ReIdolized (The Soundtrack to the Crimson Idol)</td>
      <td>2018</td>
      <td>Heavy Metal/Hard Rock</td>
      <td>14</td>
      <td>96.21</td>
      <td>90.28</td>
    </tr>
    <tr>
      <th>16</th>
      <td>W.A.S.P.</td>
      <td>The Crimson Idol</td>
      <td>1992</td>
      <td>Heavy Metal/Hard Rock</td>
      <td>14</td>
      <td>96.21</td>
      <td>90.28</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Death</td>
      <td>Symbolic</td>
      <td>1995</td>
      <td>Death Metal (early), Progressive Death Metal (...</td>
      <td>30</td>
      <td>93.00</td>
      <td>90.24</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Artillery</td>
      <td>By Inheritance</td>
      <td>1990</td>
      <td>Thrash Metal</td>
      <td>20</td>
      <td>94.30</td>
      <td>90.18</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Judas Priest</td>
      <td>Stained Class</td>
      <td>1978</td>
      <td>Heavy Metal</td>
      <td>20</td>
      <td>94.25</td>
      <td>90.15</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Black Sabbath</td>
      <td>Paranoid</td>
      <td>1970</td>
      <td>Heavy/Doom Metal</td>
      <td>30</td>
      <td>92.87</td>
      <td>90.13</td>
    </tr>
  </tbody>
</table>
</div>

At the other end of the rankings, Waking the Cadaver's debut album
[Perverse Recollections of a Necromangler](https://en.wikipedia.org/wiki/Perverse_Recollections_of_a_Necromangler)
earns itself the honor of most disliked album on MA.
It's even clearer here that the number of reviews weighs heavily on the score.
It's also funny to imagine that more people have reviewed this record than have reviewed Ride the Lightning.
Maybe this just reflects on the sort of opinions album reviewers enjoy sharing more...

Metallica's infamous [St. Anger](https://en.wikipedia.org/wiki/St._Anger)
does indeed get thrown to the gutters by the weighted scoring method.
In fact, most of the worst albums were made by highly talented artists whose fans were expecting so much more.
It almost seems like a rite of passage to disappoint your fans after a decade or two of consistency.
Next we'll look at which bands have done the best or worst over the full course of their careers.

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
      <td>33.39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hellyeah</td>
      <td>Hellyeah</td>
      <td>2007</td>
      <td>Groove Metal</td>
      <td>13</td>
      <td>10.69</td>
      <td>34.65</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Six Feet Under</td>
      <td>Graveyard Classics 2</td>
      <td>2004</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>10</td>
      <td>9.30</td>
      <td>38.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thrash or Die</td>
      <td>Poser Holocaust</td>
      <td>2011</td>
      <td>Thrash Metal</td>
      <td>14</td>
      <td>17.86</td>
      <td>38.27</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Massacre</td>
      <td>Promise</td>
      <td>1996</td>
      <td>Death Metal</td>
      <td>7</td>
      <td>7.86</td>
      <td>43.45</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cryptopsy</td>
      <td>The Unspoken King</td>
      <td>2008</td>
      <td>Brutal/Technical Death Metal, Deathcore (2008)</td>
      <td>18</td>
      <td>31.00</td>
      <td>44.45</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Morbid Angel</td>
      <td>Illud Divinum Insanus</td>
      <td>2011</td>
      <td>Death Metal</td>
      <td>28</td>
      <td>35.82</td>
      <td>44.46</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Six Feet Under</td>
      <td>Graveyard Classics IV: The Number of the Priest</td>
      <td>2016</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>6</td>
      <td>7.33</td>
      <td>45.93</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Queensrÿche</td>
      <td>American Soldier</td>
      <td>2009</td>
      <td>Heavy/Power/Progressive Metal</td>
      <td>12</td>
      <td>27.17</td>
      <td>46.26</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dark Moor</td>
      <td>Project X</td>
      <td>2015</td>
      <td>Power Metal</td>
      <td>6</td>
      <td>9.50</td>
      <td>46.92</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Exodus</td>
      <td>Let There Be Blood</td>
      <td>2008</td>
      <td>Thrash Metal</td>
      <td>9</td>
      <td>22.33</td>
      <td>47.12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Celtic Frost</td>
      <td>Cold Lake</td>
      <td>1988</td>
      <td>Thrash/Death/Black Metal (early), Gothic/Doom ...</td>
      <td>14</td>
      <td>31.79</td>
      <td>47.52</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Machine Head</td>
      <td>Supercharger</td>
      <td>2001</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>11</td>
      <td>29.09</td>
      <td>48.49</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Winds of Plague</td>
      <td>Decimate the Weak</td>
      <td>2008</td>
      <td>Symphonic Deathcore</td>
      <td>19</td>
      <td>37.84</td>
      <td>48.91</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Bathory</td>
      <td>Octagon</td>
      <td>1995</td>
      <td>Black/Viking Metal, Thrash Metal</td>
      <td>11</td>
      <td>29.91</td>
      <td>48.99</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Huntress</td>
      <td>Spell Eater</td>
      <td>2012</td>
      <td>Heavy Metal</td>
      <td>11</td>
      <td>30.27</td>
      <td>49.21</td>
    </tr>
    <tr>
      <th>17</th>
      <td>In Flames</td>
      <td>Soundtrack to Your Escape</td>
      <td>2004</td>
      <td>Melodic Death Metal (early), Melodic Groove Me...</td>
      <td>20</td>
      <td>39.10</td>
      <td>49.43</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Lacuna Coil</td>
      <td>Shallow Life</td>
      <td>2009</td>
      <td>Gothic Metal/Rock (early); Alternative Rock (l...</td>
      <td>11</td>
      <td>31.45</td>
      <td>49.93</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Skinlab</td>
      <td>Revolting Room</td>
      <td>2002</td>
      <td>Groove Metal (early), Nu-Metal (later)</td>
      <td>5</td>
      <td>9.60</td>
      <td>50.06</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Queensrÿche</td>
      <td>Dedicated to Chaos</td>
      <td>2011</td>
      <td>Heavy/Power/Progressive Metal</td>
      <td>12</td>
      <td>34.33</td>
      <td>50.77</td>
    </tr>
  </tbody>
</table>
</div>

#### Best and worst bands

To accumulate a high weighted-average score, a band must put out many albums,
each garnering many positive reviews.
Many of the most popular bands in metal do poorly in this sense,
as they set very high standards for themselves and almost always have one or two albums
that far underperform compared to the rest of their discography.
This leaves us with some surprise appearances at the top of the weighted-average rankings:
[Type O Negative](https://en.wikipedia.org/wiki/Type_O_Negative)
takes the crown for highest weighted score.
Over a seven-album discography, they boast a minimum album rating of 83%
(on their delectably named [The Origin of the Feces](https://en.wikipedia.org/wiki/The_Origin_of_the_Feces)).
There's quite a lot of variety the top few bands,
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
      <td>Type O Negative</td>
      <td>Gothic/Doom Metal</td>
      <td>75</td>
      <td>91.79</td>
      <td>89.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Evoken</td>
      <td>Funeral Doom/Death Metal</td>
      <td>31</td>
      <td>94.84</td>
      <td>89.15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Moonsorrow</td>
      <td>Folk/Pagan/Black Metal</td>
      <td>72</td>
      <td>91.06</td>
      <td>88.72</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angra</td>
      <td>Power/Progressive Metal</td>
      <td>77</td>
      <td>90.83</td>
      <td>88.66</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Acid Bath</td>
      <td>Sludge/Doom Metal</td>
      <td>24</td>
      <td>95.29</td>
      <td>88.47</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Satan</td>
      <td>NWOBHM, Heavy Metal</td>
      <td>26</td>
      <td>94.04</td>
      <td>88.04</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Primordial</td>
      <td>Celtic Folk/Black Metal</td>
      <td>69</td>
      <td>90.01</td>
      <td>87.82</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Esoteric</td>
      <td>Funeral Doom/Death Metal</td>
      <td>32</td>
      <td>92.34</td>
      <td>87.68</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Ruins of Beverast</td>
      <td>Atmospheric Black/Doom Metal</td>
      <td>34</td>
      <td>91.82</td>
      <td>87.52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Morbid Saint</td>
      <td>Thrash Metal</td>
      <td>24</td>
      <td>93.12</td>
      <td>87.24</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Ahab</td>
      <td>Funeral Doom Metal</td>
      <td>35</td>
      <td>91.03</td>
      <td>87.08</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Solitude Aeturnus</td>
      <td>Epic Doom Metal</td>
      <td>27</td>
      <td>92.19</td>
      <td>87.07</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Forefather</td>
      <td>Black/Viking Metal</td>
      <td>18</td>
      <td>94.61</td>
      <td>87.01</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Kyuss</td>
      <td>Stoner Rock/Metal</td>
      <td>39</td>
      <td>90.46</td>
      <td>86.97</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Devin Townsend</td>
      <td>Progressive Metal, Ambient</td>
      <td>50</td>
      <td>89.66</td>
      <td>86.95</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Threshold</td>
      <td>Progressive Metal</td>
      <td>35</td>
      <td>90.74</td>
      <td>86.89</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Immolation</td>
      <td>Death Metal</td>
      <td>96</td>
      <td>88.27</td>
      <td>86.87</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Pentagram</td>
      <td>Doom Metal</td>
      <td>53</td>
      <td>89.32</td>
      <td>86.81</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Mercyful Fate</td>
      <td>Heavy Metal</td>
      <td>56</td>
      <td>89.14</td>
      <td>86.78</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Candlemass</td>
      <td>Epic Doom Metal</td>
      <td>109</td>
      <td>87.96</td>
      <td>86.75</td>
    </tr>
  </tbody>
</table>
</div>

Yet again [Waking the Cadaver](https://en.wikipedia.org/wiki/Waking_the_Cadaver)
take up their position at the bottom of this ranking.
It's mostly due to that horrid debut album of theirs;
it makes up 33 of their 48 overall reviews.
Their newest album, released just last year,
at least managed to get a single 80% review...

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
      <td>Waking the Cadaver</td>
      <td>Slam/Brutal Death Metal/Deathcore (early), Dea...</td>
      <td>48</td>
      <td>32.35</td>
      <td>45.42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hellyeah</td>
      <td>Groove Metal</td>
      <td>22</td>
      <td>23.73</td>
      <td>49.13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Skinlab</td>
      <td>Groove Metal (early), Nu-Metal (later)</td>
      <td>15</td>
      <td>19.33</td>
      <td>52.49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thrash or Die</td>
      <td>Thrash Metal</td>
      <td>15</td>
      <td>19.87</td>
      <td>52.73</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Winds of Plague</td>
      <td>Symphonic Deathcore</td>
      <td>34</td>
      <td>38.79</td>
      <td>53.09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Six Feet Under</td>
      <td>Death/Groove Metal, Death 'n' Roll</td>
      <td>121</td>
      <td>50.52</td>
      <td>54.35</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Animae Capronii</td>
      <td>Black Metal</td>
      <td>10</td>
      <td>22.60</td>
      <td>59.48</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Huntress</td>
      <td>Heavy Metal</td>
      <td>13</td>
      <td>32.15</td>
      <td>59.91</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Machine Head</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>123</td>
      <td>57.51</td>
      <td>60.38</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Whitechapel</td>
      <td>Deathcore</td>
      <td>75</td>
      <td>55.71</td>
      <td>60.40</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Car Door Dick Smash</td>
      <td>Thrash Metal/Grindcore</td>
      <td>6</td>
      <td>6.67</td>
      <td>61.61</td>
    </tr>
    <tr>
      <th>12</th>
      <td>In Flames</td>
      <td>Melodic Death Metal (early), Melodic Groove Me...</td>
      <td>184</td>
      <td>60.72</td>
      <td>62.43</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Damageplan</td>
      <td>Groove Metal</td>
      <td>10</td>
      <td>31.30</td>
      <td>62.55</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Enmity</td>
      <td>Brutal Death Metal</td>
      <td>20</td>
      <td>47.80</td>
      <td>63.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The Great Kat</td>
      <td>Speed/Thrash Metal, Shred</td>
      <td>14</td>
      <td>42.29</td>
      <td>63.44</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Liturgy</td>
      <td>Experimental Black Metal</td>
      <td>36</td>
      <td>55.67</td>
      <td>63.74</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Lacuna Coil</td>
      <td>Gothic Metal/Rock (early); Alternative Rock (l...</td>
      <td>81</td>
      <td>60.35</td>
      <td>63.90</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Underoath</td>
      <td>Metalcore (early), Alternative Rock/Post-Hardc...</td>
      <td>39</td>
      <td>57.08</td>
      <td>64.28</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Blackguard</td>
      <td>Folk/Melodic Death Metal</td>
      <td>11</td>
      <td>40.91</td>
      <td>65.08</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Brain Drill</td>
      <td>Brutal/Technical Death Metal</td>
      <td>33</td>
      <td>57.12</td>
      <td>65.14</td>
    </tr>
  </tbody>
</table>
</div>

#### Best and worst genres

Funeral (doom) metal tops the chart of total album score.
Interestingly, Middle-Eastern metal (which gets split up into middle and eastern because of the genre tag parsing)
does very well, although I think this mostly shows that the weighted average
does not punish low review counts very much when the distribution of un-weighted averages
is less spread out.
Because of this I'm not sure the metric is as informative when looking at genres.
That said, I think there's a bit of a preference here for genres that emphasize long,
instrumentally-focused song structures and fewer lyrics, most notably funeral/doom metal,
atmospheric metal, and avant-garde metal.
I can see how those genres are hard to hate, and therefore attract high rates of positive reviews.
Reviewers might also find it easier to assess the quality of the albums on the basis of instrumentation alone.
Finally, these genres are probably easier to listen to frequently enough to inspire writing reviews.

It should be noted that avant-garde appears more often as a modifier to other genres than as a genre on its own.
This may suggest that the more innovative bands in their respective genres,
such as avant-garde black metal (where the term is probably used the most)
attract more consistently positive reviews.

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
      <td>funeral</td>
      <td>396</td>
      <td>85.67</td>
      <td>84.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>eastern</td>
      <td>120</td>
      <td>86.82</td>
      <td>84.57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>middle</td>
      <td>103</td>
      <td>87.03</td>
      <td>84.46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>avant-garde</td>
      <td>974</td>
      <td>83.90</td>
      <td>83.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>grunge</td>
      <td>120</td>
      <td>85.47</td>
      <td>83.60</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pagan</td>
      <td>363</td>
      <td>84.05</td>
      <td>83.45</td>
    </tr>
    <tr>
      <th>7</th>
      <td>doom</td>
      <td>4710</td>
      <td>83.26</td>
      <td>83.21</td>
    </tr>
    <tr>
      <th>8</th>
      <td>viking</td>
      <td>843</td>
      <td>83.40</td>
      <td>83.16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>epic</td>
      <td>895</td>
      <td>83.36</td>
      <td>83.13</td>
    </tr>
    <tr>
      <th>10</th>
      <td>atmospheric</td>
      <td>1305</td>
      <td>83.05</td>
      <td>82.90</td>
    </tr>
  </tbody>
</table>
</div>

At the bottom, deathcore, slam-metal, and nu-metal are perhaps to no one's surprise the lowest-scoring genres on MA.
Whether you like them or not, it seems they simply don't match the preferences of the typical MA reviewer.

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
      <td>deathcore</td>
      <td>782</td>
      <td>65.29</td>
      <td>66.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>slam</td>
      <td>209</td>
      <td>63.74</td>
      <td>66.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nu-metal</td>
      <td>453</td>
      <td>65.68</td>
      <td>66.90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>acoustic</td>
      <td>54</td>
      <td>59.59</td>
      <td>68.49</td>
    </tr>
    <tr>
      <th>5</th>
      <td>metalcore</td>
      <td>1137</td>
      <td>68.40</td>
      <td>68.81</td>
    </tr>
    <tr>
      <th>6</th>
      <td>alternative</td>
      <td>616</td>
      <td>68.12</td>
      <td>68.87</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pop</td>
      <td>62</td>
      <td>63.92</td>
      <td>70.30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>groove</td>
      <td>2880</td>
      <td>70.37</td>
      <td>70.50</td>
    </tr>
    <tr>
      <th>9</th>
      <td>n</td>
      <td>343</td>
      <td>70.44</td>
      <td>71.44</td>
    </tr>
    <tr>
      <th>10</th>
      <td>roll</td>
      <td>343</td>
      <td>70.44</td>
      <td>71.44</td>
    </tr>
  </tbody>
</table>
</div>

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
      <td>184</td>
      <td>85.67</td>
      <td>84.74</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>630</td>
      <td>84.76</td>
      <td>84.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Iceland</td>
      <td>47</td>
      <td>87.53</td>
      <td>84.23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Israel</td>
      <td>95</td>
      <td>85.48</td>
      <td>83.94</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hungary</td>
      <td>85</td>
      <td>85.35</td>
      <td>83.72</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Colombia</td>
      <td>64</td>
      <td>85.41</td>
      <td>83.39</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Malta</td>
      <td>33</td>
      <td>87.00</td>
      <td>83.24</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Andorra</td>
      <td>27</td>
      <td>87.19</td>
      <td>82.94</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Singapore</td>
      <td>53</td>
      <td>84.79</td>
      <td>82.74</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Japan</td>
      <td>588</td>
      <td>82.78</td>
      <td>82.60</td>
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
      <td>Philippines</td>
      <td>19</td>
      <td>67.79</td>
      <td>74.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pakistan</td>
      <td>7</td>
      <td>60.71</td>
      <td>75.91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Italy</td>
      <td>988</td>
      <td>76.50</td>
      <td>76.58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Botswana</td>
      <td>9</td>
      <td>67.22</td>
      <td>76.61</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Belarus</td>
      <td>11</td>
      <td>70.45</td>
      <td>77.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>International</td>
      <td>96</td>
      <td>76.32</td>
      <td>77.06</td>
    </tr>
    <tr>
      <th>7</th>
      <td>United States</td>
      <td>16069</td>
      <td>77.51</td>
      <td>77.51</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Unknown</td>
      <td>25</td>
      <td>75.48</td>
      <td>77.60</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Korea, South</td>
      <td>11</td>
      <td>73.55</td>
      <td>77.80</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Belgium</td>
      <td>218</td>
      <td>77.71</td>
      <td>77.91</td>
    </tr>
  </tbody>
</table>
</div>

## Geographic distribution of albums and genres

Now we'll look at some geographic numbers describing the popularity of metal around the world.

#### Countries of origin

The U.S. clearly dominates in album production (or at least the production of albums that are cataloged on MA;
although I'm quite confident the selection bias is not too strong considering how active the MA community is).
Most of the top countries in the scene are of course European.

    
![png](/assets/images/heavy-metal-lyrics/reviews/donut_albums.png)


#### U.S. states of origin

Within the U.S. itself, California produces the most metal albums,
followed by a few other high-population states.
Per capita, however, Washington state comes out on top,
with nearly 24 albums per million people.

![png](/assets/images/heavy-metal-lyrics/reviews/donut_states.png)


#### Top countries in each genre

The U.S. is still the top producer of albums in most genres,
but when it comes to the "melodic" genre tag, which most often
is associated with death metal, the Swedes rise to the top.
The Germans take up an impressively large plurality of speed metal albums.
    
![png](/assets/images/heavy-metal-lyrics/reviews/donut_genres.png)
    

#### Top genres in each country

For most countries, death and black metal are the most common genres.
The Greeks have the highest plurality of any genre,
with nearly a third of Greek albums coming from bands identified as black metal.
The largest stake power metal takes is in Germany,
while thrash surprisingly is most proportionately popular in Japan.
    
![png](/assets/images/heavy-metal-lyrics/reviews/donut_countries.png)
    


## Decline of top-rated bands

    
![png](/assets/images/heavy-metal-lyrics/reviews/decline.png)
    


## Global album trends

Now for a more comprehensive look at the history of metal, as told by the MA community.
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

Although album production has been in a bit of a Dark Age in the last few years,
and review rates for newer albums still trail behind those that have
been around for a while, records from the last few years are performing much better
than those from the 2000s and early 2010s.
The annual weighted average is almost back to matching that of albums from the 70s.

![png](/assets/images/heavy-metal-lyrics/reviews/global.png)


#### Yearly album output by genre

Although most genres contributed to the rise of metal in the 2000s,
black and death metal dominated the trend.
Before then, it was heavy metal and thrash metal that ruled supreme,
spearheading the 80s upswing that brought heavy metal into the public spotlight.

    
![png](/assets/images/heavy-metal-lyrics/reviews/genre_albums.png)
    


#### Yearly average album score

Here we see both genre-specific and global trends in the annual average of weighted-average scores.
Early doom metal (led by Black Sabbath) is very highly rated, and even more recent doom metal albums
continue to outperform material from other genres rather consistently.
Early black metal albums likewise had a high-scoring start, but have mostly converged towards
the overal average since then.
Death metal enjoyed its peak in the early nineties, followed by a long brutal decline throughout the 2000s.
Progressive rock/metal peaked twice, once in the late-80s wave (perhaps thanks to Rush),
and again the early 2000s as progressive metal started to take shape.
Recently it's been the most highly-rated genre after doom metal.
    
![png](/assets/images/heavy-metal-lyrics/reviews/genre_scores.png)
    


## Geographic trends

This plot shows just how consistently American bands have dominated the metal scence over the decades.
The U.K. was the first nation to pull ahead of others in producing metal albums,
but with the rise of the thrash era in the late 80s,
the U.S.A. took the lead and never lost it,
despite the Scandinavian black/death metal scene and the German power metal bands
pulling some attention back across the Atlantic.
    
![png](/assets/images/heavy-metal-lyrics/reviews/country_albums.png)


Despite producing the most albums, the popularity of U.S. bands have never quite
broken beyond their Scandinavian counterparts.
After the initial peak in U.K. scores, the Scandinavians took over and continued
to put out the highest-rated albums throughout the decades.
    
![png](/assets/images/heavy-metal-lyrics/reviews/country_scores.png)
    


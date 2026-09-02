# Read this first — the whole project in 30 seconds

## The question

If you know what someone bought over the last 6 months, can you tell whether
they'll buy anything at all in the next 3 months?

## The data

Real Amazon purchase histories that 5,027 American shoppers volunteered to a
research study — **1.85 million purchases** between 2018 and 2022. Each row is
one item: what it was, what it cost, when it was ordered. A separate survey
file gives each person's age, gender, state, income, education, ethnicity and
household size.

## How we did it

Pick a date. Look **only backwards** 180 days to describe the customer — how
recently they bought, how often, how much they spent, how many different
things and categories. Then look **only forwards** 90 days to see what actually
happened: did they buy, or go quiet?

```
   last 180 days          |          next 90 days
   ---------------------- | ----------------------
   describe the customer  |  did they buy? yes/no
                    the cutoff date
```

We did this at four dates — three to teach the model, one held back to test it.
The test date comes *after* all the training dates, so the model is always
predicting forward in time, never peeking at the future. Then we trained three
different algorithms on three different sets of information (shopping
behaviour only, demographics only, both) — nine experiments in total.

## What we found

**1. Yes, past buying predicts future buying — quite well.**
The model scores **0.90 ROC-AUC** (0.50 would be a coin flip). At a practical
setting it correctly spots **65% of the customers who go quiet**, though half
the people it flags would have bought anyway.

**2. How *recently* someone bought matters most — but it saturates.**
Customers who bought in the last week returned 98% of the time. Those who
hadn't bought in 4–6 months returned only 45% of the time. But past a modest
threshold, more activity tells you nothing extra: someone who shopped 40 days
out of 180 looks the same to the model as someone who shopped 20 days. Spending
$2,000 predicts no better than spending $500. The model is really separating
*barely active* from *active* — not casual shoppers from heavy ones.

A related surprise: most of our 17 measurements turned out to be the same
measurement wearing different hats. Counting purchases, counting shopping days,
counting distinct products, counting categories and totalling spend all move
together almost perfectly — because someone who buys more of anything buys more
of everything. Only "how recently" was genuinely separate information.

**3. Demographics added essentially nothing.**
Age, income, gender, state, education and ethnicity, on their own, do beat
random guessing. But once you already know how someone shops, adding who they
are improves predictions by roughly **0.003** — indistinguishable from noise,
and the direction of the "improvement" flipped depending on which measure we
looked at. Who you are is already reflected in how you shop.

This third finding is a negative result, and we report it as prominently as the
other two.

## The most honest thing in the project

The model's most confident mistake and its most confident correct call had
**identical inputs** — both customers had bought nothing in the previous 180
days. One stayed away; one came back. On the information available, they are
indistinguishable. No amount of model tuning fixes that, because the answer
isn't in the data. Roughly 350 customers per snapshot fall in this blind spot.

## The catch we nearly missed

Each person's purchase record stops on the day they *joined the study*, not on
a shared end date. If we'd predicted into late 2022, hundreds of customers
would have been labelled "stopped buying" when the truth was "we stopped
watching." We measured how bad this got — label reliability fell from 97% to
61% as we moved the cutoff later — and moved the whole analysis back to
mid-2021 to avoid it. That single decision cost us a year of recency and is the
most important methodological choice in the project.

## What you'd actually do with this

Flag the quiet customers and send them a cheap re-engagement email — you'd
reach two-thirds of the people about to lapse. Don't send them an expensive
discount, because half of them were going to buy anyway and you'd be giving
away margin for nothing.

## One important caveat

Everything here is **association, not cause**. Customers who buy often are more
likely to buy again. That does *not* mean making someone buy more often will
make them loyal. This is observational data from one marketplace, one
volunteer panel, one time period.

# RWGAN/RandomSamplingGAN

## Overview
This repository contains a novel solution to the problem of survey reweighting: Taking a cheap biased non-probability survey and a census and finding weights for the survey such that the survey becomes more representative of the population level statistics. 

## Problem Statement
Assumptions and framing
- We assume the existence of a Census data set, C, that has d demographic variables
- We are given a small survey, S, that shares the same d demographic variables
  - S also contains a target variable, v, (such as 'Have you received at least one dose of vaccine')
- |C| >> |S|
- C and S are drawn from different distributions, P(X,Y) and Q(X,Y) respectively, that share the same support
  - e.g. Survey takers tend to be higher percentage female and older than population means.

Question: Can we extract census level predictions from a small, cheap, online survey framed above. 

## Results on 2021 Vaccine Uptake
| Date (Wave ID) | Our Method (%) | CDC (%) | HHP (%) | Raking (%) |
| :--- | :---: | :---: | :---: | :---: |
| Feb 1st, 2021 (23) | **10.5** | 11.8 | 13.0 | 11.6 |
| Feb 15th, 2021 (24) | **15.0** | 17.2 | 20.1 | 17.0 |
| March 1st, 2021 (25) | **20.3** | 22.2 | 25.1 | 23.2 |
| March 15th, 2021 (26) | **30.4** | 30.9 | 34.4 | 31.4 |
| March 29th, 2021 (27) | **39.4** | 40.2 | 47.8 | 43.3 |
| April 26th, 2021 (28) | **55.0** | 56.7 | 69.9 | 65.3 |
| May 10th, 2021 (29) | **62.5** | 60.5 | 74.2 | 69.3 |

Table 1. Results of our method compared to Raking and Survey Collector Household Pulse (HHP) weights. Benchmark values are given in the column "CDC". 

Our method is significantly closer to benchmark (CDC) values compared to traditional methods like Raking and the prediction of the survey collector themselves (HHP). This trend holds true for multiple waves of the HHP survey and the D4P surveys. 

**We reduce error by up to 14% for predicting vaccine uptake over the first half of 2021**

## Methodology

The solution is an adaptation of Wasserstein Generative Adversarial Networks (WGANs). We adapt a WGAN generator to produce a weight distribution over a survey rather than synthesize new data. 

![Alt text](./images/RWGANDiagram.png)

Figure 1. Overall methodology diagram. GT = Census data.

### Algorithm

The algorithm adapts the critic/generator structure from WGANs. 

Critic
- Assign high scores to mini-batches sampled from the census/GT data set
- Assign low scores to mini-batches sampled from the survey

Generator
- Create a weight distribution, s, over S such that mini-batches sampled of S using w lead critic to assign erroneously high scores.

### Comparison methods and data sets

We test our method against the following established survey analysis methods:
- Raking*
- Multi-level Regression and Post-Stratification
- Expectation Maximization (EM)
- Support Vector Machine (SVM)
- Survey Collectors Weights* (exact methods are not disclosed by the survey collectors)

We test our method on the following survey sets:
- Household Pulse (7 surveys)*
- Data 4 Progress (6 Surveys)
- Axios Ipsos (7 Surveys)
  
* = results displayed in this ReadMe

## Feature

# Introduction to Machine Learning (67577) - Hackathon 2022 - Waze Challenge

![GitHub last commit](https://img.shields.io/github/last-commit/OrrMatzkin/IML-Hackathon-Waze)
![GitHub issues](https://img.shields.io/github/issues/OrrMatzkin/IML-Hackathon-Waze?color=yellow)
![GitHub pull requests](https://img.shields.io/github/issues-pr/OrrMatzkin/IML-Hackathon-Waze?color=yellow)
![GitHub repo size](https://img.shields.io/github/repo-size/OrrMatzkin/IML-Hackathon-Waze)
![GitHub](https://img.shields.io/github/license/OrrMatzkin/IML-Hackathon-Waze?color=orange)

## About the project

Unless you live on the moon you probably used Waze for navigation when driving.
The app’s ability to plan the best route and react to new events made it popular.
But merely reacting to events isn’t enough - the ultimate goal is to predict events
and solve congestion problems before they occur. 

As part of The Hebrew University of Jerusalem course _"Introduction to Machine Learning (67577)"_, the four of us participated in a hackathon, determined
to answer the following questions:
1. What is the most likely next event given a sequence of Waze events?
2. What is the distribution of Waze events in a given time point?

## Table of context
- [Dataset](#dataset)
- [Model Tasks](#tasks)
    - [Next Event Prediction](#next-event-prediction)
    - [Event Distribution Prediction](#event-distribution-prediction)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Run Locally](#run-locally)   
- [Dive In](#dive-in)    
  - [Preprocess](#reprocess)
  - [Next Event](#next-event)                     
  - [Event Distribution](#event-distribution)     
       


## Dataset
The dataset holds about 18K real traffic events from the Waze application,
collected between 2021-02-07 to 2022-05-24. Each row describes a single event and hold 19 features:

| Feature                   | Description                             | Type   | Example                 |
|---------------------------|-----------------------------------------|--------|-------------------------|
| id                        | Unique identifier for the event Cell    | int    | 16519                   |
| linqmap_type              | describing the event family             | string | JAM                     |
| linqmap_subtype           | describing the event in details         | string | JAM_STAND_STILL_TRAFFIC |
| pubDate                   | the report date                         | string | 05/15/2022 09:31:17     |
| linqmap_reportDescription | description of the event (Hebrew)       | string | -                       |
| linqmap_street            | the street name (Hebrew)                | string | תל אביב - יפו           |
| linqmap_nearby            | interest points near the event (Hebrew) | string | שתולים                  |
| linqmap_roadType          | road type code                          | string | 6                       |
| linqmap_reportMood        | user mood (as assessed by Waze)         | string | 0                       |
| linqmap_reportRating      | report rating                           | int    | 5                       |
| linqmap_expectedBeginDate | event expected beginning                | string | -                       |
| linqmap_expectedEndDate   | event expected ending                   | string | -                       |
| linqmap_magvar            | orientation w.r.t to the north pole     | int    | 244                     |
| nComments                 | comments                                | string | 0                       |
| linqmap_reliability       | event reliability                       | int    | 9                       |
| update_date               | when the event was last updated         | string | 1652608382312           |
| x                         | x coordinate of the event               | int    | 180774.21999999974      |
| y                         | y coordinate of the event               | int    | 661479.4800000004       |

*The only features that are guaranteed to be present are ID, linqmap_type, x, y.

**The dates are given in POSIX time.

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Model Tasks
As mentioned, in this Hackathon we were asked to answer 2 independent question. Therefore, this program runs 2 
independent task: 
1. Predict Next Event.
2. Predict Event Distribution.

### Next Event Prediction

Given a sequence of 4 consecutive events in Tel-Aviv (ordered by time) predict the next event.
That is, given a sequence of 4 events $x_1,...,x_4$ predict the following features of the 5th event:
(linqmap_type, linqmap_subtype, x coordinate, y coordinate).

#### Input & Output
The input for this problem is a dataframe with groups of 4 events in Tel Aviv with same structure as the
training data and a number indicating which group they belong to (the last column).

The output is a dataframe with a single row per group and 4 columns corresponding to the values above.

#### Evaluation
In this section the evaluation method is a weighted combination of F1-macro loss for
linqmap_type, linqmap_subtype and l2 loss for the location - $(\hat{x} − x)^2 + (\hat{y} − y)^2$.


### Event Distribution Prediction
Given a time range (start-end) predict the distribution of events across the nation.
That is, for the following 3 time slots 8:00-10:00, 12:00-14:00, 18:00-20:00
predict the number of events of each type.

#### Input & Output
The input is one of the dates: 05.06.2022, 07.06.2022 and 09.06.2022.

The output is a 3 by 4 table where each row corresponds to a time slot, the columns match the linqmap_type (ACCIDENT, JAM, ROAD_CLOSED, WEATHERHAZARD).

#### Evaluation
In this section the grading is computed by the following weighted MSE:
<div align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://latex.codecogs.com/svg.image?%5Csum_%7Brow%7D%5E%7B%7D%5Csum_%7Bt%7D%5E%7B%7D%5Cfrac%7B(%5Chat%7By%7D_%7Bevent,%5C%20t%7D-y_%7Bevent,%5C%20t%7D)%5E2%7D%7By_%7Bevent,%5C%20t%7D&plus;1%7D">
  <source media="(prefers-color-scheme: dark)" srcset="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5Csum_%7Brow%7D%5E%7B%7D%5Csum_%7Bt%7D%5E%7B%7D%5Cfrac%7B(%5Chat%7By%7D_%7Bevent,%5C%20t%7D-y_%7Bevent,%5C%20t%7D)%5E2%7D%7By_%7Bevent,%5C%20t%7D&plus;1%7D">
  <img alt="MSE">
</picture>

[//]: # (![MSE_dark]&#40;https://latex.codecogs.com/svg.image?%5Csum_%7Brow%7D%5E%7B%7D%5Csum_%7Bt%7D%5E%7B%7D%5Cfrac%7B&#40;%5Chat%7By%7D_%7Bevent,%5C%20t%7D-y_%7Bevent,%5C%20t%7D&#41;%5E2%7D%7By_%7Bevent,%5C%20t%7D&plus;1%7D#gh-light-mode-only&#41;)

[//]: # (![MSE_light]&#40;https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%5Csum_%7Brow%7D%5E%7B%7D%5Csum_%7Bt%7D%5E%7B%7D%5Cfrac%7B&#40;%5Chat%7By%7D_%7Bevent,%5C%20t%7D-y_%7Bevent,%5C%20t%7D&#41;%5E2%7D%7By_%7Bevent,%5C%20t%7D&plus;1%7D#gh-drark-mode-only&#41;)
</div>
where $\hat{y}_{event, t}$ is the number of predicted of events of some type at time t,
$y_{event, t}$ is the actual number of events of that type at time t.

## Getting Started

Our model requires ```Python 3.7+``` to run.


### Installation

1. Clone the repo and enter the project directory:
   ```bash
   git clone https://github.com/OrrMatzkin/IML.Hackathon.Waze.git
   cd IML.Hackathon.Waze
   ```
2. Install and run a virtualenv, isolated Python environment (this step is not mandatory but recommended):
    ```bash
    pip3 install virtualenv
    virtualenv IML.Hackathon.Waze
    source IML.Hackathon.Waze/bin/activate
   ```

3. The ```requirements.txt``` file lists all Python libraries that our program depends on, they will be installed using:
   ```bash
   pip3 install -r requirements.txt
   ```

### Run Locally

The program is set to run both tasks. The program needs ```data``` to train its models, the Next Event Prediction task
also needs ```take_features``` sequences of 4 consecutive events. Therefore, the program requires 2 arguments in total to run
(we already supplied real time data):

```bash
python3 main.py data/waze_data.cvs data/waze_take_features.csv
```
While the program runs it will update you of its stage. After the program would train the models it will run both tasks
and save the prediction as csv, as defied in the tasks section.

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Dive In

In the next few sections we will walk you through how are program and models work.

### Preprocess

Before even approaching the task, we saved 20% of the data for a last minute test, as the data we have been given is 
precious and of-course limited.

Then we looked and examined the Data, we wanted to figure out what features we hold and how the data is represented.
We found out that we have two types of dates, many features about location and little data about the reporters themselves. 

Before we tried to understand what features we wish to keep and what new one to create by computing the correlation
between them, we cleaned the data with a couple of ways including:
- Getting rid of duplicates (by id).
- Filling missing data by analyze same samples with close destination and time to the area
- Converting dates to date format.
- Finding correlation between subtypes and location inside and outside of town.
- Creating dummies values for non-numeric features
- And many more...
 
One example of what we succeeded to learn from the raw data is where most of the events (by type) occurs geographically.
We saw by to printing (x,y) location of events that most of the jams are in Tel-Aviv (no surprise here)

<div align="center">
<img src="https://github.com/OrrMatzkin/IML.Hackathon.Waze/blob/main/figures/x_y_events_map.png?raw=true" alt="drawing" width="600"/>
</div>

### Next Event   



### Event Distribution       










<p align="right">(<a href="#about-the-project">back to top</a>)</p>


See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).


## Contributors

<a href="https://github.com/OrrMatzkin/IML-Hackathon-Waze/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OrrMatzkin/IML-Hackathon-Waze" width="200"/>
</a>

Thank you for reading and having interest in our hackathon project...


<div align="center">
<img src="https://github.com/OrrMatzkin/IML-Hackathon-Waze/blob/main/figures/us.jpeg?raw=true" alt="drawing" width="600"/>
</div>

## Copyright

MIT License

Copyright (c) 2022 OrrMatzkin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.BS7HXHXp0QKBBe6sNYvjJZ0/edit?usp=sharing).
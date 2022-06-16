# Introduction to Machine Learning (67577) - Hackathon 2022 - Waze Challenge

![GitHub last commit](https://img.shields.io/github/last-commit/OrrMatzkin/IML.Hackathon.Waze=orange)
![GitHub issues](https://img.shields.io/github/issues/OrrMatzkin/IML.Hackathon.Waze?color=yellow)
![GitHub pull requests](https://img.shields.io/github/issues-pr/OrrMatzkin/IML.Hackathon.Waze?color=yellow)
![GitHub repo size](https://img.shields.io/github/repo-size/OrrMatzkin/IML.Hackathon.Waze)
![GitHub](https://img.shields.io/github/license/OrrMatzkin/IML.Hackathon.Waze)

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
    - [Requirements](#requirements)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Run Locally](#run-locally)   
- [Features](#features)        
- [Adding new Music Videos](#adding-new-music-videos)    
       


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
2. Install and run a virtualenv isolated Python environment (this step is not mandatory but recommended):
    ```bash
    pip3 install virtualenv
    virtualenv IML.Hackathon.Waze
    source IML.Hackathon.Waze/bin/activate
   ```

3. The ```requirements.txt``` file should list all Python libraries that your notebooks depend on, and they will be installed using:
   ```bash
   pip3 install -r requirements.txt
   ```

### Run Locally


The program is set to run both tasks. Both tasks need the ```data``` to train their models, the Next Event Prediction task
also needs ```take_features``` sequences of 4 consecutive events. Therefore, the program requires 2 arguments in total to run:

```bash
python3 main.py data/waze_data.cvs data/waze_take_features.csv
```

The program would train the models, run both tasks and save the prediction as csv, as defied in the tasks section.

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Features

- [x] Play/stop local video songs
- [x] Show available songs
- [ ] Increase/decrease volume
- [ ] Play/stop youtube video songs (pafy integratiion)
- [ ] Create and play a playlist of songs
  - [ ] By artist name
  - [ ] By a preloaded playlist 
- [ ] Remove the need for a json data file
- [ ] A Better score mechanism for songs
- [ ] A "hard" integrated assistant control


See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

## Adding new Music Videos

This Repo comes with 6 (great) song:

1. David Bowie - Space Oddity.mp4
2. Louis Armstrong - What a Wonderful World.mp4
3. Marvin Gaye Tammi Terrell - Aint No Mountain High Enough.mp4
4. Oasis - Wonderwall.mp4
5. Roy Orbison - You Got It.mov
6. The Police - Every Breath You Take.mp4

To add a new song follow this steps:

1. Download your favorite music video. *
2. Rename the file to "\<Artist Name\> - \<Song Name\>.\<FIle Format\>" (see exmaple above).
4. Move the file to the `songs` directory.
3. Add the song details in `songs_data.json`
   ```json
   {
        "id": 0,
        "name": "<Song Name>",
        "artist": "<Artist Name>",
        "path": "<Path to song fille>",
        "matches": ["<match1>", "<match2>", "<match3>", "<match4>",...]
    }
   ```

\* VLC supports: ASF, AVI, FLAC, FLV, Fraps, Matroska, MP4, MPJPEG, MPEG-2 (ES, MP3), Ogg, PS, PVA, QuickTime File Format, TS, WAV, WebM

\*\* The Matches field is how a song is picked after a voice command. For an example check the given `songs_data.json` file

<p align="right">(<a href="#about-the-project">back to top</a>)</p>

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
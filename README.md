<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
<h3 align="center">Gym Super Mario AI Agents</h3>

  <p align="center">
    This project involves building Artificial Intelligence agents to play Gym Super Mario Bros using various algorithms.
    <br />
    <a href="https://github.com/KhanhHua2102/CITS3403-Project"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/KhanhHua2102/super-mario-ai/">View Demo</a>
    ·
    <a href="https://github.com/KhanhHua2102/super-mario-ai/issues">Report Bug</a>
    ·
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

# About The Project

This project is about building an Artificial Intelligence agent to play Super Mario. It utilizes various Artificial Intelligence algorithms such as rule-based learning, Monte Carlo Tree Search, Q-learning, and Proximal Policy Optimization Algorithms. In addition to implementations, comparisons including strengths and weaknesses and the performances for using these algorithms will be in the context of the Super Mario game.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

[![Python][python.com]][python-url]
[![Poetry][poetry.com]][poetry-url]
[![OpenCV][opencv.com]][opencv-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

# Getting Started

### Prerequisites

- python 3.8 or newer
- poetry 1.1.4 or newer

### Installation

#### Running the app locally by cloning the Github repo

1. Clone the repo

2. Install dependencies using poetry

```sh
poetry install
```

3. Activate the poetry environment

```sh
poetry shell
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

# Usage

**Run the specific agent as a module**

**Rule-based Agent Running**

```sh
python -m Rules_Based.rule_based_ai_faster
```

**Monte Carlo Tree Search Agent Training**

```sh
python -m Monte_Carlo.mcts
```

**Monte Carlo Tree Search Agent Running**

```sh
python -m Monte_Carlo.run
```

**Q-learning Agent Training**

```sh
python -m Q_Learning.q_obs
```

**Q-learning Agent Running**

```sh
python -m Q_Learning.run
```

**Proximal Policy Optimization Agent Training**

```sh
python -m PPO.ppo
```

**Proximal Policy Optimization Agent Running**

```sh
python -m PPO.run
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

**Quang Khanh Hua (Henry)**

**Yin Min Aung (Ivan)**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Quang Khanh Hua - henry@khanhhua2102.com

[![linkedin-shield]][linkedin-url]

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/khanhhua2102
[product-screenshot]: images/screenshot.png
[python.com]: https://img.shields.io/badge/python-3.8-blue?style=for-the-badge&logo=python
[python-url]: https://www.python.org/downloads/
[poetry.com]: https://img.shields.io/badge/poetry-1.1.4-blue?style=for-the-badge&logo=python
[poetry-url]: https://python-poetry.org/docs/
[opencv.com]: https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white
[opencv-url]: https://pypi.org/project/opencv-python/

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# mae-271d-project-algames
UCLA MAE 271D Spring 2025 Project 

This project implements a simplified ALGAMES algorithm from RSS 2020 [[1]](#1).

## Usage 
Run `sim.ipynb` to run the ALGAMES solver in a single example configuration.
Run `mpc.ipynb` to run the MPC wrapped ALGAMES solver in an example configuration.

The example configuration is currently set to the "lane-merge" example as detailed in the Homework 2 report.

Initial and final states (x, y, vx, vy), as well as weights on the cost and convergence criteria can be tuned in blocks 2 and 3 at the top of each `.ipynb` file.

## References
> <a id="1">[1]</a> Simon Le Cleac’h, Mac Schwager, and Zachary Manchester. “ALGAMES: A Fast Solver for Constrained
Dynamic Games”. In: Robotics: Science and Systems XVI. RSS2020. Robotics: Science and Systems
Foundation, July 2020. doi: 10.15607/rss.2020.xvi.091. url: https://doi.org/10.15607/RSS.2020.XVI.091.

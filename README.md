# Collective Fish Behavior: A Hydrodynamic Interaction Model

## Team members
- Jakob Petek ([@Friday202](https://github.com/Friday202))
- Andraž Čeh ([@andrazceh](https://github.com/andrazceh), [@student8694](https://github.com/andrazceh))
- Matic Šutar ([@AllIAskOfYou](https://github.com/AllIAskOfYou))
- Gregor Kovač ([@gregorkovac](https://github.com/gregorkovac))

## Project overview
This project is based on the paper [[1]](#1). The paper introduces a novel model of fish schooling that incorporates both behavioural rules and hydrodynamic interactions. Our goal is to implement, validate, and potentially extend this model, exploring the implications of hydrodynamic interactions on collective fish behaviour. We plan to add several extensions to this project, such as simulations of different fluids (e.g. fresh water and salty water) to see how they change the behaviour of fish, external water flows of different shapes to see how they change the movement patterns of a school and a predator to see if the fish can exploit hydrodynamics to develop better survival tactics.

A demo of the simulation can be seen [here](https://youtu.be/F9MiLQuiUbI?si=GKoHH3ob-DJ_hvHu).

## Repository Structure
- `docs/`: Contains all report documents, both source files and PDFs.
- `src/`: Contains the source code for the model implementation and analysis.
- `presentation/`: Contains the slides for the final presentation.

## Running the Simulation
### Prerequisites
- Python 3.x
- `pip` for installing Python packages

### Installation
1. Clone the repository or download the source code to your local machine.
2. Open a terminal or command prompt.
3. Navigate to the directory containing the project files.
4. Install the required libraries by running `pip install -r requirements.txt`.

### Running the Simulation
To run the simulation, follow these steps:
1. In the terminal or command prompt, navigate to the project's root directory.
2. Execute the command: `python main.py`.


![Simulation Demo](https://github.com/gregorkovac/collective-fish-behaviour/blob/master/simulation.gif)


## Milestones
- [x] [Milestone 1](https://github.com/gregorkovac/collective-fish-behaviour/milestone/1) 
- [x] [Milestone 2](https://github.com/gregorkovac/collective-fish-behaviour/milestone/2) 
- [x] [Milestone 3](https://github.com/gregorkovac/collective-fish-behaviour/milestone/3) 

## References
<a id="1">[1]</a> 
Audrey Filella, François Nadal, Clément Sire, Eva Kanso, Christophe Eloy. (May 2018).
Model of collective fish behavior with hydrodynamic interactions.
American Physical Society ({APS}), Physical Review Letters. 120, 19.

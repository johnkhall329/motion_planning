# Motion Planner for Autonomous Vehicle Overtaking

## Dependecies

To install all package dependencies for our project, we recommend creating a virtual environment. 
The required packages are contained in `requirements.txt` and should be installed using:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1 #for Windows powershell
source venv/bin/activate #for Linux

pip install -r requirements.txt
```

## Running our demo

To run our demo, you will need to run `sim.py`.

You should see the ego car (blue) start and go towards the non-ego car (red) and then find a constant velocity. To plan a path, you have two options:
- Press O for a full Hybrid A* path
- Press P for a full P-Dubins-RRT* path

The ego car should plan and execute the overtaking maneuver and return to the same lane it started in.

If you want to reset the simulation, simply press R.
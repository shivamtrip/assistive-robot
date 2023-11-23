
from ompl import base as ob
from ompl import geometric as og
 
def isStateValid(state):
    return state.getX() < .6
 
def plan():
    # create an SE2 state space
    space = ob.SE2StateSpace()
    # newspace = ob.CompoundStateSpace()
    # newspace.addSubspace(space, 1.0)
 
    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)
 
    # create a simple setup object
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
 
    start = ob.State(space)
    # we can pick a random start state...
    start.random()
    # ... or set specific values
    start().setX(.5)
 
    goal = ob.State(space)
    # we can pick a random goal state...
    goal.random()
    # ... or set specific values
    goal().setX(-.5)
 
    ss.setStartAndGoalStates(start, goal)
 
    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(1.0)
 
    if solved:
        # try to shorten the path
        ss.simplifySolution()
        # print the simplified path
        print (ss.getSolutionPath())
 
 
if __name__ == "__main__":
    plan()

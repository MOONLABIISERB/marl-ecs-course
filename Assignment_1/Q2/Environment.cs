using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using Classes;
using Microsoft.Win32.SafeHandles;

namespace Q2
{
    public class Environment
    {
        public int n { get; set; }
        public Dictionary<Coordinate, State> Walkable = new();
        public Dictionary<Coordinate, State> Obstacles = new();
        // public Dictionary<Coordinate, Coordinate> PortalsInOut;
        public Dictionary<Coordinate, State> PortalIn = new();
        public Dictionary<Coordinate, State> PortalOut = new();
        public Dictionary<Coordinate, State> Goal = new();

        public dynamic States;
        public dynamic Actions;
        public Environment(int n, List<Coordinate> obstacles, Coordinate portalIn, Coordinate portalOut, Coordinate goal)
        {
            this.n = n;
            StateSpace states = new StateSpace();

            foreach (var obstacle in obstacles)
            {
                State obstacleState = new State($"obstacle_{obstacle.X}_{obstacle.Y}", 0);
                Obstacles[obstacle] = obstacleState;
                states.Add(obstacleState);
            }

            State portalInState = new State($"portalIn_{portalIn.X}_{portalIn.Y}", 0);
            PortalIn[portalIn] = portalInState;
            states.Add(portalInState);

            State portalOutState = new State($"portalOut_{portalOut.X}_{portalOut.Y}", 0);
            PortalOut[portalOut] = portalOutState;
            states.Add(portalOutState);

            State goalState = new State($"goal_{goal.X}_{goal.Y}", 1);
            Goal[goal] = goalState;
            states.Add(goalState);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Coordinate current = new Coordinate(i,j);
                    if (!obstacles.Contains(current) &&
                        portalIn != current &&
                        portalOut != current &&
                        goal != current)
                    {
                        State walkableState = new State($"walkable_{current.X}_{current.Y}", 0);
                        Walkable[current] = walkableState;
                        states.Add(walkableState);
                    }
                }
            }

            States = states;

            ActionSpace actions = new ActionSpace();

            actions.Add("Left");
            actions.Add("Right");
            actions.Add("Up");
            actions.Add("Down");

            Actions = actions;
        }
    
    }
}
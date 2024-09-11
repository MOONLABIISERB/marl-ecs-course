using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Q2.Environment;

namespace Q2.RL
{
    public class MDP
    {
        GameWindow Gamewindow;
        Level Level;
        State[] States;
        
        public Dictionary<Point[], double> OptimalValue { get; set; }

        public MDP(GameWindow window, Level level, double gamma = 0.9, double epsilon = 1e-6)
        {
            Gamewindow = window;
            Level = level;
        }

        public void ValueIteration(out Dictionary<State, double> value, out Dictionary<State, Environment.Action> policy, double gamma, int numIter)
        {
            // bool isConverged = false;
            int current = 0;
            // Initialize Policies and values
            InitializePolicyAndValue(out policy,out value);
            while (current < numIter)
            {
                Dictionary<State, double> VNew = new Dictionary<State, double>();
                System.Console.WriteLine($"Iteration number:{current}");
                foreach (var state in States)
                {
                    double maxValue = double.NegativeInfinity;
                    Environment.Action bestAction = Environment.Action.Right;
                    foreach (Environment.Action action in Enum.GetValues(typeof(Environment.Action)))
                    {
                        Gamewindow.InitializeLevel();
                        Level.Player.Position = state.PlayerPosition;

                        for (int i = 0; i < Level.Boxes.Length; i++)
                        {
                            Level.Boxes[i].Position = state.BoxesPosition[i];
                        }
                        Level.MovePlayer(action);
                        Point[] boxesPos = new Point[Level.Boxes.Length];
                        for (int i = 0; i < boxesPos.Length; i++)
                        {
                            boxesPos[i] = Level.Boxes[i].Position;
                        }
                        State newState = new State(Level.Player.Position, boxesPos);

                        double actionValue = -1;
                        if (Level.LevelComplete())
                        {
                            actionValue = 0;
                        }
                        actionValue += gamma * value[newState];

                        if (actionValue > maxValue)
                        {
                            maxValue = actionValue;
                            bestAction = action;
                        }
                    }
                    
                    VNew[state] = maxValue;
                    policy[state] = bestAction;
                }
                value = VNew;
                current++;
            }
        }

        private void InitializePolicyAndValue(out Dictionary<State, Environment.Action> policy, out Dictionary<State, double> value)
        {
            // Initialize Policies and values
            policy = new Dictionary<State, Environment.Action>();
            value = new Dictionary<State, double>();
            States = GenerateStates();
            foreach (var state in States)
            {
                policy[state] = Environment.Action.Right;
                value[state] = 0;
            }
        }

        public State[] GenerateStates()
        {
            var positions = GetAllPositions();
            var states = new List<State>();
            // Generate all permutations for player + boxes
            var permutations = GetPermutations(positions, 1 + Level.Boxes.Length);
            foreach (var perm in permutations)
            {
                var state = new State(perm);
                bool walkable = true;
                // if (HasDuplicatePoints(perm))
                //     walkable = false;
                foreach (var point in perm)
                {
                    if (Level.Map[point.X, point.Y].Type == Extension.GameObjectType.Wall)
                        walkable = false;   
                }
                if (walkable)
                {
                    states.Add(state);
                }
            }

            return states.ToArray();
        }

        private List<Point> GetAllPositions()
        {
            var positions = new List<Point>();
            int rows = Level.Rows;
            int cols = Level.Columns;

            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    positions.Add(Level.Map[x,y].Position);
                }
            }

            return positions;
        }

        private IEnumerable<Point[]> GetPermutations(List<Point> list, int length)
        {
            if (length == 1)
                return list.Select(t => new Point[] { t });

            return GetPermutations(list, length - 1)
                .SelectMany(t => list.Where(e => !t.Contains(e)),
                            (t1, t2) => t1.Concat(new[] { t2 }).ToArray());
        }

        public bool HasDuplicatePoints(Point[] points)
        {
            HashSet<Point> seenPoints = new HashSet<Point>();
            foreach (var point in points)
            {
                if (!seenPoints.Add(point))
                {
                    return true; // A duplicate point was found
                }
            }
            return false; // No duplicates
        }

    }
}
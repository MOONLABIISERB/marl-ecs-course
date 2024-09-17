using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Q2.Environment;

namespace Q2.RL
{
    public class Reinforcement_Learning
    {
        GameWindow Gamewindow;
        Level Level;
        State[] States;
        static Random random = new Random();
        
        public Dictionary<Point[], double> OptimalValue { get; set; }

        public Reinforcement_Learning(GameWindow window, Level level, double gamma = 0.9, double epsilon = 1e-6)
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

        public void MCESFirstVisit(out Dictionary<State, Environment.Action> policy, double gamma, int numEpisodes, int maxSteps)
        {
            // Initialization
            States = GenerateStates();
            policy = new Dictionary<State, Environment.Action>();
            Dictionary<StateAction, double> Qvalue = new();
            Dictionary<StateAction, List<double>> Returns = new();
            HashSet<StateAction> stateAction = new();

            foreach (State state in States)
            {
                foreach (Environment.Action action in Enum.GetValues(typeof(Environment.Action)))
                {
                    Qvalue[new(state, action)] = double.NegativeInfinity;
                }
                policy[state] = (Environment.Action)Enum.GetValues(typeof(Environment.Action))
                     .GetValue(random.Next(Enum.GetValues(typeof(Environment.Action)).Length));
            }


            // Loop
            for (int episode = 0; episode < numEpisodes; episode++)
            {
                
                List<EpisodeStep> episodeSteps = GenerateEpisode(policy, maxSteps);
                double G = 0;
                // Console.WriteLine($"Episode {episode + 1} started with steps:{episodeSteps.Count} ");
                for (int t = episodeSteps.Count - 2; t >= 0; t--)
                {
                    G = gamma * G + episodeSteps[t + 1].Reward;
                    StateAction stateActionPair = new StateAction(episodeSteps[t].State, episodeSteps[t].Action);
                    // Console.WriteLine($"Prev State X:{episodeSteps[t+1].State.PlayerPosition.X} Y:{episodeSteps[t+1].State.PlayerPosition.Y} " +
                                        // $"Current State X:{episodeSteps[t].State.PlayerPosition.X} Y:{episodeSteps[t].State.PlayerPosition.Y}");
                    if (stateAction.Add(stateActionPair))
                    {
                        if (!Returns.ContainsKey(stateActionPair))
                        {
                            Returns[stateActionPair] = new List<double>();
                        }
                        Returns[stateActionPair].Add(G);
                        Qvalue[stateActionPair] = Average(Returns[stateActionPair]);
                        Environment.Action bestAction = stateActionPair.Action;
                        double bestQ = Qvalue[stateActionPair];
                        foreach (Environment.Action action in Enum.GetValues(typeof(Environment.Action)))
                        {
                            stateActionPair = new StateAction(episodeSteps[t].State, action);
                            if (Qvalue[stateActionPair] > bestQ)
                            {
                                bestQ = Qvalue[stateActionPair];
                                bestAction = action;
                            }
                        }
                        policy[episodeSteps[t].State] = bestAction;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

        private List<EpisodeStep> GenerateEpisode(Dictionary<State, Environment.Action> policy, int maxSteps)
        {
            Gamewindow.InitializeLevel();
            List<EpisodeStep> episode = new();
            State state = States[random.Next(States.Length)];
            Environment.Action action = (Environment.Action)Enum.GetValues(typeof(Environment.Action))
                     .GetValue(random.Next(Enum.GetValues(typeof(Environment.Action)).Length));
            Level.Player.Position = state.PlayerPosition;
            for (int i = 0; i < Level.Boxes.Length; i++)
            {
                Level.Boxes[i].Position = state.BoxesPosition[i];
            }
            int step = 0;
            while (!Level.LevelComplete() && step < maxSteps)
            {   
                Level.MovePlayer(action);
                Point[] boxesPos = new Point[Level.Boxes.Length];
                for (int i = 0; i < boxesPos.Length; i++)
                {
                    boxesPos[i] = Level.Boxes[i].Position;
                }
                double reward = -1;
                if (Level.LevelComplete())
                {
                    reward = 0;
                }
                episode.Add(new EpisodeStep(state, action, reward));
                state = new State(Level.Player.Position, boxesPos);
                action = policy[state];
                step++;
            }

            return episode;
        }

        public static double Average(List<double> list)
        {
            double sum = 0;
            foreach (double x in list)
                sum += x;
            return sum / list.Count;
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